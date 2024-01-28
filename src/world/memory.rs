use std::alloc::{Allocator, AllocError, Global, GlobalAlloc, Layout, System};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type ConstructorFn<T, A> = Box<dyn Fn(A) -> T + Send + Sync + 'static>;
pub type ResetFn<T> = Box<dyn Fn(&mut T)>;

/// Pool allocates new instances using `constructor` on demand, if no previous instance is
/// available for reuse. Every allocated object has an [`Pooled`] guard, that returns the
/// instance to the internal memory pool upon drop. If an old instance is reused, it will be
/// `reset` before reuse.
///
/// Allocator is thread-safe and might be wrapped inside [`Arc`] or similar smart pointers.
pub struct Pool<T, A: Allocator = Global> {
    alloc: A,
    pool: Arc<crossbeam_queue::SegQueue<T>>,
    total_allocated: AtomicUsize,
    constructor: ConstructorFn<T, A>,
    reset: Option<ResetFn<T>>,
}

impl<T> Pool<T> {
    pub fn new(constructor: ConstructorFn<T, Global>, reset: Option<ResetFn<T>>) -> Pool<T> {
        Self::new_in(constructor, reset, Global)
    }
}

impl<T, A: Allocator + Clone> Pool<T, A> {
    pub fn new_in(constructor: ConstructorFn<T, A>, reset: Option<ResetFn<T>>, alloc: A) -> Pool<T, A> {
        Pool {
            alloc,
            pool: Arc::new(crossbeam_queue::SegQueue::new()),
            total_allocated: AtomicUsize::new(0),
            constructor,
            reset,
        }
    }

    /// Returns either a reused & reset instance from the pool, or creates a new instance.
    pub fn allocate(&self) -> Pooled<T> {
        if let Some(mut elem) = self.pool.pop() {
            if self.reset.is_some() {
                self.reset.as_ref().unwrap()(&mut elem);
            }
            return Pooled::new(Arc::clone(&self.pool), elem);
        }
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        Pooled::new(Arc::clone(&self.pool), (self.constructor)(self.alloc.clone()))
    }

    /// Returns the total number of instances created by this pool, both in-use and reusable.
    pub fn allocated_count(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Returns the number of instances that are currently owned by some component.
    pub fn used_count(&self) -> usize {
        self.allocated_count() - self.pool.len()
    }

    /// Drops all currently pooled instances.
    pub fn clear(&self) {
        while !self.pool.is_empty() {
            self.pool.pop();
        }
    }
}

unsafe impl<T: Send, A: Allocator> Send for Pool<T, A> {}

unsafe impl<T: Sync, A: Allocator> Sync for Pool<T, A> {}

pub trait AllocatorStats {
    fn allocated_bytes(&self) -> usize;
}

impl<T, A: Allocator + AllocatorStats> Pool<T, A> {
    pub fn allocated_bytes(&self) -> usize {
        self.alloc.allocated_bytes()
    }
}

// -------------------------------------------------------------------------------------------------

/// Pooled ownership return their value back to the pool once dropped.
pub struct Pooled<T> {
    pool: Arc<crossbeam_queue::SegQueue<T>>,
    value: Option<T>,
}

impl<T> Pooled<T> {
    fn new(pool: Arc<crossbeam_queue::SegQueue<T>>, value: T) -> Pooled<T> {
        Pooled { pool, value: Some(value) }
    }
}

impl<T> Drop for Pooled<T> {
    fn drop(&mut self) {
        self.pool.push(self.value.take().unwrap())
    }
}

impl<T> Deref for Pooled<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap()
    }
}

impl<T> DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap()
    }
}

#[cfg(test)]
mod pool_tests {
    use std::cell::RefCell;

    use crate::world::memory::Pool;

    /// Tests that object allocation and reset/reuse works properly.
    #[test]
    fn test() {
        let alloc = Pool::new(
            Box::new(|_| RefCell::new(0)),
            Some(Box::new(|cell| *cell.borrow_mut() = 0)),
        );

        // assert constructor
        let instance = alloc.allocate();
        *instance.borrow_mut() = 5;
        assert_eq!(*instance.borrow(), 5);
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 1);
        drop(instance);

        // assert return to pool
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 0);

        // assert reuse and reset function
        let instance = alloc.allocate();
        assert_eq!(*instance.borrow(), 0);
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 1);
    }
}

// -------------------------------------------------------------------------------------------------

/// StatsAllocator is a custom rust allocator that keeps track of how much memory it allocated and freed. Its allocation
/// behaviour is implemented by [Global].
///
/// It is safe to Clone, all clones of the an instance contribute to the same metric.
/// It is safe to use across multiple threads. It uses an AtomicUsize to avoid race conditions.
#[derive(Clone, Default, Debug)]
pub struct StatsAllocator {
    allocated_bytes: Arc<AtomicUsize>,
}

impl StatsAllocator {
    pub fn new() -> StatsAllocator {
        Self {
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
}

unsafe impl Allocator for StatsAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocated_bytes.fetch_add(layout.size(), Ordering::Relaxed);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.allocated_bytes.fetch_sub(layout.size(), Ordering::Relaxed);
        Global.deallocate(ptr, layout)
    }
}

impl AllocatorStats for StatsAllocator {
    fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

// -------------------------------------------------------------------------------------------------

/// GlobalStatsAllocator is identical to StatsAllocator but implements the GlobalAlloc trait, allowing it to be used
/// as a replacement allocator for the whole rust runtime.
pub struct GlobalStatsAllocator {
    pub allocated_bytes: AtomicUsize,
}

unsafe impl GlobalAlloc for GlobalStatsAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated_bytes.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.allocated_bytes.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}
