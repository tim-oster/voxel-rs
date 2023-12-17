use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::world::chunk::ChunkStorage;

/// Pool holds instances of type T as a simple implementation of a memory pool.
struct Pool<T> {
    buffer: crossbeam_queue::SegQueue<T>,
}

impl<T> Pool<T> {
    fn new() -> Pool<T> {
        Pool {
            buffer: crossbeam_queue::SegQueue::new(),
        }
    }

    fn allocate(&self) -> Option<T> {
        self.buffer.pop()
    }

    fn free(&self, instance: T) {
        self.buffer.push(instance);
    }
}

// -------------------------------------------------------------------------------------------------

pub type ConstructorFn<T> = Box<dyn Fn() -> T + Send + Sync + 'static>;
pub type ResetFn<T> = Box<dyn Fn(&mut T)>;

/// Allocator allocates new instance using `constructor` on demand, if no previous instance are
/// available for reuse. Every allocated object has an [`Allocated`] guard, that returns the
/// instance to the internal memory pool upon drop. If an old instance is reused, it will be
/// `reset` before reuse.
///
/// Allocator is thread-safe and might be wrapped inside [`Arc`] or other smart pointers.
pub struct Allocator<T> {
    pool: Arc<Pool<T>>,
    total_allocated: AtomicUsize,
    constructor: ConstructorFn<T>,
    reset: Option<ResetFn<T>>,
}

impl<T> Allocator<T> {
    pub fn new(constructor: ConstructorFn<T>, reset: Option<ResetFn<T>>) -> Allocator<T> {
        Allocator {
            pool: Arc::new(Pool::new()),
            total_allocated: AtomicUsize::new(0),
            constructor,
            reset,
        }
    }

    pub fn allocate(&self) -> Allocated<T> {
        if let Some(mut elem) = self.pool.allocate() {
            if self.reset.is_some() {
                self.reset.as_ref().unwrap()(&mut elem);
            }
            return Allocated::new(self.pool.clone(), elem);
        }
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        Allocated::new(self.pool.clone(), (self.constructor)())
    }

    pub fn allocated_count(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    pub fn used_count(&self) -> usize {
        self.allocated_count() - self.pool.buffer.len()
    }
}

unsafe impl<T: Send> Send for Allocator<T> {}

unsafe impl<T: Sync> Sync for Allocator<T> {}

pub struct Allocated<T> {
    pool: Arc<Pool<T>>,
    value: Option<T>,
}

impl<T> Allocated<T> {
    fn new(pool: Arc<Pool<T>>, value: T) -> Allocated<T> {
        Allocated { pool, value: Some(value) }
    }
}

impl<T> Drop for Allocated<T> {
    fn drop(&mut self) {
        self.pool.free(self.value.take().unwrap())
    }
}

impl<T> Deref for Allocated<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap()
    }
}

impl<T> DerefMut for Allocated<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap()
    }
}

#[cfg(test)]
mod allocator_tests {
    use std::cell::RefCell;

    use crate::world::memory::Allocator;

    /// Tests that object allocation and reset/reuse works properly.
    #[test]
    fn test() {
        let alloc = Allocator::new(
            Box::new(|| RefCell::new(0)),
            Some(Box::new(|cell| *cell.borrow_mut() = 0)),
        );

        // assert constructor
        let mut instance = alloc.allocate();
        *instance.borrow_mut() = 5;
        assert_eq!(*instance.borrow(), 5);
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 1);
        drop(instance);

        // assert return to pool
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 0);

        // assert reuse and reset function
        let mut instance = alloc.allocate();
        assert_eq!(*instance.borrow(), 0);
        assert_eq!(alloc.allocated_count(), 1);
        assert_eq!(alloc.used_count(), 1);
    }
}

// -------------------------------------------------------------------------------------------------

/// ChunkStorageAllocator is an allocator for ChunkStorage objects.
pub struct ChunkStorageAllocator {
    allocator: Allocator<ChunkStorage>,
}

impl ChunkStorageAllocator {
    pub fn new() -> ChunkStorageAllocator {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(5)), // log2(32) = 5
            Some(Box::new(|storage| {
                storage.reset();
                storage.expand_to(5);
            })),
        );
        ChunkStorageAllocator { allocator }
    }
}

impl Deref for ChunkStorageAllocator {
    type Target = Allocator<ChunkStorage>;

    fn deref(&self) -> &Self::Target {
        &self.allocator
    }
}

#[cfg(test)]
mod chunk_storage_allocator_tests {
    use crate::world::memory::ChunkStorageAllocator;

    /// Tests that newly allocated and reused storage objects always have a depth of 5 blocks to prevent visual voxel
    /// scale issues in the world.
    #[test]
    fn new() {
        let alloc = ChunkStorageAllocator::new();

        // new allocation
        let storage = alloc.allocate();
        assert_eq!(storage.depth(), 5);
        assert_eq!(alloc.allocated_count(), 1);

        drop(storage);

        // reused allocation
        let storage = alloc.allocate();
        assert_eq!(storage.depth(), 5);
        assert_eq!(alloc.allocated_count(), 1);
    }
}
