use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type ConstructorFn<T> = Box<dyn Fn() -> T + Send + Sync + 'static>;
pub type ResetFn<T> = Box<dyn Fn(&mut T)>;

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
                (self.reset.as_ref().unwrap())(&mut elem);
            }
            return Allocated::new(self.pool.clone(), elem);
        }
        self.total_allocated.fetch_add(1, Ordering::SeqCst);
        Allocated::new(self.pool.clone(), (self.constructor)())
    }

    pub fn allocated_count(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst)
    }

    pub fn used_count(&self) -> usize {
        self.allocated_count() - self.pool.buffer.len()
    }
}

unsafe impl<T: Send> Send for Allocator<T> {}

unsafe impl<T: Sync> Sync for Allocator<T> {}

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
