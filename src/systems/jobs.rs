use std::{panic, thread};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;

pub struct JobSystem {
    worker_handles: HashMap<ThreadId, JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
    currently_executing: Arc<AtomicU8>,

    queue: Arc<SegQueue<Job>>,
    prio_queue: Arc<SegQueue<Job>>,
    sleeping_threads: Arc<SegQueue<ThreadId>>,
}

struct Job {
    cancelled: Arc<AtomicBool>,
    exec: Box<dyn FnOnce() + Send>,
}

#[allow(dead_code)]
impl JobSystem {
    pub fn new(worker_count: usize) -> JobSystem {
        let mut system = JobSystem {
            worker_handles: HashMap::new(),
            is_running: Arc::new(AtomicBool::new(true)),
            currently_executing: Arc::new(AtomicU8::new(0)),
            queue: Arc::new(SegQueue::<Job>::new()),
            prio_queue: Arc::new(SegQueue::<Job>::new()),
            sleeping_threads: Arc::new(SegQueue::<ThreadId>::new()),
        };

        for _ in 0..worker_count {
            let handle = system.spawn_worker();
            system.worker_handles.insert(handle.thread().id(), handle);
        }

        system
    }

    pub fn stop(self) {
        self.is_running.store(false, Ordering::Relaxed);

        for (_, handle) in self.worker_handles {
            handle.thread().unpark();
            handle.join().unwrap();
        }
    }

    pub fn push(&self, prioritize: bool, exec: Box<dyn FnOnce() + Send>) -> JobHandle {
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job { cancelled: Arc::clone(&cancelled), exec };

        if prioritize {
            self.prio_queue.push(job);
        } else {
            self.queue.push(job);
        }

        if let Some(thread) = self.sleeping_threads.pop() {
            if let Some(handle) = self.worker_handles.get(&thread) {
                handle.thread().unpark();
            }
        }

        JobHandle { cancelled }
    }

    pub fn clear(&self) {
        while !self.queue.is_empty() { self.queue.pop(); }
        while !self.prio_queue.is_empty() { self.prio_queue.pop(); }
    }

    pub fn len(&self) -> usize {
        self.queue.len() + self.prio_queue.len()
    }

    pub fn wait_until_processed(&self) {
        loop {
            let count = self.currently_executing.load(Ordering::Relaxed);
            if count == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn spawn_worker(&self) -> JoinHandle<()> {
        let is_running = self.is_running.clone();
        let currently_executing = self.currently_executing.clone();
        let queue = self.queue.clone();
        let prio_queue = self.prio_queue.clone();
        let sleeping_threads = self.sleeping_threads.clone();

        thread::spawn(move || {
            let mut last_exec = Instant::now();

            while is_running.load(Ordering::Relaxed) {
                let job = prio_queue.pop().or_else(|| queue.pop());
                if job.is_none() {
                    if last_exec.elapsed().as_millis() > 100 {
                        sleeping_threads.push(thread::current().id());
                        thread::park();
                        last_exec = Instant::now();
                    }
                    continue;
                }
                last_exec = Instant::now();

                let job = job.unwrap();
                if job.cancelled.load(Ordering::Relaxed) {
                    continue;
                }

                currently_executing.fetch_add(1, Ordering::Relaxed);
                _ = panic::catch_unwind(AssertUnwindSafe(|| {
                    (job.exec)();
                }));
                currently_executing.fetch_sub(1, Ordering::Relaxed);
            }
        })
    }
}

pub struct JobHandle {
    cancelled: Arc<AtomicBool>,
}

impl JobHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

// TODO write tests
