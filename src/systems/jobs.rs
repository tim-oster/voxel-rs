use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::{JoinHandle, ThreadId};
use std::time::Instant;

use crossbeam_queue::SegQueue;

pub struct JobSystem {
    worker_handles: HashMap<ThreadId, JoinHandle<()>>,
    is_running: Arc<AtomicBool>,

    queue: Arc<SegQueue<Job>>,
    prio_queue: Arc<SegQueue<Job>>,
    sleeping_threads: Arc<SegQueue<ThreadId>>,
}

struct Job {
    exec: Box<dyn FnOnce() + Send>,
}

impl JobSystem {
    pub fn new(worker_count: usize) -> JobSystem {
        let mut system = JobSystem {
            worker_handles: HashMap::new(),
            is_running: Arc::new(AtomicBool::new(true)),
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

    pub fn push(&self, prioritize: bool, exec: Box<dyn FnOnce() + Send>) {
        if prioritize {
            self.prio_queue.push(Job { exec });
        } else {
            self.queue.push(Job { exec });
        }

        if let Some(thread) = self.sleeping_threads.pop() {
            if let Some(handle) = self.worker_handles.get(&thread) {
                handle.thread().unpark();
            }
        }
    }

    pub fn clear(&self) {
        while !self.queue.is_empty() { self.queue.pop(); }
        while !self.prio_queue.is_empty() { self.prio_queue.pop(); }
    }

    pub fn len(&self) -> usize {
        self.queue.len() + self.prio_queue.len()
    }

    fn spawn_worker(&self) -> JoinHandle<()> {
        let is_running = self.is_running.clone();
        let queue = self.queue.clone();
        let prio_queue = self.prio_queue.clone();
        let sleeping_threads = self.sleeping_threads.clone();

        thread::spawn(move || {
            let mut last_exec = Instant::now();

            while is_running.load(Ordering::Relaxed) {
                let job = prio_queue.pop().or_else(|| queue.pop());
                if job.is_none() {
                    if last_exec.elapsed().as_millis() > 100 {
                        sleeping_threads.push(std::thread::current().id());
                        std::thread::park();
                        last_exec = Instant::now();
                    }
                    continue;
                }
                last_exec = Instant::now();
                (job.unwrap().exec)();
            }
        })
    }
}

