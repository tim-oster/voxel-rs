use std::{panic, thread};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;

use crate::world::chunk::ChunkPos;

/// JobSystem manages a configurable amount of worker threads to distribute jobs to. Each job
/// can produce a result which can be retrieved on the job producer's side.
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

pub struct JobHandle {
    cancelled: Arc<AtomicBool>,
}

impl JobHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
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

    /// stop signals all worker threads to stop and joins them. Currently processed jobs are not
    /// cancelled and will cause this method to block.
    pub fn stop(self) {
        self.is_running.store(false, Ordering::Relaxed);

        for (_, handle) in self.worker_handles {
            handle.thread().unpark();
            handle.join().unwrap();
        }
    }

    /// push enqueues a new job. Setting `prioritize` to true, will queue it up in a separate queue,
    /// which is drained before the normal, un-prioritized queue.
    pub fn push<Fn: FnOnce() + Send + 'static>(&self, prioritize: bool, exec: Fn) -> JobHandle {
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            cancelled: Arc::clone(&cancelled),
            exec: Box::new(exec),
        };

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

    /// clear discards all queued up jobs.
    pub fn clear(&self) {
        while !self.queue.is_empty() { self.queue.pop(); }
        while !self.prio_queue.is_empty() { self.prio_queue.pop(); }
    }

    /// len returns the current amount of all queued up jobs.
    pub fn len(&self) -> usize {
        self.queue.len() + self.prio_queue.len()
    }

    /// wait_until_empty_and_processed spin-loops until all queued elements have been picked up by
    /// a worker thread and all threads have finished processing their jobs.
    pub fn wait_until_empty_and_processed(&self) {
        while self.len() > 0 {
            thread::sleep(Duration::from_millis(50));
        }
        self.wait_until_processed();
    }

    /// wait_until_processed spin-loops until all worker threads have finished processing their
    /// jobs.
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

#[cfg(test)]
mod job_system_tests {
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::thread;
    use std::time::Duration;

    use crate::systems::jobs::JobSystem;

    pub fn wait(atomic: Arc<AtomicBool>) {
        while !atomic.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Tests that prioritization of jobs works.
    #[test]
    fn push_prio_and_normal() {
        // only one worker to process one at a time
        let js = JobSystem::new(1);

        // push job to allow for other jobs to be enqueued
        let signal = Arc::new(AtomicBool::new(false));
        let s0 = signal.clone();
        js.push(false, move || {
            s0.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(250))
        });
        wait(signal);

        // enqueue prioritized and normal job
        let list = Arc::new(Mutex::new(Vec::new()));
        let l0 = list.clone();
        let l1 = list.clone();
        js.push(false, move || l0.lock().unwrap().push("normal"));
        js.push(true, move || l1.lock().unwrap().push("prio"));

        js.wait_until_empty_and_processed();
        js.stop();

        // assert that prioritized job was processed first
        let list = Arc::try_unwrap(list).unwrap().into_inner().unwrap();
        assert_eq!(list, vec!["prio", "normal"]);
    }

    /// Tests that clear discards all pending jobs.
    #[test]
    fn clear() {
        // only one worker to process one at a time
        let js = JobSystem::new(1);

        // push job to allow for other jobs to be enqueued
        let signal = Arc::new(AtomicBool::new(false));
        let s0 = signal.clone();
        js.push(false, move || {
            s0.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(250))
        });
        wait(signal);

        let counter = Arc::new(AtomicI32::new(0));
        for i in 0..5 {
            let c = counter.clone();
            js.push(false, move || { c.fetch_add(1, Ordering::Relaxed); });
        }

        // clear & sleep to ensure that nothing is processed
        js.clear();
        thread::sleep(Duration::from_millis(500));

        js.stop();

        // assert that no job was processed
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Tests that cancelling already queued jobs works.
    #[test]
    fn cancelling_queued_job() {
        // only one worker to process one at a time
        let js = JobSystem::new(1);

        // push job to allow for other jobs to be enqueued
        let signal = Arc::new(AtomicBool::new(false));
        let s0 = signal.clone();
        js.push(false, move || {
            s0.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(250))
        });
        wait(signal);

        let list = Arc::new(Mutex::new(Vec::new()));
        let handle = js.push(false, {
            let list = list.clone();
            move || list.lock().unwrap().push("cancelled")
        });
        js.push(false, {
            let list = list.clone();
            move || list.lock().unwrap().push("normal")
        });

        // cancel first job
        handle.cancel();

        js.wait_until_empty_and_processed();
        js.stop();

        // assert that first job was not processed first
        let list = Arc::try_unwrap(list).unwrap().into_inner().unwrap();
        assert_eq!(list, vec!["normal"]);
    }
}

/// ChunkProcessor is a decorator for [`JobSystem`]. It allows de-/queueing jobs per [`ChunkPos`].
/// Enqueuing multiple jobs for the same position will override previously enqueued jobs.
/// Results contain their original chunk position in addition to their actual value.
pub struct ChunkProcessor<T> {
    job_system: Rc<JobSystem>,
    tx: Sender<ChunkResult<T>>,
    rx: Receiver<ChunkResult<T>>,
    chunk_jobs: HashMap<ChunkPos, JobHandle>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ChunkResult<T> {
    pub pos: ChunkPos,
    pub value: T,
}

impl<T: Send + 'static> ChunkProcessor<T> {
    pub fn new(job_system: Rc<JobSystem>) -> ChunkProcessor<T> {
        let (tx, rx) = mpsc::channel();
        ChunkProcessor { job_system, tx, rx, chunk_jobs: HashMap::new() }
    }

    pub fn enqueue<Fn: FnOnce() -> T + Send + 'static>(&mut self, pos: ChunkPos, prioritize: bool, exec: Fn) {
        self.dequeue(&pos);

        let tx = self.tx.clone();

        let handle = self.job_system.push(prioritize, move || {
            let result = exec();
            let result = ChunkResult { pos, value: result };
            tx.send(result).unwrap();
        });
        self.chunk_jobs.insert(pos, handle);
    }

    pub fn dequeue(&mut self, pos: &ChunkPos) {
        if let Some(handle) = self.chunk_jobs.remove(pos) {
            handle.cancel();
        }
    }

    pub fn get_results(&mut self, limit: u32) -> Vec<ChunkResult<T>> {
        let mut results = Vec::new();

        for _ in 0..limit {
            if let Ok(result) = self.rx.try_recv() {
                self.chunk_jobs.remove(&result.pos);
                results.push(result);
            } else {
                break;
            }
        }

        results
    }
}

#[cfg(test)]
mod chunk_processor_tests {
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    use crate::systems::jobs::{ChunkProcessor, ChunkResult, JobSystem};
    use crate::systems::jobs::job_system_tests::wait;
    use crate::world::chunk::ChunkPos;

    /// Tests that enqueue & dequeue work for all possible scenarios.
    #[test]
    fn enqueue() {
        // only one worker to process one at a time
        let js = Rc::new(JobSystem::new(1));
        let mut cp = ChunkProcessor::new(js.clone());

        // push job to allow for other jobs to be enqueued
        let signal = Arc::new(AtomicBool::new(false));
        let s0 = signal.clone();
        cp.enqueue(ChunkPos::new(0, 0, 0), false, move || {
            s0.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(250));
            "waiter"
        });
        wait(signal);

        // enqueue same chunk pos twice to test override
        cp.enqueue(ChunkPos::new(1, 0, 0), false, || "first");
        cp.enqueue(ChunkPos::new(1, 0, 0), false, || "first override");

        // test prioritize enqueue
        cp.enqueue(ChunkPos::new(2, 0, 0), true, || "prio");

        // test dequeue
        cp.enqueue(ChunkPos::new(3, 0, 0), false, || "second");
        cp.dequeue(&ChunkPos::new(3, 0, 0));

        // test normal
        cp.enqueue(ChunkPos::new(4, 0, 0), false, || "third");

        js.wait_until_empty_and_processed();

        // get no results for limit=0
        let results = cp.get_results(0);
        assert!(results.is_empty());

        // get first result
        let results = cp.get_results(1);
        assert_eq!(results, vec![
            ChunkResult { pos: ChunkPos::new(0, 0, 0), value: "waiter" }
        ]);
        assert!(!cp.chunk_jobs.is_empty());

        // get remaining results
        let results = cp.get_results(100);
        assert_eq!(results, vec![
            ChunkResult { pos: ChunkPos::new(2, 0, 0), value: "prio" },
            ChunkResult { pos: ChunkPos::new(1, 0, 0), value: "first override" },
            ChunkResult { pos: ChunkPos::new(4, 0, 0), value: "third" },
        ]);
        assert!(cp.chunk_jobs.is_empty());

        // return immediately as no more results exist
        let results = cp.get_results(100);
        assert!(results.is_empty());
    }
}
