use std::{panic, thread};
use std::cell::RefCell;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::world::chunk::ChunkPos;

/// `JobSystem` manages a configurable amount of worker threads to distribute jobs to.
pub struct JobSystem {
    worker_handles: FxHashMap<ThreadId, JoinHandle<()>>,
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
    pub fn new(worker_count: usize) -> Self {
        let mut system = Self {
            worker_handles: FxHashMap::default(),
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

    /// Signals all worker threads to stop and joins them. Currently processed jobs are not
    /// cancelled and will cause this method to block.
    pub fn stop(self) {
        self.is_running.store(false, Ordering::Relaxed);

        for (_, handle) in self.worker_handles {
            handle.thread().unpark();
            handle.join().unwrap();
        }
    }

    /// Enqueues a new job. Setting `prioritize` to true, will queue it up in a separate queue,
    /// which is drained before the normal, un-prioritized queue.
    pub fn push<Fn: FnOnce() + Send + 'static>(&self, prioritize: bool, exec: Fn) -> JobHandle {
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            cancelled: cancelled.clone(),
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

    /// Discards all queued up jobs and marks them as cancelled.
    pub fn clear(&self) {
        let mut handles = Vec::new();
        while !self.queue.is_empty() {
            handles.push(self.queue.pop());
        }
        while !self.prio_queue.is_empty() {
            handles.push(self.prio_queue.pop());
        }
        for h in handles.into_iter().flatten() {
            h.cancelled.store(true, Ordering::Relaxed);
        }
    }

    /// Returns the current amount of all queued up jobs.
    pub fn queue_len(&self) -> usize {
        self.queue.len() + self.prio_queue.len()
    }

    /// Spin-loops until all queued elements have been picked up by
    /// a worker thread and all threads have finished processing their jobs.
    pub fn wait_until_empty_and_processed(&self) {
        while self.queue_len() > 0 {
            thread::sleep(Duration::from_millis(50));
        }
        self.wait_until_processed();
    }

    /// Spin-loops until all worker threads have finished processing their jobs.
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
        while !atomic.load(Ordering::Relaxed) {
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
            s0.store(true, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(250));
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
            s0.store(true, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(250));
        });
        wait(signal);

        let counter = Arc::new(AtomicI32::new(0));
        for _ in 0..5 {
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
            s0.store(true, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(250));
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

/// `ChunkProcessor` is a decorator for [`JobSystem`]. It allows de-/queueing jobs per [`ChunkPos`].
/// Enqueuing multiple jobs for the same position will override previously enqueued jobs.
/// Each job can produce a result of type [`T`] which is retrievable on the job producer's side.
pub struct ChunkProcessor<T> {
    job_system: Rc<JobSystem>,
    tx: Sender<ChunkResult<T>>,
    rx: Receiver<ChunkResult<T>>,
    chunk_jobs: RefCell<FxHashMap<ChunkPos, JobHandle>>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ChunkResult<T> {
    pub pos: ChunkPos,
    pub value: T,
}

impl<T: Send + 'static> ChunkProcessor<T> {
    pub fn new(job_system: Rc<JobSystem>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self { job_system, tx, rx, chunk_jobs: RefCell::new(FxHashMap::default()) }
    }

    /// Enqueues a new chunk job in either the prioritized or normal queue. If there is already
    /// a queued up job for this position in the queue, that job is cancelled. Note that it is still
    /// possible for both jobs to return a result for that position in case the first chunk's job
    /// is currently being processed when the second chunk is enqueued.
    pub fn enqueue<Fn: FnOnce() -> T + Send + 'static>(&mut self, pos: ChunkPos, prioritize: bool, exec: Fn) {
        self.dequeue(&pos);

        let tx = self.tx.clone();

        let handle = self.job_system.push(prioritize, move || {
            let result = exec();
            let result = ChunkResult { pos, value: result };
            tx.send(result).unwrap();
        });
        self.chunk_jobs.borrow_mut().insert(pos, handle);
    }

    /// Dequeues a chunk position's job, if any exists. Note that a dequeue whilst processing of the
    /// chunk has no effect. The result is still produced and consumable.
    /// This function should be used to prevent unnecessary work, but the caller must double check
    /// if the result is correct and usable.
    pub fn dequeue(&mut self, pos: &ChunkPos) {
        if let Some(handle) = self.chunk_jobs.borrow_mut().remove(pos) {
            handle.cancel();
        }
    }

    /// Returns all produced results up to `limit`. This is a non-blocking operation
    /// so it might return 0 results immediately.
    pub fn get_results(&mut self, limit: u32) -> Vec<ChunkResult<T>> {
        let mut results = Vec::new();

        for _ in 0..limit {
            if let Ok(result) = self.rx.try_recv() {
                self.chunk_jobs.borrow_mut().remove(&result.pos);
                results.push(result);
            } else {
                break;
            }
        }

        results
    }

    /// Returns true if there are any jobs queued or unconsumed (via [`ChunkProcessor::get_results`])
    /// for any chunk position.
    pub fn has_pending(&self) -> bool {
        // NOTE: uses interior mutability to not expose `has_pending` as `&mut self` to hide
        // cleanup logic from the API.

        // fast path: jobs map is empty
        if self.chunk_jobs.borrow().is_empty() {
            return false;
        }

        // If not empty: iterate through all jobs and check if they have been cancelled. Interrupt
        // at the first non-cancelled job as that indicates that there is at least one pending job
        // in the queue for either processing or consuming.
        let mut set = FxHashSet::default();
        for (pos, handle) in self.chunk_jobs.borrow().iter() {
            let cancelled = handle.cancelled.load(Ordering::Relaxed);
            if !cancelled {
                break;
            }
            set.insert(*pos);
        }

        // Remove all cancelled jobs, if any.
        let mut jobs = self.chunk_jobs.borrow_mut();
        for pos in set {
            jobs.remove(&pos);
        }

        // If, after cleaning up cancelled jobs, there are still jobs in the map, then there is
        // still work to be done.
        !jobs.is_empty()
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
            s0.store(true, Ordering::Relaxed);
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
        assert!(cp.has_pending());

        // get remaining results
        let results = cp.get_results(100);
        assert_eq!(results, vec![
            ChunkResult { pos: ChunkPos::new(2, 0, 0), value: "prio" },
            ChunkResult { pos: ChunkPos::new(1, 0, 0), value: "first override" },
            ChunkResult { pos: ChunkPos::new(4, 0, 0), value: "third" },
        ]);
        assert!(!cp.has_pending());

        // return immediately as no more results exist
        let results = cp.get_results(100);
        assert!(results.is_empty());
    }

    /// Tests that `has_pending` handles externally cancelled jobs properly.
    #[test]
    fn has_pending() {
        // only one worker to process one at a time
        let js = Rc::new(JobSystem::new(1));
        let mut cp = ChunkProcessor::new(js.clone());

        // push one job to block other jobs from being processed, that waits until the signal is set
        let signal_started = Arc::new(AtomicBool::new(false));
        let signal_queued = Arc::new(AtomicBool::new(false));
        let s0 = signal_started.clone();
        let s1 = signal_queued.clone();
        cp.enqueue(ChunkPos::new(0, 0, 0), false, move || {
            s0.store(true, Ordering::Relaxed);
            wait(s1);
            "waiter"
        });
        wait(signal_started);

        // enqueue second job that is immediately cancelled and hence never processed
        cp.enqueue(ChunkPos::new(1, 0, 0), false, || "cancelled");
        js.clear();

        // signal pending job to finish and wait
        signal_queued.store(true, Ordering::Relaxed);
        js.wait_until_processed();

        // ensure that two jobs were still in jobs map
        assert_eq!(cp.chunk_jobs.borrow().len(), 2);
        assert!(cp.has_pending());

        // assert waiter was processed
        let results = cp.get_results(2);
        assert_eq!(results, vec![ChunkResult { pos: ChunkPos::new(0, 0, 0), value: "waiter" }]);

        // make sure that no job is pending anymore and that the cancelled job was removed
        assert!(!cp.has_pending());
        assert_eq!(cp.chunk_jobs.borrow().len(), 0);
    }
}
