use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::sync::mpsc::{Receiver, Sender};

use crate::systems::jobs::{JobHandle, JobSystemHandle};
use crate::world::chunk::{Chunk, ChunkPos};
use crate::world::generator;

pub struct Generator<'js> {
    jobs: JobSystemHandle<'js>,
    gen: Arc<generator::Generator>,
    tx: Sender<Chunk>,
    rx: Receiver<Chunk>,
    chunk_jobs: HashMap<ChunkPos, JobHandle>,
}

impl<'js> Generator<'js> {
    pub fn new(jobs: JobSystemHandle<'js>, seed: u32, cfg: generator::Config) -> Generator {
        let (tx, rx) = mpsc::channel::<Chunk>();
        Generator {
            jobs,
            gen: Arc::new(generator::Generator::new(seed, cfg)),
            tx,
            rx,
            chunk_jobs: HashMap::new(),
        }
    }

    /// Takes ownership of the chunk and schedules a job for async generation of the chunk's
    /// content. Use get_generated_chunks to retrieve generated chunks.
    pub fn enqueue_chunk(&mut self, chunk: Chunk) {
        let pos = chunk.pos;
        self.dequeue_chunk(&pos);

        let mut chunk = chunk;
        let gen = Arc::clone(&self.gen);
        let tx = self.tx.clone();

        let handle = self.jobs.push(false, Box::new(move || {
            gen.generate_chunk(&mut chunk);
            tx.send(chunk).unwrap();
        }));
        self.chunk_jobs.insert(pos, handle);
    }

    /// Allows removing the enqueued generation job, if it has not started yet. This will
    /// also drop the chunk instance.
    pub fn dequeue_chunk(&mut self, pos: &ChunkPos) {
        if let Some(handle) = self.chunk_jobs.remove(pos) {
            handle.cancel();
        }
    }

    /// Returns up to limit chunks from finished generation jobs. This is a non-blocking operation
    /// so it might return 0 chunks immediately. Limit must be greater than 0.
    pub fn get_generated_chunks(&mut self, limit: u32) -> Vec<Chunk> {
        assert!(limit > 0);

        let mut chunks = Vec::new();
        for _ in 0..limit {
            let result = self.rx.try_recv();
            if result.is_err() {
                break;
            }
            let chunk = result.unwrap();
            self.chunk_jobs.remove(&chunk.pos);
            chunks.push(chunk);
        }
        chunks
    }
}

// TODO write tests
