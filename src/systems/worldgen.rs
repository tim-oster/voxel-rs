use std::rc::Rc;
use std::sync::Arc;

use crate::systems::jobs::{ChunkProcessor, JobSystem};
use crate::world::chunk::{Chunk, ChunkPos};

pub trait ChunkGenerator {
    fn generate_chunk(&self, chunk: &mut Chunk);
}

// TODO is this system required or can it be moved to game?

pub struct Generator {
    processor: ChunkProcessor<Chunk>,
    gen: Arc<dyn ChunkGenerator + Send + Sync + 'static>,
}

impl Generator {
    pub fn new(job_system: Rc<JobSystem>, chunk_generator: impl ChunkGenerator + Send + Sync + 'static) -> Generator {
        Generator {
            processor: ChunkProcessor::new(job_system),
            gen: Arc::new(chunk_generator),
        }
    }

    /// Takes ownership of the chunk and schedules a job for async generation of the chunk's
    /// content. Use get_generated_chunks to retrieve generated chunks.
    pub fn enqueue_chunk(&mut self, chunk: Chunk) {
        let pos = chunk.pos;
        let mut chunk = chunk;
        let gen = Arc::clone(&self.gen);

        self.processor.enqueue(pos, false, move || {
            gen.generate_chunk(&mut chunk);
            chunk
        })
    }

    /// Allows removing the enqueued generation job, if it has not started yet. This will
    /// also drop the chunk instance.
    pub fn dequeue_chunk(&mut self, pos: &ChunkPos) {
        self.processor.dequeue(pos);
    }

    /// Returns up to limit chunks from finished generation jobs. This is a non-blocking operation
    /// so it might return 0 chunks immediately. Limit must be greater than 0.
    pub fn get_generated_chunks(&mut self, limit: u32) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        for result in self.processor.get_results(50) {
            chunks.push(result.value);
        }
        chunks
    }
}

// TODO write tests
