use std::rc::Rc;
use std::sync::Arc;

use crate::systems::jobs::{ChunkProcessor, JobSystem};
use crate::world::chunk::{Chunk, ChunkPos, ChunkStorageAllocator};

pub trait ChunkGenerator {
    /// Returns true if the chunk needs generation, otherwise the chunk will be skipped and no
    /// memory will be allocated.
    fn is_interested_in(&self, pos: &ChunkPos) -> bool;

    /// Modifies the chunk according to the generator's logic.
    fn generate_chunk(&self, chunk: &mut Chunk);
}

pub struct Generator {
    processor: ChunkProcessor<Option<Chunk>>,
    storage_allocator: Arc<ChunkStorageAllocator>,
    gen: Arc<dyn ChunkGenerator + Send + Sync + 'static>,
}

impl Generator {
    pub fn new(job_system: Rc<JobSystem>, storage_allocator: Arc<ChunkStorageAllocator>, chunk_generator: impl ChunkGenerator + Send + Sync + 'static) -> Self {
        Self {
            processor: ChunkProcessor::new(job_system),
            storage_allocator,
            gen: Arc::new(chunk_generator),
        }
    }

    /// Enqueues a position for a chunk to be generated. The `lod` will be kept for the generated
    /// chunk. If the underlying chunk generator does not indicate interest in the chunk, no chunk
    /// will be allocated! Use `get_generated_chunks` to retrieve generated chunks.
    pub fn enqueue_chunk(&mut self, pos: ChunkPos, lod: u8) {
        let alloc = self.storage_allocator.clone();
        let gen = Arc::clone(&self.gen);

        self.processor.enqueue(pos, false, move || {
            if !gen.is_interested_in(&pos) {
                return None;
            }

            let mut chunk = Chunk::new(pos, lod, alloc.allocate());
            gen.generate_chunk(&mut chunk);

            Some(chunk)
        });
    }

    /// Allows removing the enqueued generation job, if it has not started yet. This will
    /// also drop the chunk instance.
    pub fn dequeue_chunk(&mut self, pos: &ChunkPos) {
        self.processor.dequeue(pos);
    }

    /// Returns up to limit chunks from finished generation jobs. This is a non-blocking operation
    /// so it might return 0 chunks immediately. Note that not every enqueued chunk position must
    /// return a job, depending on the chunk generator implementation.
    pub fn get_generated_chunks(&mut self, limit: u32) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        for result in self.processor.get_results(limit) {
            if let Some(chunk) = result.value {
                chunks.push(chunk);
            }
        }
        chunks
    }

    /// Returns if the generator still has in-work chunks or if there are unconsumed chunks in the
    /// buffer.
    pub fn has_pending_jobs(&self) -> bool {
        self.processor.has_pending()
    }
}
