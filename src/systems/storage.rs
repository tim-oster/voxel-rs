use std::sync::Arc;

use crate::world::allocator::Allocator;
use crate::world::chunk::{Chunk, ChunkPos, ChunkStorage};

pub struct Storage {
    // TODO can arc be removed?
    allocator: Arc<Allocator<ChunkStorage>>,
}

pub enum LoadError {
    NotFound,
}

pub enum StoreError {}

// TODO should storage return normal chunk objects or just their storage? right now chunks also include
//      lod data and potentially other values in the future which are outside the domain of the storage
//      system.

// TODO should storage system reference count chunks and automatically free & store them once they are
//      unused? or should other components return their chunks back to the storage layer instead?

pub struct MemoryStats {
    pub in_use: usize,
    pub allocated: usize,
}

impl Storage {
    pub fn new() -> Storage {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        Storage { allocator: Arc::new(allocator) }
    }

    pub fn load(&mut self, pos: &ChunkPos) -> Result<Chunk, LoadError> {
        Err(LoadError::NotFound)
    }

    pub fn store(&mut self, chunk: &Chunk) -> Result<(), StoreError> {
        Ok(())
    }

    pub fn new_chunk(&mut self, pos: ChunkPos) -> Chunk {
        Chunk::new(pos, Arc::clone(&self.allocator))
    }

    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            in_use: self.allocator.used_count(),
            allocated: self.allocator.allocated_count(),
        }
    }
}
