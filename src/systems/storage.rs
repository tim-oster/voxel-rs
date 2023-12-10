use crate::world::chunk::{Chunk, ChunkPos};

pub struct Storage {}

pub enum LoadError {
    NotFound,
}

pub enum StoreError {}

// TODO should storage return normal chunk objects or just their storage? right now chunks also include
//      lod data and potentially other values in the future which are outside the domain of the storage
//      system.

// TODO should storage system reference count chunks and automatically free & store them once they are
//      unused? or should other components return their chunks back to the storage layer instead?

impl Storage {
    pub fn new() -> Storage {
        Storage {}
    }

    pub fn load(&mut self, _pos: &ChunkPos) -> Result<Chunk, LoadError> {
        Err(LoadError::NotFound)
    }

    pub fn store(&mut self, _chunk: &Chunk) -> Result<(), StoreError> {
        Ok(())
    }
}
