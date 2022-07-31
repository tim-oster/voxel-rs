use std::sync::{Arc, RwLock};

use crate::ChunkPos;
use crate::world::octree::{Octree, Position};

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub type ChunkStorage = Octree<BlockId>;

pub struct Chunk {
    pub pos: ChunkPos,
    // TODO any other way?
    storage: Arc<RwLock<ChunkStorage>>,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Chunk {
        let octree = Octree::with_size(32f32.log2() as u32);
        Chunk {
            pos,
            storage: Arc::new(RwLock::new(octree)),
        }
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        *self.storage.read().unwrap().get_leaf(Position(x, y, z)).unwrap_or(&NO_BLOCK)
    }

    pub fn set_block(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        if block == NO_BLOCK {
            self.storage.write().unwrap().remove_leaf(Position(x, y, z));
        } else {
            self.storage.write().unwrap().add_leaf(Position(x, y, z), block);
        }
    }

    pub fn get_storage(&self) -> Arc<RwLock<ChunkStorage>> {
        self.storage.clone()
    }
}
