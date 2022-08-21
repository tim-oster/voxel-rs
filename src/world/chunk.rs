use std::sync::{Arc, RwLock};

use crate::ChunkPos;
use crate::world::allocator::{Allocated, Allocator};
use crate::world::octree::{Octree, Position};

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub type ChunkStorage = Octree<BlockId>;

pub struct Chunk {
    pub pos: ChunkPos,

    allocator: Arc<Allocator<ChunkStorage>>,
    // TODO any other way?
    storage: Option<Arc<RwLock<Allocated<ChunkStorage>>>>,
}

impl Chunk {
    pub fn new(pos: ChunkPos, allocator: Arc<Allocator<ChunkStorage>>) -> Chunk {
        Chunk { pos, allocator, storage: None }
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        if self.storage.is_none() {
            return NO_BLOCK;
        }
        *self.storage.as_ref().unwrap().read().unwrap().get_leaf(Position(x, y, z)).unwrap_or(&NO_BLOCK)
    }

    pub fn set_block(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        if block == NO_BLOCK && self.storage.is_none() {
            return;
        }
        if self.storage.is_none() {
            let octree = self.allocator.allocate();
            self.storage = Some(Arc::new(RwLock::new(octree)));
        }
        if block == NO_BLOCK {
            self.storage.as_ref().unwrap().write().unwrap().remove_leaf(Position(x, y, z));
        } else {
            self.storage.as_ref().unwrap().write().unwrap().add_leaf(Position(x, y, z), block);
        }
    }

    pub fn get_storage(&self) -> Option<Arc<RwLock<Allocated<ChunkStorage>>>> {
        self.storage.clone()
    }
}
