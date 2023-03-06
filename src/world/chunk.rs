use std::sync::{Arc, RwLock};

use crate::world::allocator::{Allocated, Allocator};
use crate::world::octree::{Octree, Position};

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub type ChunkStorage = Octree<BlockId>;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[allow(dead_code)]
impl ChunkPos {
    pub fn new(x: i32, y: i32, z: i32) -> ChunkPos {
        ChunkPos { x, y, z }
    }

    pub fn from_block_pos(x: i32, y: i32, z: i32) -> ChunkPos {
        ChunkPos { x: x >> 5, y: y >> 5, z: z >> 5 }
    }

    pub fn dst_sq(&self, other: &ChunkPos) -> f32 {
        let dx = (other.x - self.x) as f32;
        let dy = (other.y - self.y) as f32;
        let dz = (other.z - self.z) as f32;
        dx * dx + dy * dy + dz * dz
    }

    pub fn dst_2d_sq(&self, other: &ChunkPos) -> f32 {
        let dx = (other.x - self.x) as f32;
        let dz = (other.z - self.z) as f32;
        dx * dx + dz * dz
    }
}

pub struct Chunk {
    // TODO should this be exposed? it must be read-only
    pub pos: ChunkPos,
    pub lod: u8,

    // TODO remove allocator once block placement is deferred by using a block change queue on system level
    allocator: Arc<Allocator<ChunkStorage>>,
    // TODO any other way?
    storage: Option<Arc<RwLock<Allocated<ChunkStorage>>>>,
}

impl Chunk {
    pub fn new(pos: ChunkPos, allocator: Arc<Allocator<ChunkStorage>>) -> Chunk {
        Chunk { pos, lod: 0, allocator, storage: None }
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
