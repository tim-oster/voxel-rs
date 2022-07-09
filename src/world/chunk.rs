use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::world::octree::{Octree, Position};

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub struct Chunk {
    storage: Rc<ChunkStorage>,
}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            storage: Rc::new(ChunkStorage::new()),
        }
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        self.storage.get_block(x, y, z)
    }

    pub fn set_block(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        self.storage.set_block(x, y, z, block);
    }

    pub fn get_storage(&self) -> Rc<ChunkStorage> {
        Rc::clone(&self.storage)
    }
}

pub struct ChunkStorage {
    octree: RefCell<Octree<BlockId>>,
}

impl ChunkStorage {
    fn new() -> ChunkStorage {
        let octree = Octree::with_size(32f32.log2() as u32);
        ChunkStorage { octree: RefCell::new(octree) }
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        *self.octree.borrow().get_leaf(Position(x, y, z)).unwrap_or(&NO_BLOCK)
    }

    fn set_block(&self, x: u32, y: u32, z: u32, block: BlockId) {
        if block == NO_BLOCK {
            self.octree.borrow_mut().remove_leaf(Position(x, y, z));
        } else {
            self.octree.borrow_mut().add_leaf(Position(x, y, z), block);
        }
    }

    pub fn get_octree_ref(&self) -> Ref<Octree<BlockId>> {
        self.octree.borrow()
    }
}
