// TODO does it make sense to use 16 instead of 32 for faster octree rebuilds?

use crate::world::octree::{Octree, Position};

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub struct Chunk {
    pub blocks: [BlockId; 32 * 32 * 32],
}

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            // TODO use memory pool
            blocks: [NO_BLOCK; 32 * 32 * 32],
        }
    }

    pub fn get_block_index(x: u32, y: u32, z: u32) -> usize {
        // TODO optimize by using morton codes?
        (x + y * 32 + z * 32 * 32) as usize
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        self.blocks[Chunk::get_block_index(x, y, z)]
    }

    pub fn set_block(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        self.blocks[Chunk::get_block_index(x, y, z)] = block;
    }

    pub fn build_octree(&self) -> Octree<BlockId> {
        let mut octree = Octree::new();
        octree.expand_to(32f32.log2() as u32);

        for z in 0u32..32 {
            for y in 0u32..32 {
                for x in 0u32..32 {
                    let block = self.blocks[Chunk::get_block_index(x, y, z)];
                    if block == NO_BLOCK {
                        continue;
                    }
                    octree.add_leaf(Position(x, y, z), block);
                }
            }
        }

        octree.compact();
        octree
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunk_get_and_set_blocks() {
        let mut chunk = super::Chunk::new();

        let block = chunk.get_block(10, 20, 30);
        assert_eq!(block, super::NO_BLOCK);

        chunk.set_block(10, 20, 30, 99);
        assert_eq!(chunk.blocks[10 + 20 * 32 + 30 * 32 * 32], 99);

        let block = chunk.get_block(10, 20, 30);
        assert_eq!(block, 99);
    }
}
