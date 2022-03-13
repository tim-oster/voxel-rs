// TODO does it make sense to use 16 instead of 32 for faster octree rebuilds?

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub struct Chunk {
    pub blocks: [BlockId; 32 * 32 * 32],
}

// TODO make positions unsigned?

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            blocks: [NO_BLOCK; 32 * 32 * 32],
        }
    }

    pub fn get_block_index(x: i32, y: i32, z: i32) -> usize {
        // TODO optimize by using morton codes?
        (x + y * 32 + z * 32 * 32) as usize
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.blocks[Chunk::get_block_index(x, y, z)]
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockId) {
        self.blocks[Chunk::get_block_index(x, y, z)] = block;
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
