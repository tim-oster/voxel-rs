use std::collections::{HashMap, HashSet, VecDeque};

use crate::world::chunk;
use crate::world::chunk::{Chunk, ChunkPos};

pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
    changed_chunks_set: HashSet<ChunkPos>,
    changed_chunks: VecDeque<ChunkPos>,
}

impl World {
    pub fn new() -> World {
        World {
            chunks: HashMap::new(),
            changed_chunks_set: HashSet::new(),
            changed_chunks: VecDeque::new(),
        }
    }

    fn mark_chunk_as_changed(&mut self, pos: &ChunkPos) {
        if !self.changed_chunks_set.contains(&pos) {
            self.changed_chunks_set.insert(*pos);
            self.changed_chunks.push_back(*pos);
        }
    }

    pub fn set_chunk(&mut self, chunk: Chunk) {
        let pos = chunk.pos;
        self.chunks.insert(pos, chunk);
        self.mark_chunk_as_changed(&pos);
    }

    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        self.chunks.remove(pos);
        self.mark_chunk_as_changed(&pos);
    }

    pub fn get_chunk(&self, pos: &ChunkPos) -> Option<&Chunk> {
        self.chunks.get(pos)
    }

    pub fn get_chunk_mut(&mut self, pos: &ChunkPos) -> Option<&mut Chunk> {
        self.mark_chunk_as_changed(pos);
        self.chunks.get_mut(pos)
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> chunk::BlockId {
        let pos = ChunkPos::from_block_pos(x, y, z);
        if let Some(chunk) = self.chunks.get(&pos) {
            return chunk.get_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32);
        }
        chunk::NO_BLOCK
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: chunk::BlockId) -> bool {
        let pos = ChunkPos::from_block_pos(x, y, z);
        if let Some(chunk) = self.chunks.get_mut(&pos) {
            chunk.set_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32, block);
            self.mark_chunk_as_changed(&pos);
            return true;
        }
        false
    }

    pub fn get_changed_chunks(&mut self, limit: u32) -> Vec<ChunkPos> {
        let mut changed = Vec::new();
        for _ in 0..limit {
            let pos = self.changed_chunks.pop_front();
            if pos.is_none() {
                break;
            }

            let pos = pos.unwrap();
            self.changed_chunks_set.remove(&pos);
            changed.push(pos);
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};
    use std::sync::Arc;

    use crate::world::allocator::Allocator;
    use crate::world::chunk::{Chunk, ChunkPos, ChunkStorage};

    #[test]
    fn world_get_and_set_block() {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        let allocator = Arc::new(allocator);

        let mut world = super::World::new();
        world.set_chunk(Chunk::new(ChunkPos::new(0, 1, 2), allocator));

        let block = world.get_block(1, 33, 65);
        assert_eq!(block, super::chunk::NO_BLOCK);

        world.set_block(1, 33, 65, 99);

        let chunk = world.chunks.get(&ChunkPos { x: 0, y: 1, z: 2 }).unwrap();
        assert_eq!(chunk.get_block(1, 1, 1), 99);

        let block = world.get_block(1, 33, 65);
        assert_eq!(block, 99);
    }

    #[test]
    fn world_changed_chunks() {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        let allocator = Arc::new(allocator);

        let mut world = super::World::new();
        world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), allocator));

        for _ in 0..2 {
            world.set_block(0, 0, 0, 1);
        }

        let mut set = HashSet::new();
        set.insert(ChunkPos::from_block_pos(0, 0, 0));
        assert_eq!(world.changed_chunks_set, set);

        assert_eq!(world.changed_chunks, VecDeque::from(vec![ChunkPos::from_block_pos(0, 0, 0)]));

        let changed = world.get_changed_chunks(10);
        assert_eq!(changed, vec![ChunkPos::from_block_pos(0, 0, 0)]);

        assert_eq!(true, world.changed_chunks_set.is_empty());
        assert_eq!(true, world.changed_chunks.is_empty());
    }
}
