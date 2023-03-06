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

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: chunk::BlockId) {
        let pos = ChunkPos::from_block_pos(x, y, z);
        let chunk = self.chunks.get_mut(&pos).unwrap();
        chunk.set_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32, block);
        self.mark_chunk_as_changed(&pos);
    }

    pub fn get_changed_chunks(&mut self) -> HashSet<ChunkPos> {
        let changed = self.changed_chunks.drain(..).collect::<HashSet<ChunkPos>>();
        self.changed_chunks_set.clear();
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};

    use crate::systems::world::ChunkPos;

    #[test]
    fn chunk_pos_from_block_pos() {
        use super::ChunkPos;

        let pos = ChunkPos::from_block_pos(10, 20, 30);
        assert_eq!(pos, ChunkPos { x: 0, y: 0, z: 0 });

        let pos = ChunkPos::from_block_pos(31, 32, 0);
        assert_eq!(pos, ChunkPos { x: 0, y: 1, z: 0 });

        let pos = ChunkPos::from_block_pos(-10, -20, -30);
        assert_eq!(pos, ChunkPos { x: -1, y: -1, z: -1 });

        let pos = ChunkPos::from_block_pos(-32, -33, 0);
        assert_eq!(pos, ChunkPos { x: -1, y: -2, z: 0 });
    }

    #[test]
    fn world_get_and_set_block() {
        let mut world = super::World::new();

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
        let mut world = super::World::new();

        for _ in 0..2 {
            world.set_block(0, 0, 0, 1);
        }

        let mut set = HashSet::new();
        set.insert(ChunkPos::from_block_pos(0, 0, 0));
        assert_eq!(world.changed_chunks_set, set);

        assert_eq!(world.changed_chunks, VecDeque::from(vec![ChunkPos::from_block_pos(0, 0, 0)]));

        let changed = world.get_changed_chunks();
        assert_eq!(changed, HashSet::from([ChunkPos::from_block_pos(0, 0, 0)]));

        assert_eq!(true, world.changed_chunks_set.is_empty());
        assert_eq!(true, world.changed_chunks.is_empty());
    }
}
