use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::chunk::ChunkStorage;
use crate::world::allocator::Allocator;
use crate::world::chunk;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

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

pub struct World {
    // TODO do not make public later
    pub chunks: HashMap<ChunkPos, chunk::Chunk>,
    changed_chunks_set: HashSet<ChunkPos>,
    changed_chunks: VecDeque<ChunkPos>,
    // TODO do not make public later
    pub allocator: Arc<Allocator<ChunkStorage>>,
}

impl World {
    pub fn new() -> World {
        let allocator = Allocator::new(
            Box::new(|| ChunkStorage::with_size(32f32.log2() as u32)),
            Some(Box::new(|storage| storage.reset())),
        );
        World {
            chunks: HashMap::new(),
            changed_chunks_set: HashSet::new(),
            changed_chunks: VecDeque::new(),
            allocator: Arc::new(allocator),
        }
    }

    pub fn set_chunk(&mut self, chunk: chunk::Chunk) {
        let pos = chunk.pos;
        self.chunks.insert(pos, chunk);

        if !self.changed_chunks_set.contains(&pos) {
            self.changed_chunks_set.insert(pos);
            self.changed_chunks.push_back(pos);
        }
    }

    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        self.chunks.remove(pos);

        if !self.changed_chunks_set.contains(pos) {
            self.changed_chunks_set.insert(*pos);
            self.changed_chunks.push_back(*pos);
        }
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
        let mut chunk = self.chunks.get_mut(&pos);
        if chunk.is_none() {
            self.chunks.insert(pos, chunk::Chunk::new(pos, self.allocator.clone()));
            chunk = self.chunks.get_mut(&pos);
        }
        chunk.unwrap().set_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32, block);

        if !self.changed_chunks_set.contains(&pos) {
            self.changed_chunks_set.insert(pos);
            self.changed_chunks.push_back(pos);
        }
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

    use crate::world::world::ChunkPos;

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

        let chunk = world.chunks.get(&super::ChunkPos { x: 0, y: 1, z: 2 }).unwrap();
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
