use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use crate::chunk::ChunkStorage;
use crate::world::chunk;
use crate::world::octree::Position;
use crate::world::svo::Svo;

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
}

pub struct World {
    // TODO do not make public later
    pub chunks: HashMap<ChunkPos, chunk::Chunk>,
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

    pub fn set_chunk(&mut self, pos: ChunkPos, chunk: chunk::Chunk) {
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
            self.chunks.insert(pos, chunk::Chunk::new());
            chunk = self.chunks.get_mut(&pos);
        }
        chunk.unwrap().set_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32, block);

        if !self.changed_chunks_set.contains(&pos) {
            self.changed_chunks_set.insert(pos);
            self.changed_chunks.push_back(pos);
        }
    }

    pub fn get_changed_chunks(&mut self) -> Vec<ChunkPos> {
        let changed = self.changed_chunks.drain(..).collect::<Vec<ChunkPos>>();
        self.changed_chunks_set.clear();
        changed
    }
}

impl World {
    pub fn add_vox_at(&mut self, data: &dot_vox::DotVoxData, block_x: i32, block_y: i32, block_z: i32) {
        let model = &data.models[0];
        for v in &model.voxels {
            self.set_block(block_x + v.x as i32, block_y + v.z as i32, block_z + v.y as i32, data.palette[v.i as usize]);
        }
    }

    pub fn build_svo(&self) -> Svo<Rc<ChunkStorage>> {
        let mut svo = Svo::new();
        for (pos, chunk) in self.chunks.iter() {
            let storage = chunk.get_storage();
            svo.set(Position(pos.x as u32, pos.y as u32, pos.z as u32), Some(storage));
        }
        svo
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
    fn world_new_from_vox() {
        let data = dot_vox::DotVoxData {
            version: 0,
            models: vec![
                dot_vox::Model {
                    size: dot_vox::Size { x: 2, y: 2, z: 2 },
                    voxels: vec![
                        dot_vox::Voxel { x: 0, y: 0, z: 0, i: 1 },
                        dot_vox::Voxel { x: 1, y: 0, z: 0, i: 2 },
                        dot_vox::Voxel { x: 0, y: 1, z: 0, i: 3 },
                        dot_vox::Voxel { x: 1, y: 1, z: 0, i: 4 },
                    ],
                },
            ],
            palette: vec![0, 101, 102, 103, 104],
            materials: vec![],
        };

        let mut world = super::World::new();
        world.add_vox_at(&data, 0, 0, 0);
        let chunk = world.chunks.get(&super::ChunkPos { x: 0, y: 0, z: 0 }).unwrap();

        assert_eq!(chunk.get_block(0, 0, 0), 101);
        assert_eq!(chunk.get_block(1, 0, 0), 102);
        assert_eq!(chunk.get_block(0, 0, 1), 103);
        assert_eq!(chunk.get_block(1, 0, 1), 104);
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
        assert_eq!(changed, vec![ChunkPos::from_block_pos(0, 0, 0)]);

        assert_eq!(true, world.changed_chunks_set.is_empty());
        assert_eq!(true, world.changed_chunks.is_empty());
    }
}
