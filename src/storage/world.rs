use std::collections::HashMap;

use crate::storage::chunk;
use crate::storage::svo::SVO;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct ChunkPos {
    x: i32,
    y: i32,
    z: i32,
}

impl ChunkPos {
    fn from_block_pos(x: i32, y: i32, z: i32) -> ChunkPos {
        ChunkPos {
            x: x >> 5,
            y: y >> 5,
            z: z >> 5,
        }
    }
}

pub struct World {
    chunks: HashMap<ChunkPos, chunk::Chunk>,
}

impl World {
    pub fn new() -> World {
        World {
            chunks: HashMap::new(),
        }
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> chunk::BlockId {
        let pos = ChunkPos::from_block_pos(x, y, z);
        if let Some(chunk) = self.chunks.get(&pos) {
            return chunk.get_block(x & 31, y & 31, z & 31);
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
        chunk.unwrap().set_block(x & 31, y & 31, z & 31, block);
    }
}


impl World {
    pub fn new_from_vox(data: dot_vox::DotVoxData) -> World {
        let model = &data.models[0];
        let mut world = World::new();
        for v in &model.voxels {
            world.set_block(v.x as i32, v.y as i32, v.z as i32, data.palette[v.i as usize]);
        }
        world
    }

    pub fn build_svo(&self) -> SVO {
        // TODO
        for (pos, chunk) in self.chunks.iter() {
            return chunk.build_svo();
        }
        SVO {
            max_depth: 0,
            max_depth_exp2: 0.0,
            descriptors: vec![],
        }
    }
}


#[cfg(test)]
mod tests {
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

        let world = super::World::new_from_vox(data);
        let chunk = world.chunks.get(&super::ChunkPos { x: 0, y: 0, z: 0 }).unwrap();

        assert_eq!(chunk.get_block(0, 0, 0), 101);
        assert_eq!(chunk.get_block(1, 0, 0), 102);
        assert_eq!(chunk.get_block(0, 1, 0), 103);
        assert_eq!(chunk.get_block(1, 1, 0), 104);
    }
}
