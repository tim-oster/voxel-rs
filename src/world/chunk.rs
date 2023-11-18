use std::ops::Sub;
use std::sync::{Arc, RwLock};

use cgmath::Point3;

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

    pub fn to_block_pos(&self) -> Point3<i32> {
        Point3::new(self.x << 5, self.y << 5, self.z << 5)
    }
}

impl Sub for ChunkPos {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ChunkPos {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

#[cfg(test)]
mod chunk_pos_test {
    use cgmath::Point3;

    use crate::world::chunk::ChunkPos;

    #[test]
    fn from_block_pos() {
        let pos = ChunkPos::from_block_pos(15, -28, 35);
        assert_eq!(pos, ChunkPos { x: 0, y: -1, z: 1 });
        assert_eq!(pos.to_block_pos(), Point3::new(0, -32, 32));
    }

    #[test]
    fn dst_sq() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos.dst_sq(&other), 11.0);
        assert_eq!(other.dst_sq(&pos), 11.0);
    }

    #[test]
    fn dst_2d_sq() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos.dst_2d_sq(&other), 2.0);
        assert_eq!(other.dst_2d_sq(&pos), 2.0);
    }

    #[test]
    fn sub() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos - other, ChunkPos { x: 1, y: -3, z: 1 });
    }
}

#[derive(Debug, PartialEq)]
pub struct BlockPos {
    pub chunk: ChunkPos,
    pub rel_x: f32,
    pub rel_y: f32,
    pub rel_z: f32,
}

impl BlockPos {
    pub fn new(x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos {
            chunk: ChunkPos::from_block_pos(x, y, z),
            rel_x: (x & 31) as f32,
            rel_y: (y & 31) as f32,
            rel_z: (z & 31) as f32,
        }
    }

    pub fn from(pos: Point3<f32>) -> BlockPos {
        let (x, y, z) = (pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        let (mut fx, mut fy, mut fz) = (pos.x.fract(), pos.y.fract(), pos.z.fract());

        if fx != 0.0 && pos.x < 0.0 { fx += 1.0; }
        if fy != 0.0 && pos.y < 0.0 { fy += 1.0; }
        if fz != 0.0 && pos.z < 0.0 { fz += 1.0; }

        BlockPos {
            chunk: ChunkPos::from_block_pos(x, y, z),
            rel_x: (x & 31) as f32 + fx,
            rel_y: (y & 31) as f32 + fy,
            rel_z: (z & 31) as f32 + fz,
        }
    }

    pub fn to_point(&self) -> Point3<f32> {
        let mut pos = self.chunk.to_block_pos();
        pos.x |= (self.rel_x as i32) & 31;
        pos.y |= (self.rel_y as i32) & 31;
        pos.z |= (self.rel_z as i32) & 31;

        Point3::new(
            pos.x as f32 + self.rel_x.fract(),
            pos.y as f32 + self.rel_y.fract(),
            pos.z as f32 + self.rel_z.fract(),
        )
    }
}

#[cfg(test)]
mod block_pos_test {
    use cgmath::Point3;

    use crate::world::chunk::{BlockPos, ChunkPos};

    #[test]
    fn new() {
        let pos = BlockPos::new(15, -28, 35);
        assert_eq!(pos, BlockPos {
            chunk: ChunkPos::new(0, -1, 1),
            rel_x: 15.0,
            rel_y: 4.0,
            rel_z: 3.0,
        });
        assert_eq!(Point3::new(15.0, -28.0, 35.0), pos.to_point());
    }

    #[test]
    fn from_positive() {
        let original = Point3::new(0.25, 32.75, 8.5);
        let pos = BlockPos::from(original);
        assert_eq!(pos, BlockPos {
            chunk: ChunkPos::new(0, 1, 0),
            rel_x: 0.25,
            rel_y: 0.75,
            rel_z: 8.5,
        });
        assert_eq!(original, pos.to_point());
    }

    #[test]
    fn from_negative() {
        let original = Point3::new(-0.25, -32.75, -8.5);
        let pos = BlockPos::from(original);
        assert_eq!(pos, BlockPos {
            chunk: ChunkPos::new(-1, -2, -1),
            rel_x: 31.75,
            rel_y: 31.25,
            rel_z: 23.5,
        });
        assert_eq!(original, pos.to_point());
    }

    #[test]
    fn from_edge_case() {
        let original = Point3::new(-1.0, -1.0, -1.0);
        let pos = BlockPos::from(original);
        assert_eq!(pos, BlockPos {
            chunk: ChunkPos::new(-1, -1, -1),
            rel_x: 31.0,
            rel_y: 31.0,
            rel_z: 31.0,
        });
        assert_eq!(original, pos.to_point());
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
