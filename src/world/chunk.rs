use std::ops::{Deref, Sub};

use cgmath::{num_traits, Point3};

use crate::world::memory::{Pool, Pooled, StatsAllocator};
use crate::world::octree::{Octree, Position};

pub type BlockId = u32;
pub type ChunkStorage = Octree<BlockId, StatsAllocator>;

pub const NO_BLOCK: BlockId = 0;

// -------------------------------------------------------------------------------------------------

/// `ChunkStorageAllocator` is an allocator for `ChunkStorage` objects.
pub struct ChunkStorageAllocator {
    pool: Pool<ChunkStorage, StatsAllocator>,
}

impl ChunkStorageAllocator {
    pub fn new() -> Self {
        let pool = Pool::new_in(
            Box::new(|alloc| {
                // It is difficult to choose the correct capacity for the octree storage, as octrees can differ a lot.
                // Here, an average is taken to avoid repetitive storage expansion during game startup. This will not
                // prevent from the program's memory usage to grow during runtime however.
                let mut storage = ChunkStorage::with_capacity_in(5000, alloc);
                storage.expand_to(5); // log2(32) = 5
                storage
            }),
            Some(Box::new(|storage| {
                storage.reset();
                storage.expand_to(5);
            })),
            StatsAllocator::new(),
        );
        Self { pool }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.pool.allocated_bytes()
    }
}

impl Deref for ChunkStorageAllocator {
    type Target = Pool<ChunkStorage, StatsAllocator>;

    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

#[cfg(test)]
mod chunk_storage_allocator_tests {
    use crate::world::chunk::ChunkStorageAllocator;

    /// Tests that newly allocated and reused storage objects always have a depth of 5 blocks to prevent visual voxel
    /// scale issues in the world.
    #[test]
    fn new() {
        let alloc = ChunkStorageAllocator::new();

        // new allocation
        let storage = alloc.allocate();
        assert_eq!(storage.depth(), 5);
        assert_eq!(alloc.used_count(), 1);
        assert_eq!(alloc.allocated_count(), 1);
        assert_ne!(alloc.allocated_bytes(), 0);

        drop(storage);

        assert_eq!(alloc.used_count(), 0);
        assert_eq!(alloc.allocated_count(), 1);

        // reused allocation
        let storage = alloc.allocate();
        assert_eq!(storage.depth(), 5);
        assert_eq!(alloc.used_count(), 1);
        assert_eq!(alloc.allocated_count(), 1);
        assert_ne!(alloc.allocated_bytes(), 0);

        drop(storage);

        // remove from buffer to check if allocator stats reset
        alloc.pool.clear();
        assert_eq!(alloc.allocated_bytes(), 0);
    }
}

// -------------------------------------------------------------------------------------------------

/// Chunk is a group of 32^3 voxels. It is the smallest voxel container. Many chunks make up the
/// world.
pub struct Chunk {
    pub pos: ChunkPos,
    /// Indicates the level of detail. Defined as the maximum depth to iterate inside the chunk's
    /// octree. 5 = maximum depth/full level of detail (2^5=32 - chunk block size along each axis).
    pub lod: u8,
    pub storage: Option<Pooled<ChunkStorage>>,
}

impl Chunk {
    pub fn new(pos: ChunkPos, lod: u8, storage: Pooled<ChunkStorage>) -> Self {
        Self { pos, lod, storage: Some(storage) }
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> BlockId {
        if self.storage.is_none() {
            return NO_BLOCK;
        }
        *self.storage.as_ref().unwrap().get_leaf(Position(x, y, z)).unwrap_or(&NO_BLOCK)
    }

    pub fn set_block(&mut self, x: u32, y: u32, z: u32, block: BlockId) {
        assert!(self.storage.is_some());

        if block == NO_BLOCK {
            self.storage.as_mut().unwrap().remove_leaf(Position(x, y, z));
        } else {
            self.storage.as_mut().unwrap().set_leaf(Position(x, y, z), block);
        }
    }

    /// Iterates through the whole chunk calling `f` for each block and sets it to the returned value. Any previous
    /// block information is cleared.
    pub fn fill_with<F: Fn(u32, u32, u32) -> Option<BlockId>>(&mut self, f: F) {
        assert!(self.storage.is_some());

        self.storage.as_mut().unwrap().construct_octants_with(5, |pos| f(pos.0, pos.1, pos.2));
    }
}

// -------------------------------------------------------------------------------------------------

/// `ChunkPos` represents a chunk's position in world space. One increment in chunk coord space is
/// equal to 32 increments in block coord space.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[allow(dead_code)]
impl ChunkPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub fn from_block_pos(x: i32, y: i32, z: i32) -> Self {
        Self { x: x >> 5, y: y >> 5, z: z >> 5 }
    }

    /// Returns the squared distance between this and the other chunk position.
    pub fn dst_sq(&self, other: &Self) -> f32 {
        let dx = (other.x - self.x) as f32;
        let dy = (other.y - self.y) as f32;
        let dz = (other.z - self.z) as f32;
        dz.mul_add(dz, dx.mul_add(dx, dy * dy))
    }

    /// Returns the squared distance between this and the other chunk position, but ignores the difference on the
    /// y-axis.
    pub fn dst_2d_sq(&self, other: &Self) -> f32 {
        let dx = (other.x - self.x) as f32;
        let dz = (other.z - self.z) as f32;
        dx.mul_add(dx, dz * dz)
    }

    pub fn as_block_pos(&self) -> Point3<i32> {
        Point3::new(self.x << 5, self.y << 5, self.z << 5)
    }
}

impl<T: num_traits::AsPrimitive<i32>> From<Point3<T>> for ChunkPos {
    fn from(value: Point3<T>) -> Self {
        Self::from_block_pos(value.x.as_(), value.y.as_(), value.z.as_())
    }
}

impl Sub for ChunkPos {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
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

    /// Tests that conversion from block positions works for both positive and negative coordinates.
    #[test]
    fn from_block_pos() {
        let pos = ChunkPos::from_block_pos(15, -28, 35);
        assert_eq!(pos, ChunkPos { x: 0, y: -1, z: 1 });
        assert_eq!(pos.as_block_pos(), Point3::new(0, -32, 32));

        let pos = ChunkPos::from_block_pos(10, 20, 30);
        assert_eq!(pos, ChunkPos { x: 0, y: 0, z: 0 });

        let pos = ChunkPos::from_block_pos(31, 32, 0);
        assert_eq!(pos, ChunkPos { x: 0, y: 1, z: 0 });

        let pos = ChunkPos::from_block_pos(-10, -20, -30);
        assert_eq!(pos, ChunkPos { x: -1, y: -1, z: -1 });

        let pos = ChunkPos::from_block_pos(-32, -33, 0);
        assert_eq!(pos, ChunkPos { x: -1, y: -2, z: 0 });
    }

    /// Tests distance calculation.
    #[test]
    fn dst_sq() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos.dst_sq(&other), 11.0);
        assert_eq!(other.dst_sq(&pos), 11.0);
    }

    /// Tests distance calculation without y axis.
    #[test]
    fn dst_2d_sq() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos.dst_2d_sq(&other), 2.0);
        assert_eq!(other.dst_2d_sq(&pos), 2.0);
    }

    /// Tests subtraction of two chunk positions.
    #[test]
    fn sub() {
        let pos = ChunkPos { x: 0, y: -1, z: 1 };
        let other = ChunkPos { x: -1, y: 2, z: 0 };
        assert_eq!(pos - other, ChunkPos { x: 1, y: -3, z: 1 });
    }
}

// -------------------------------------------------------------------------------------------------

/// `BlockPos` represents a block's position relative to the chunk it is in. Negative coordinates are
/// special because a block position of x=-1 is x=31 inside the actual chunk.
#[derive(Debug, PartialEq)]
pub struct BlockPos {
    pub chunk: ChunkPos,
    pub rel_x: f32,
    pub rel_y: f32,
    pub rel_z: f32,
}

#[allow(dead_code)]
impl BlockPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            chunk: ChunkPos::from_block_pos(x, y, z),
            rel_x: (x & 31) as f32,
            rel_y: (y & 31) as f32,
            rel_z: (z & 31) as f32,
        }
    }

    pub fn from(pos: Point3<f32>) -> Self {
        let (x, y, z) = (pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
        let (mut fx, mut fy, mut fz) = (pos.x.fract(), pos.y.fract(), pos.z.fract());

        if fx != 0.0 && pos.x < 0.0 { fx += 1.0; }
        if fy != 0.0 && pos.y < 0.0 { fy += 1.0; }
        if fz != 0.0 && pos.z < 0.0 { fz += 1.0; }

        Self {
            chunk: ChunkPos::from_block_pos(x, y, z),
            rel_x: (x & 31) as f32 + fx,
            rel_y: (y & 31) as f32 + fy,
            rel_z: (z & 31) as f32 + fz,
        }
    }

    pub fn to_point(&self) -> Point3<f32> {
        let mut pos = self.chunk.as_block_pos();
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

    /// Tests creation of block position and conversion back to a point.
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

    /// Tests that positive coordinates work.
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

    /// Tests that negative coordinates are wrapped correctly.
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

    /// Tests -1 edge case is calculated correctly.
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
