use std::alloc::Allocator;
use std::rc::Rc;
use std::sync::Arc;

use cgmath::Point3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graphics;
use crate::graphics::framebuffer::Framebuffer;
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult};
use crate::systems::jobs::{ChunkProcessor, ChunkResult, JobSystem};
use crate::systems::physics::Raycaster;
use crate::world;
use crate::world::chunk::{BlockPos, ChunkPos};
use crate::world::memory::{AllocatorStats, Pool, StatsAllocator};
use crate::world::octree::LeafId;
use crate::world::svo::{ChunkBuffer, ChunkBufferPool, SerializedChunk, SvoSerializable};
use crate::world::world::BorrowedChunk;

/// Svo takes ownership of a [`graphics::Svo`] and populates it with world [`world::chunk::Chunk`]s.
/// Adding chunks will serialize them in the background and attach them the GPU SVO. Removing
/// chunks will also remove them from the GPU.
///
/// In addition to serialization, this Svo manages "chunk shifting". One major limitation of the
/// SVO structure used in this project is, that it can only grow in the positive direction of each
/// axis. Supporting an "infinitely" large world in all directions is consequently not possible by
/// default. This implementation solves this shortcoming by always keeping the camera position
/// inside the center chunk of the SVO and shifting all chunks in the opposite movement direction
/// if the camera leaves the chunk.
pub struct Svo {
    processor: ChunkProcessor<SerializedChunk>,

    world_svo_alloc: StatsAllocator,
    world_svo: world::Svo<SerializedChunk, StatsAllocator>,

    graphics_svo: graphics::Svo,
    chunk_buffer_pool: Arc<ChunkBufferPool>,

    leaf_ids: FxHashMap<ChunkPos, LeafId>,
    has_changed: bool,
    svo_coord_space: SvoCoordSpace,
}

pub struct AllocStats {
    pub chunk_buffers_used: usize,
    pub chunk_buffers_allocated: usize,
    pub chunk_buffers_bytes_total: usize,
    pub world_svo_buffer_bytes: usize,
}

impl Svo {
    pub fn new(job_system: Rc<JobSystem>, graphics_svo: graphics::Svo, render_distance: u32) -> Self {
        let world_svo_alloc = StatsAllocator::new();

        let chunk_buffer_pool = Pool::new_in(
            // It is difficult to pre-allocate memory here as chunk sizes are random/depend heavily on the world generation
            // mechanism. The naive approach is to allocate the maximum amount of memory, but that is too wasteful. Hence,
            // an average size is taken so that it is sufficient in most cases and at worst, the storage is expanded a few
            // times until it fits. This is still more stable than and safes a lot of allocations.
            Box::new(|alloc| ChunkBuffer::with_capacity_in(100_000, alloc)),
            Some(Box::new(ChunkBuffer::reset)),
            StatsAllocator::new(),
        );
        Self {
            processor: ChunkProcessor::new(job_system),
            world_svo_alloc: world_svo_alloc.clone(),
            world_svo: world::Svo::new_in(world_svo_alloc),
            graphics_svo,
            chunk_buffer_pool: Arc::new(chunk_buffer_pool),
            leaf_ids: FxHashMap::default(),
            has_changed: false,
            svo_coord_space: SvoCoordSpace {
                center: ChunkPos::new(0, 0, 0),
                dst: render_distance,
            },
        }
    }

    /// Enqueues the borrowed chunk to be serialized into the GPU SVO structure. All moved chunk
    /// ownerships can be reclaimed by calling [`Svo::update`].
    pub fn set_chunk(&mut self, chunk: BorrowedChunk) {
        let alloc = self.chunk_buffer_pool.clone();
        self.processor.enqueue(chunk.pos, true, move || SerializedChunk::new(chunk, &alloc));
    }

    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        self.processor.dequeue(pos);

        if let Some(id) = self.leaf_ids.remove(pos) {
            self.world_svo.remove_leaf(id);
            self.has_changed = true;
        }
    }

    /// Returns if the SVO still has in-work chunks or if there are unconsumed chunks in the buffer.
    pub fn has_pending_jobs(&self) -> bool {
        self.processor.has_pending()
    }

    pub fn get_render_distance(&self) -> u32 {
        self.svo_coord_space.dst
    }

    pub fn get_alloc_stats(&self) -> AllocStats {
        AllocStats {
            chunk_buffers_used: self.chunk_buffer_pool.used_count(),
            chunk_buffers_allocated: self.chunk_buffer_pool.allocated_count(),
            chunk_buffers_bytes_total: self.chunk_buffer_pool.allocated_bytes(),
            world_svo_buffer_bytes: self.world_svo_alloc.allocated_bytes(),
        }
    }

    /// Updates the internal reference world center and performs "chunk shifting", if necessary.
    /// Additionally, it uploads all serialized chunks to the GPU, that have finished since the
    /// last update. Position is in world space.
    ///
    /// Returns borrowed chunk ownership from finished chunk jobs there were enqueued before.
    pub fn update(&mut self, world_center: &ChunkPos) -> Vec<BorrowedChunk> {
        if self.svo_coord_space.center != *world_center {
            self.svo_coord_space.center = *world_center;
            self.on_coord_space_change();
        }

        let results = self.processor.get_results(400);
        let chunks = self.process_serialized_chunks(results);

        if !self.has_changed {
            return chunks;
        }

        self.has_changed = false;
        self.world_svo.serialize();
        self.graphics_svo.update(&mut self.world_svo);

        chunks
    }

    fn on_coord_space_change(&mut self) {
        self.has_changed = true;
        Self::shift_chunks(&self.svo_coord_space, &mut self.leaf_ids, &mut self.world_svo);
    }

    /// Iterates through all chunks and "shifts" them, if necessary, to their new position in SVO
    /// space by replacing the previous chunk in the new position. Also removes all chunks, that
    /// are out of SVO bounds.
    fn shift_chunks<T: SvoSerializable, A: Allocator>(coord_space: &SvoCoordSpace, leaf_ids: &mut FxHashMap<ChunkPos, LeafId>, world_svo: &mut world::Svo<T, A>) {
        let mut overridden_leaves = FxHashMap::default();
        let mut removed = FxHashSet::default();

        for (chunk_pos, leaf_id) in leaf_ids.iter_mut() {
            let new_svo_pos = coord_space.cnv_chunk_pos(*chunk_pos);
            if new_svo_pos.is_none() {
                // remove leaf from octree if it hasn't yet been overridden by another moved leaf
                if !overridden_leaves.contains_key(leaf_id) {
                    world_svo.remove_leaf(*leaf_id);
                }
                overridden_leaves.remove(leaf_id);
                removed.insert(*chunk_pos);
                continue;
            }

            let new_svo_pos = new_svo_pos.unwrap();

            let (new_leaf_id, old_value) = if let Some(value) = overridden_leaves.remove(leaf_id) {
                // try to bypass serialization in svo since the leaf was only moved and might
                // already in the serialization buffer
                world_svo.set_leaf(new_svo_pos, value, false)
            } else {
                world_svo.move_leaf(*leaf_id, new_svo_pos)
            };

            *leaf_id = new_leaf_id;
            if let Some(value) = old_value {
                overridden_leaves.insert(new_leaf_id, value);
            }
        }

        for pos in removed {
            leaf_ids.remove(&pos);
        }
    }

    fn process_serialized_chunks(&mut self, results: Vec<ChunkResult<SerializedChunk>>) -> Vec<BorrowedChunk> {
        let mut chunks = Vec::new();

        for mut result in results {
            let chunk = result.value.borrowed_chunk.take().unwrap();
            chunks.push(chunk);

            let svo_pos = self.svo_coord_space.cnv_chunk_pos(result.pos);
            if svo_pos.is_none() {
                continue;
            }

            // NOTE: this moves ownership of the serialized ChunkBuffer into the world svo octree.
            //       If not freed properly, the otherwise pooled objects cannot be reused.
            let (id, _) = self.world_svo.set_leaf(svo_pos.unwrap(), result.value, true);
            self.leaf_ids.insert(result.pos, id);
            self.has_changed = true;
        }

        chunks
    }

    pub fn set_radius(&mut self, radius: u32) {
        self.svo_coord_space.dst = radius;
        self.on_coord_space_change();
    }
}

//noinspection DuplicatedCode
#[cfg(test)]
mod svo_tests {
    use rustc_hash::FxHashMap;

    use crate::systems::worldsvo::{Svo, SvoCoordSpace};
    use crate::world;
    use crate::world::chunk::ChunkPos;
    use crate::world::octree::Position;
    use crate::world::svo::{SerializationResult, SvoSerializable};

    impl SvoSerializable for u32 {
        fn unique_id(&self) -> u64 {
            *self as u64
        }

        fn serialize(&mut self, dst: &mut Vec<u32>, _lod: u8) -> SerializationResult {
            dst.push(*self);
            SerializationResult { child_mask: 1, leaf_mask: 1, depth: 1 }
        }
    }

    /// Tests that chunk shifting in positive x direction works.
    #[test]
    fn shift_chunks_x_positive() {
        let mut leaf_ids = FxHashMap::default();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let (c0, _) = world_svo.set_leaf(Position(0, 1, 1), 1u32, true);
        leaf_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let (c1, _) = world_svo.set_leaf(Position(1, 1, 1), 2u32, true);
        leaf_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let (c2, _) = world_svo.set_leaf(Position(2, 1, 1), 3u32, true);
        leaf_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(-1, 0, 0), c0),
            (ChunkPos::new(0, 0, 0), c1),
            (ChunkPos::new(1, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), Some(&3u32));

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(1, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(0, 0, 0), c0),
            (ChunkPos::new(1, 0, 0), c1),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), None);

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(2, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(1, 0, 0), c0),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), None);

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::default());
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), None);
    }

    /// Tests that chunk shifting in negative x direction works.
    #[test]
    fn shift_chunks_x_negative() {
        let mut leaf_ids = FxHashMap::default();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let (c0, _) = world_svo.set_leaf(Position(0, 1, 1), 1u32, true);
        leaf_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let (c1, _) = world_svo.set_leaf(Position(1, 1, 1), 2u32, true);
        leaf_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let (c2, _) = world_svo.set_leaf(Position(2, 1, 1), 3u32, true);
        leaf_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(-1, 0, 0), c0),
            (ChunkPos::new(0, 0, 0), c1),
            (ChunkPos::new(1, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), Some(&3u32));

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-1, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(-1, 0, 0), c1),
            (ChunkPos::new(0, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), Some(&2u32));

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-2, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::from_iter([
            (ChunkPos::new(-1, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), Some(&1u32));

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::default());
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), None);
    }

    /// Tests that chunk shifting removes all out of bounds chunks, even if a larger leap in the
    /// world center happens.
    #[test]
    fn shift_chunks_x_out_of_range() {
        let mut leaf_ids = FxHashMap::default();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let (c0, _) = world_svo.set_leaf(Position(0, 1, 1), 1u32, true);
        leaf_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let (c1, _) = world_svo.set_leaf(Position(1, 1, 1), 2u32, true);
        leaf_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let (c2, _) = world_svo.set_leaf(Position(2, 1, 1), 3u32, true);
        leaf_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), Some(&3u32));

        // shift x out of range
        let cs = SvoCoordSpace::new(ChunkPos::new(3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut leaf_ids, &mut world_svo);
        assert_eq!(leaf_ids, FxHashMap::default());
        assert_eq!(world_svo.get_leaf(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_leaf(Position(2, 1, 1)), None);
    }
}

/// Implement "overrides" for [`graphics::Svo`]. All positions are transformed from world space
/// into SVO space.
impl Svo {
    /// Calls [`graphics::Svo::reload_resources`].
    pub fn reload_resources(&mut self) {
        self.graphics_svo.reload_resources();
    }

    /// Calls [`graphics::Svo::render`]. Positions are expected to be in world space.
    pub fn render(&self, params: graphics::svo::RenderParams, target: &Framebuffer) {
        let mut params = params;

        // translate camera position into SVO
        params.cam_pos = self.svo_coord_space.cnv_block_pos(params.cam_pos);

        // translate selected voxel position into SVO
        if let Some(pos) = params.selected_voxel {
            params.selected_voxel = Some(self.svo_coord_space.cnv_block_pos(pos));
        }

        self.graphics_svo.render(&params, target);
    }

    /// Calls [`graphics::Svo::get_stats`].
    pub fn get_stats(&self) -> graphics::svo::Stats {
        self.graphics_svo.get_stats()
    }
}

/// Implement [`Raycaster`] that calls [`graphics::Svo`] underneath. All positions are transformed
/// from world space into SVO space.
impl Raycaster for Svo {
    /// Calls [`graphics::Svo::raycast`]. Positions are expected to be in world space.
    fn raycast(&self, batch: &mut PickerBatch, result: &mut PickerBatchResult) {
        for ray in &mut batch.rays {
            ray.pos = self.svo_coord_space.cnv_block_pos(ray.pos);
        }
        for aabb in &mut batch.aabbs {
            aabb.pos = self.svo_coord_space.cnv_block_pos(aabb.pos);
        }

        self.graphics_svo.raycast(batch, result);

        for ray in &mut result.rays {
            ray.pos = self.svo_coord_space.cnv_svo_pos(ray.pos);
        }
    }
}

#[derive(Copy, Clone)]
struct SvoCoordSpace {
    pub center: ChunkPos,
    pub dst: u32,
}

type SvoPos = Point3<f32>;

#[allow(dead_code)]
impl SvoCoordSpace {
    fn new(center: ChunkPos, dst: u32) -> Self {
        Self { center, dst }
    }

    /// Converts a block position from world space to SVO space.
    fn cnv_block_pos(&self, pos: Point3<f32>) -> SvoPos {
        let mut block_pos = BlockPos::from(pos);
        let delta = block_pos.chunk - self.center;

        let rd = self.dst as i32;
        block_pos.chunk.x = rd + delta.x;
        block_pos.chunk.y = rd + delta.y;
        block_pos.chunk.z = rd + delta.z;

        block_pos.to_point()
    }

    /// Converts a block position from SVO space to world space.
    fn cnv_svo_pos(&self, pos: SvoPos) -> Point3<f32> {
        let mut block_pos = BlockPos::from(pos);

        let rd = self.dst as i32;
        let delta = block_pos.chunk - ChunkPos::new(rd, rd, rd);

        block_pos.chunk.x = self.center.x + delta.x;
        block_pos.chunk.y = self.center.y + delta.y;
        block_pos.chunk.z = self.center.z + delta.z;

        block_pos.to_point()
    }

    /// Converts a chunk position from world space to the respective chunk position in SVO space,
    /// if possible. Conversion is not possible if the position is outside the coordinate space's
    /// `dst`.
    fn cnv_chunk_pos(&self, pos: ChunkPos) -> Option<world::octree::Position> {
        let r = self.dst as f32;

        let pos = pos.as_block_pos();
        let pos = self.cnv_block_pos(pos.cast().unwrap());
        let pos = pos / 32.0;

        // y is height based, so the full radius is used in both directions
        let dcy = pos.y - r;
        if dcy < -r || dcy > r {
            return None;
        }

        // perform radial check for x and z
        let dcx = pos.x - r;
        let dcz = pos.z - r;
        if dcx.mul_add(dcx, dcz * dcz) > r * r {
            return None;
        }

        Some(world::octree::Position(pos.x as u32, pos.y as u32, pos.z as u32))
    }
}

#[cfg(test)]
mod coord_space_tests {
    use cgmath::Point3;

    use crate::systems::worldsvo::SvoCoordSpace;
    use crate::world::chunk::ChunkPos;
    use crate::world::octree::Position;

    /// Tests transformation for positive coordinates.
    #[test]
    fn coord_space_positive() {
        let cs = SvoCoordSpace::new(ChunkPos::new(4, 5, 12), 2);

        let world_pos = Point3::new(32.0f32.mul_add(5.0, 16.25), 32.0f32.mul_add(3.0, 4.25), 32.0f32.mul_add(10.0, 20.5));
        let svo_pos = cs.cnv_block_pos(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0f32.mul_add(3.0, 16.25), 32.0f32.mul_add(0.0, 4.25), 32.0f32.mul_add(0.0, 20.5)));

        let cnv_back = cs.cnv_svo_pos(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    /// Tests transformation for negative coordinates.
    #[test]
    fn coord_space_negative() {
        let cs = SvoCoordSpace::new(ChunkPos::new(-1, -1, -1), 2);

        let world_pos = Point3::new(-16.25, -4.25, -20.5);
        let svo_pos = cs.cnv_block_pos(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0f32.mul_add(2.0, 15.75), 32.0f32.mul_add(2.0, 27.75), 32.0f32.mul_add(2.0, 11.5)));

        let cnv_back = cs.cnv_svo_pos(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    /// Tests if chunk position conversion and all edge cases work properly.
    #[test]
    fn cnv_chunk_pos() {
        let cs = SvoCoordSpace::new(ChunkPos::new(0, 0, 0), 1);

        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(-1, 0, 0));
        assert_eq!(svo_pos, Some(Position(0, 1, 1)));
        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(0, 0, 0));
        assert_eq!(svo_pos, Some(Position(1, 1, 1)));
        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(1, 0, 0));
        assert_eq!(svo_pos, Some(Position(2, 1, 1)));

        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(-2, 0, 0));
        assert_eq!(svo_pos, None);
        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(2, 0, 0));
        assert_eq!(svo_pos, None);
        let svo_pos = cs.cnv_chunk_pos(ChunkPos::new(1, 0, 1));
        assert_eq!(svo_pos, None);
    }
}
