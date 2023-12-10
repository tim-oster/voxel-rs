use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use cgmath::Point3;

use crate::graphics;
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult};
use crate::systems::jobs::{ChunkProcessor, ChunkResult, JobSystem};
use crate::systems::physics::Raycaster;
use crate::world;
use crate::world::chunk::{BlockPos, Chunk, ChunkPos};
use crate::world::svo::{SerializedChunk, SvoSerializable};
use crate::world::world::BorrowedChunk;

/// Svo takes ownership of a [`graphics::Svo`] and populates it with world [`Chunk`]s. Adding chunks
/// will serialize them in the background and attach them the the GPU SVO. Removing chunks will
/// also remove them from the GPU.
///
/// In addition to serialization, this Svo manages "chunk shifting". One major limitation of the
/// SVO structure used in this project is, that it can only grow in the positive direction of each
/// axis. Supporting a "infinitely" large world in all directions is consequently not possible by
/// default. This implementation solves this shortcoming by always keeping the camera position
/// inside the center chunk of the SVO and shifting all chunks in the opposite movement direction
/// if the camera leaves the chunk.
pub struct Svo {
    processor: ChunkProcessor<SerializedChunk>,

    world_svo: world::Svo<SerializedChunk>,
    graphics_svo: graphics::Svo,

    octant_ids: HashMap<ChunkPos, world::octree::OctantId>,
    has_changed: bool,
    svo_coord_space: SvoCoordSpace,
}

impl Svo {
    pub fn new(job_system: Rc<JobSystem>, graphics_svo: graphics::Svo, render_distance: u32) -> Svo {
        Svo {
            processor: ChunkProcessor::new(job_system),
            world_svo: world::Svo::new(),
            graphics_svo,
            octant_ids: HashMap::new(),
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
        self.processor.enqueue(chunk.pos, true, move || SerializedChunk::new(chunk));
    }

    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        self.processor.dequeue(pos);

        if let Some(id) = self.octant_ids.remove(pos) {
            self.world_svo.remove_octant(id);
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

    /// Updates the internal reference world center and performs "chunk shifting", if necessary.
    /// Additionally, it uploads all serialized chunks to the GPU, that have finished since the
    /// last update. Position is in world space.
    ///
    /// Returns borrowed chunk ownership from finished chunk jobs there were enqueued before.
    pub fn update(&mut self, world_center: &ChunkPos) -> Vec<BorrowedChunk> {
        if self.svo_coord_space.center != *world_center {
            self.svo_coord_space.center = *world_center;
            self.has_changed = true;

            Self::shift_chunks(&self.svo_coord_space, &mut self.octant_ids, &mut self.world_svo);
        }

        let results = self.processor.get_results(50);
        let chunks = self.process_serialized_chunks(results);

        if !self.has_changed {
            return chunks;
        }

        self.has_changed = false;
        self.world_svo.serialize();
        self.graphics_svo.update(&self.world_svo);
        self.world_svo.reset_changes();

        chunks
    }

    /// Iterates through all chunks and "shifts" them, if necessary, to their new position in SVO
    /// space by replacing the previous chunk in the new position. Also removes all chunks, that
    /// are out of SVO bounds.
    fn shift_chunks<T: SvoSerializable>(coord_space: &SvoCoordSpace, octant_ids: &mut HashMap<ChunkPos, world::octree::OctantId>, world_svo: &mut world::Svo<T>) {
        let mut octant_delete_set = HashSet::new();
        let mut id_delete_pos = HashSet::new();

        for (chunk_pos, octant_id) in octant_ids.iter() {
            // Depending on iteration order of map entries, it can happen that a previous operation
            // shifted its chunk into a chunk using a next iteration's octant_id. In that case, it
            // is safe to remove that octant_id from the delete set again and process it again.
            octant_delete_set.remove(octant_id);

            // Check if chunk position (world space) can be converted into SVO space. If not, remove
            // both octant and chunk position.
            let new_svo_pos = coord_space.cnv_chunk_pos(*chunk_pos);
            if new_svo_pos.is_none() {
                octant_delete_set.insert(*octant_id);
                id_delete_pos.insert(*chunk_pos);
                continue;
            }

            // If the position can be converted, try to replace it in the SVO and remove any old
            // octant that might potentially be present. If that octant is still used, an upcoming
            // iteration will remove it from the delete set.
            let old_octant_id = world_svo.replace(new_svo_pos.unwrap(), *octant_id);
            if let Some(old_octant_id) = old_octant_id {
                octant_delete_set.insert(old_octant_id);
            }
        }

        // use delete sets to remove all unused octants
        octant_ids.retain(|_, v| !octant_delete_set.contains(v));
        for id in octant_delete_set {
            world_svo.remove_octant(id);
        }
        for pos in id_delete_pos {
            octant_ids.remove(&pos);
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

            if let Some(id) = self.world_svo.set(svo_pos.unwrap(), Some(result.value)) {
                self.octant_ids.insert(result.pos, id);
                self.has_changed = true;
            }
        }

        chunks
    }
}

#[cfg(test)]
mod svo_tests {
    use std::collections::HashMap;

    use crate::systems::worldsvo::{Svo, SvoCoordSpace};
    use crate::world;
    use crate::world::chunk::ChunkPos;
    use crate::world::octree::Position;
    use crate::world::svo::{SerializationResult, SvoSerializable};

    impl SvoSerializable for u32 {
        fn serialize(&self, dst: &mut Vec<u32>, lod: u8) -> SerializationResult {
            dst.push(*self);
            SerializationResult { child_mask: 0, leaf_mask: 0, depth: 0 }
        }
    }

    /// Tests that chunk shifting in positive x direction works.
    #[test]
    fn shift_chunks_x_positive() {
        let cs = SvoCoordSpace::new(ChunkPos::new(0, 0, 0), 1);
        let mut octant_ids = HashMap::new();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let c0 = world_svo.set(Position(0, 1, 1), Some(1u32)).unwrap();
        octant_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let c1 = world_svo.set(Position(1, 1, 1), Some(2u32)).unwrap();
        octant_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let c2 = world_svo.set(Position(2, 1, 1), Some(3u32)).unwrap();
        octant_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.octant_count(), 6);

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(1, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::from([
            (ChunkPos::new(0, 0, 0), c1),
            (ChunkPos::new(1, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), None);
        assert_eq!(world_svo.octant_count(), 5);

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(2, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::from([
            (ChunkPos::new(1, 0, 0), c2),
        ]));
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), None);
        assert_eq!(world_svo.octant_count(), 4);

        // shift one in x+
        let cs = SvoCoordSpace::new(ChunkPos::new(3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::new());
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), None);
        assert_eq!(world_svo.octant_count(), 3);
    }

    /// Tests that chunk shifting in negative x direction works.
    #[test]
    fn shift_chunks_x_negative() {
        let cs = SvoCoordSpace::new(ChunkPos::new(0, 0, 0), 1);
        let mut octant_ids = HashMap::new();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let c0 = world_svo.set(Position(0, 1, 1), Some(1u32)).unwrap();
        octant_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let c1 = world_svo.set(Position(1, 1, 1), Some(2u32)).unwrap();
        octant_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let c2 = world_svo.set(Position(2, 1, 1), Some(3u32)).unwrap();
        octant_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.octant_count(), 6);

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-1, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::from([
            (ChunkPos::new(-1, 0, 0), c0),
            (ChunkPos::new(0, 0, 0), c1),
        ]));
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.octant_count(), 5);

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-2, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::from([
            (ChunkPos::new(-1, 0, 0), c0),
        ]));
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.octant_count(), 4);

        // shift one in x-
        let cs = SvoCoordSpace::new(ChunkPos::new(-3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::new());
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), None);
        assert_eq!(world_svo.octant_count(), 3);
    }

    /// Tests that chunk shifting removes all out of bounds chunks, even if a larger leap in the
    /// world center happens.
    #[test]
    fn shift_chunks_x_out_of_range() {
        let cs = SvoCoordSpace::new(ChunkPos::new(0, 0, 0), 1);
        let mut octant_ids = HashMap::new();
        let mut world_svo = world::Svo::new();

        // setup test SVO
        let c0 = world_svo.set(Position(0, 1, 1), Some(1u32)).unwrap();
        octant_ids.insert(ChunkPos::new(-1, 0, 0), c0);

        let c1 = world_svo.set(Position(1, 1, 1), Some(2u32)).unwrap();
        octant_ids.insert(ChunkPos::new(0, 0, 0), c1);

        let c2 = world_svo.set(Position(2, 1, 1), Some(3u32)).unwrap();
        octant_ids.insert(ChunkPos::new(1, 0, 0), c2);

        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), Some(&1u32));
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), Some(&2u32));
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), Some(&3u32));
        assert_eq!(world_svo.octant_count(), 6);

        // shift x out of range
        let cs = SvoCoordSpace::new(ChunkPos::new(3, 0, 0), 1);
        Svo::shift_chunks(&cs, &mut octant_ids, &mut world_svo);
        assert_eq!(octant_ids, HashMap::new());
        assert_eq!(world_svo.get_at_pos(Position(0, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(1, 1, 1)), None);
        assert_eq!(world_svo.get_at_pos(Position(2, 1, 1)), None);
        assert_eq!(world_svo.octant_count(), 3);
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
    pub fn render(&self, params: graphics::svo::RenderParams) {
        let mut params = params;

        // translate camera position into SVO
        params.cam_pos = self.svo_coord_space.cnv_block_pos(params.cam_pos);

        // translate selected voxel position into SVO
        if let Some(pos) = params.selected_voxel {
            params.selected_voxel = Some(self.svo_coord_space.cnv_block_pos(pos));
        }

        self.graphics_svo.render(params);
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
    fn raycast(&self, batch: PickerBatch) -> PickerBatchResult {
        let mut batch = batch;

        for ray in &mut batch.rays {
            ray.pos = self.svo_coord_space.cnv_block_pos(ray.pos);
        }
        for aabb in &mut batch.aabbs {
            aabb.pos = self.svo_coord_space.cnv_block_pos(aabb.pos);
        }

        let mut result = self.graphics_svo.raycast(batch);

        for ray in &mut result.rays {
            ray.pos = self.svo_coord_space.cnv_svo_pos(ray.pos);
        }

        result
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
    fn new(center: ChunkPos, dst: u32) -> SvoCoordSpace {
        SvoCoordSpace { center, dst }
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

        let pos = pos.to_block_pos();
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
        if dcx * dcx + dcz * dcz > r * r {
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

        let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 3.0 + 4.25, 32.0 * 10.0 + 20.5);
        let svo_pos = cs.cnv_block_pos(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * 0.0 + 4.25, 32.0 * 0.0 + 20.5));

        let cnv_back = cs.cnv_svo_pos(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    /// Tests transformation for negative coordinates.
    #[test]
    fn coord_space_negative() {
        let cs = SvoCoordSpace::new(ChunkPos::new(-1, -1, -1), 2);

        let world_pos = Point3::new(-16.25, -4.25, -20.5);
        let svo_pos = cs.cnv_block_pos(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 32.0 * 2.0 + 27.75, 32.0 * 2.0 + 11.5));

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
