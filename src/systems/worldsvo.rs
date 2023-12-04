use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use cgmath::{EuclideanSpace, Point3};

use crate::graphics;
use crate::graphics::svo_picker::{PickerBatch, PickerBatchResult};
use crate::systems::jobs::{ChunkProcessor, JobSystem};
use crate::systems::physics::Raycaster;
use crate::world;
use crate::world::chunk::{BlockPos, Chunk, ChunkPos};
use crate::world::octree::{OctantId, Position};
use crate::world::svo::SerializedChunk;

pub struct Svo {
    processor: ChunkProcessor<SerializedChunk>,

    world_svo: world::Svo<SerializedChunk>,
    graphics_svo: graphics::Svo,

    octant_ids: HashMap<ChunkPos, OctantId>,
    has_changed: bool,
    coord_space: CoordSpace,
}

impl Svo {
    pub fn new(job_system: Rc<JobSystem>, graphics_svo: graphics::Svo, render_distance: u32) -> Svo {
        Svo {
            processor: ChunkProcessor::new(job_system),
            world_svo: world::Svo::new(),
            graphics_svo,
            octant_ids: HashMap::new(),
            has_changed: false,
            coord_space: CoordSpace {
                center: ChunkPos::new(0, 0, 0),
                dst: render_distance,
            },
        }
    }

    pub fn set_chunk(&mut self, chunk: &Chunk) {
        self.processor.dequeue(&chunk.pos);

        if let Some(storage) = chunk.get_storage() {
            let pos = chunk.pos.clone();
            let lod = chunk.lod;
            self.processor.enqueue(chunk.pos, true, move || SerializedChunk::new(pos, storage, lod));
        }
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
        self.coord_space.dst
    }

    pub fn update(&mut self, world_center: &ChunkPos) {
        if self.coord_space.center != *world_center {
            let last_center = self.coord_space.center;
            self.coord_space.center = *world_center;
            self.shift_chunks(&last_center);
            self.has_changed = true;
        }

        for result in self.processor.get_results(50) {
            // TODO write better conversion api?
            let svo_pos = result.pos.to_block_pos();
            let svo_pos = self.coord_space.cnv_into_space(svo_pos.cast().unwrap());
            let svo_pos = Position((svo_pos.x / 32.0) as u32, (svo_pos.y / 32.0) as u32, (svo_pos.z / 32.0) as u32);

            if self.is_out_of_bounds(&svo_pos) {
                continue;
            }

            if let Some(id) = self.world_svo.set(svo_pos, Some(result.value)) {
                self.octant_ids.insert(result.pos, id);
                self.has_changed = true;
            }
        }

        if !self.has_changed {
            return;
        }

        self.has_changed = false;
        self.world_svo.serialize();
        self.graphics_svo.update(&self.world_svo);
        self.world_svo.reset_changes();
    }

    fn is_out_of_bounds(&self, pos: &Position) -> bool {
        let r = self.coord_space.dst as i32;

        let dcy = pos.1 as i32 - r;
        if dcy < -r || dcy > r {
            return true;
        }

        let dcx = pos.0 as i32 - r;
        let dcz = pos.2 as i32 - r;
        dcx * dcx + dcz * dcz > r * r
    }

    fn shift_chunks(&mut self, last_center: &ChunkPos) {
        let r = self.coord_space.dst as i32;
        let mut octant_delete_set = HashSet::new();

        for dx in -r..=r {
            for dz in -r..=r {
                if dx * dx + dz * dz > r * r {
                    continue;
                }

                for dy in -r..=r {
                    let chunk_pos = ChunkPos {
                        x: last_center.x + dx,
                        y: last_center.y + dy,
                        z: last_center.z + dz,
                    };

                    let octant_id = self.octant_ids.get(&chunk_pos);
                    if octant_id.is_none() {
                        continue;
                    }
                    let octant_id = *octant_id.unwrap();
                    octant_delete_set.remove(&octant_id);

                    // TODO write better conversion api?
                    let new_svo_pos = chunk_pos.to_block_pos();
                    let new_svo_pos = self.coord_space.cnv_into_space(new_svo_pos.cast().unwrap());
                    let new_svo_pos = Position((new_svo_pos.x / 32.0) as u32, (new_svo_pos.y / 32.0) as u32, (new_svo_pos.z / 32.0) as u32);

                    if self.is_out_of_bounds(&new_svo_pos) {
                        octant_delete_set.insert(octant_id);
                        continue;
                    }

                    let old_octant = self.world_svo.replace(new_svo_pos, octant_id);
                    if let Some(id) = old_octant {
                        octant_delete_set.insert(id);
                    }
                }
            }
        }

        if !octant_delete_set.is_empty() {
            self.octant_ids.retain(|_, v| !octant_delete_set.contains(v));

            for id in octant_delete_set {
                self.world_svo.remove_octant(id);
            }
        }
    }
}

// TODO doc that this is the decorator part? or whatever this is called
impl Svo {
    pub fn reload_resources(&mut self) {
        self.graphics_svo.reload_resources();
    }

    pub fn render(&self, params: graphics::svo::RenderParams) {
        let mut params = params;

        // translate camera position into SVO
        params.cam_pos = self.coord_space.cnv_into_space(params.cam_pos);

        // translate selected voxel position into SVO
        if let Some(pos) = params.selected_voxel {
            params.selected_voxel = Some(self.coord_space.cnv_into_space(pos));
        }

        self.graphics_svo.render(params);
    }

    pub fn get_stats(&self) -> graphics::svo::Stats {
        self.graphics_svo.get_stats()
    }
}

impl Raycaster for Svo {
    fn raycast(&self, batch: PickerBatch) -> PickerBatchResult {
        let mut batch = batch;

        for ray in &mut batch.rays {
            ray.pos = self.coord_space.cnv_into_space(ray.pos);
        }
        for aabb in &mut batch.aabbs {
            aabb.pos = self.coord_space.cnv_into_space(aabb.pos);
        }

        let mut result = self.graphics_svo.raycast(batch);

        for ray in &mut result.rays {
            ray.pos = self.coord_space.cnv_out_of_space(ray.pos);
        }

        result
    }
}

// TODO write tests


#[derive(Copy, Clone)]
pub struct CoordSpace {
    pub center: ChunkPos,
    pub dst: u32,
}

pub type CoordSpacePos = Point3<f32>;

#[allow(dead_code)]
impl CoordSpace {
    pub fn new(center: ChunkPos, dst: u32) -> CoordSpace {
        CoordSpace { center, dst }
    }

    pub fn cnv_into_space(&self, pos: Point3<f32>) -> CoordSpacePos {
        let mut block_pos = BlockPos::from(pos);
        let delta = block_pos.chunk - self.center;

        let rd = self.dst as i32;
        block_pos.chunk.x = rd + delta.x;
        block_pos.chunk.y = rd + delta.y;
        block_pos.chunk.z = rd + delta.z;

        block_pos.to_point()
    }

    pub fn cnv_out_of_space(&self, pos: CoordSpacePos) -> Point3<f32> {
        let mut block_pos = BlockPos::from(pos);

        let rd = self.dst as i32;
        let delta = block_pos.chunk - ChunkPos::new(rd, rd, rd);

        block_pos.chunk.x = self.center.x + delta.x;
        block_pos.chunk.y = self.center.y + delta.y;
        block_pos.chunk.z = self.center.z + delta.z;

        block_pos.to_point()
    }
}

#[cfg(test)]
mod coord_space_tests {
    use cgmath::Point3;

    use crate::systems::worldsvo::CoordSpace;
    use crate::world::chunk::ChunkPos;

    /// Test transformation for positive coordinates.
    #[test]
    fn coord_space_positive() {
        let cs = CoordSpace {
            center: ChunkPos::new(4, 5, 12),
            dst: 2,
        };

        let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 3.0 + 4.25, 32.0 * 10.0 + 20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * 0.0 + 4.25, 32.0 * 0.0 + 20.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    /// Test transformation for negative coordinates.
    #[test]
    fn coord_space_negative() {
        let cs = CoordSpace {
            center: ChunkPos::new(-1, -1, -1),
            dst: 2,
        };

        let world_pos = Point3::new(-16.25, -4.25, -20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 32.0 * 2.0 + 27.75, 32.0 * 2.0 + 11.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }
}
