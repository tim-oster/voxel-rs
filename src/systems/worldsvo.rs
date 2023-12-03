use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::graphics;
use crate::graphics::svo::CoordSpace;
use crate::systems::jobs::{ChunkProcessor, JobSystem};
use crate::world::chunk::{Chunk, ChunkPos};
use crate::world::octree::{OctantId, Position};
use crate::world::svo::{SerializedChunk, Svo};

pub struct Manager {
    processor: ChunkProcessor<SerializedChunk>,
    svo: Svo<SerializedChunk>,
    octant_ids: HashMap<ChunkPos, OctantId>,
    has_changed: bool,
    coord_space: CoordSpace,
}

impl Manager {
    pub fn new(job_system: Rc<JobSystem>, render_distance: u32) -> Manager {
        Manager {
            processor: ChunkProcessor::new(job_system),
            svo: Svo::new(),
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
            self.svo.remove_octant(id);
            self.has_changed = true;
        }
    }

    /// Returns if the SVO still has in-work chunks or if there are unconsumed chunks in the buffer.
    pub fn has_pending_jobs(&self) -> bool {
        self.processor.has_pending()
    }

    pub fn update(&mut self, world_center: &ChunkPos, svo: &mut graphics::svo::Svo) {
        if self.coord_space.center != *world_center {
            let last_center = self.coord_space.center;
            self.shift_chunks(&last_center, world_center);
            self.has_changed = true;
        }
        self.coord_space.center = *world_center;

        for result in self.processor.get_results(50) {
            let svo_pos = Manager::world_to_svo_pos(&result.pos, &world_center, self.coord_space.dst);
            if self.is_out_of_bounds(&svo_pos) {
                continue;
            }

            if let Some(id) = self.svo.set(svo_pos, Some(result.value)) {
                self.octant_ids.insert(result.pos, id);
                self.has_changed = true;
            }
        }

        if !self.has_changed {
            return;
        }

        self.has_changed = false;
        self.svo.serialize();
        svo.update(&self.svo, Some(self.coord_space));
        self.svo.reset_changes();
    }

    fn world_to_svo_pos(chunk_pos: &ChunkPos, world_center: &ChunkPos, render_distance: u32) -> Position {
        let offset_x = chunk_pos.x - world_center.x;
        let offset_y = chunk_pos.y - world_center.y;
        let offset_z = chunk_pos.z - world_center.z;
        Position(
            (render_distance as i32 + offset_x) as u32,
            (render_distance as i32 + offset_y) as u32,
            (render_distance as i32 + offset_z) as u32,
        )
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

    fn shift_chunks(&mut self, last_center: &ChunkPos, new_center: &ChunkPos) {
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

                    let new_svo_pos = Manager::world_to_svo_pos(&chunk_pos, new_center, self.coord_space.dst);
                    if self.is_out_of_bounds(&new_svo_pos) {
                        octant_delete_set.insert(octant_id);
                        continue;
                    }

                    let old_octant = self.svo.replace(new_svo_pos, octant_id);
                    if let Some(id) = old_octant {
                        octant_delete_set.insert(id);
                    }
                }
            }
        }

        if !octant_delete_set.is_empty() {
            self.octant_ids.retain(|_, v| !octant_delete_set.contains(v));

            for id in octant_delete_set {
                self.svo.remove_octant(id);
            }
        }
    }
}

// TODO write tests
