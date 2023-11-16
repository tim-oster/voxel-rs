use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

use cgmath::Point3;

use crate::graphics;
use crate::systems::jobs::{JobHandle, JobSystemHandle};
use crate::world::chunk::{BlockPos, Chunk, ChunkPos};
use crate::world::octree::{OctantId, Position};
use crate::world::svo::{SerializedChunk, Svo};

// TODO should this support y axis changes or not? not consistent ATM
pub struct Manager<'js> {
    jobs: JobSystemHandle<'js>,
    tx: Sender<SerializedChunk>,
    rx: Receiver<SerializedChunk>,
    chunk_jobs: HashMap<ChunkPos, JobHandle>,

    svo: Svo<SerializedChunk>,
    octant_ids: HashMap<ChunkPos, OctantId>,
    has_changed: bool,
    coord_space: CoordSpace,
}

impl<'js> Manager<'js> {
    pub fn new(jobs: JobSystemHandle<'js>, render_distance: u32) -> Manager {
        let (tx, rx) = mpsc::channel::<SerializedChunk>();
        Manager {
            jobs,
            tx,
            rx,
            chunk_jobs: HashMap::new(),

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
        self.dequeue_chunk(&chunk.pos);

        if let Some(storage) = chunk.get_storage() {
            let pos = chunk.pos;
            let lod = chunk.lod;
            let tx = self.tx.clone();

            let handle = self.jobs.push(true, Box::new(move || {
                let serialized = SerializedChunk::new(pos, storage, lod);
                tx.send(serialized).unwrap();
            }));
            self.chunk_jobs.insert(pos, handle);
        }
    }

    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        self.dequeue_chunk(pos);

        if let Some(id) = self.octant_ids.remove(pos) {
            self.svo.remove_octant(id);
            self.has_changed = true;
        }
    }

    fn dequeue_chunk(&mut self, pos: &ChunkPos) {
        if let Some(handle) = self.chunk_jobs.remove(pos) {
            handle.cancel();
        }
    }

    pub fn update(&mut self, world_center: &ChunkPos, svo: &mut graphics::svo::Svo) {
        if self.coord_space.center != *world_center {
            let last_center = self.coord_space.center;
            self.shift_chunks(&last_center, world_center);
        }
        self.coord_space.center = *world_center;

        for _ in 0..50 {
            if let Ok(chunk) = self.rx.try_recv() {
                let chunk_pos = chunk.pos;
                self.chunk_jobs.remove(&chunk_pos);

                let svo_pos = Manager::world_to_svo_pos(&chunk_pos, &world_center, self.coord_space.dst);
                if self.is_out_of_bounds(&svo_pos) {
                    continue;
                }

                if let Some(id) = self.svo.set(svo_pos, Some(chunk)) {
                    self.octant_ids.insert(chunk_pos, id);
                    self.has_changed = true;
                }
            } else {
                break;
            }
        }

        if !self.has_changed {
            return;
        }

        self.has_changed = false;
        self.svo.serialize();
        svo.update(&mut self.svo, self.coord_space); // TODO this reference should not be mutable
    }

    // TODO this should use the coord space instead
    fn world_to_svo_pos(chunk_pos: &ChunkPos, world_center: &ChunkPos, render_distance: u32) -> Position {
        let offset_x = chunk_pos.x - world_center.x;
        let offset_z = chunk_pos.z - world_center.z;
        Position(
            (render_distance as i32 + offset_x) as u32,
            chunk_pos.y as u32,
            (render_distance as i32 + offset_z) as u32,
        )
    }

    fn is_out_of_bounds(&self, pos: &Position) -> bool {
        let r = self.coord_space.dst as i32;
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

                for dy in -r..r {
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

#[derive(Copy, Clone)]
pub struct CoordSpace {
    center: ChunkPos,
    dst: u32,
}

pub type CoordSpacePos = Point3<f32>;

impl CoordSpace {
    pub fn cnv_into_space(&self, pos: Point3<f32>) -> CoordSpacePos {
        let mut block_pos = BlockPos::from(pos);
        let delta = block_pos.chunk - self.center;

        let rd = self.dst as i32;
        block_pos.chunk.x = rd + delta.x;
        block_pos.chunk.z = rd + delta.z;

        block_pos.to_point()
    }

    pub fn cnv_out_of_space(&self, pos: CoordSpacePos) -> Point3<f32> {
        let mut block_pos = BlockPos::from(pos);

        let rd = self.dst as i32;
        // TODO y=rd is ignored here
        let delta = block_pos.chunk - ChunkPos::new(rd, 0, rd);

        block_pos.chunk.x = self.center.x + delta.x;
        block_pos.chunk.z = self.center.z + delta.z;

        block_pos.to_point()
    }
}

// TODO write more tests

#[cfg(test)]
mod test {
    use cgmath::Point3;

    use crate::systems::worldsvo::CoordSpace;
    use crate::world::chunk::ChunkPos;

    #[test]
    fn coord_space_positive() {
        let cs = CoordSpace {
            center: ChunkPos::new(4, 5, 12),
            dst: 2,
        };

        let world_pos = Point3::new(32.0 * 5.0 + 16.25, 32.0 * 2.0 + 4.2, 32.0 * 10.0 + 20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 3.0 + 16.25, 32.0 * 2.0 + 4.2, 32.0 * 0.0 + 20.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }

    #[test]
    fn coord_space_negative() {
        let cs = CoordSpace {
            center: ChunkPos::new(-1, -1, -1),
            dst: 2,
        };

        let world_pos = Point3::new(-16.25, 4.2, -20.5);
        let svo_pos = cs.cnv_into_space(world_pos);
        assert_eq!(svo_pos, Point3::new(32.0 * 2.0 + 15.75, 4.2, 32.0 * 2.0 + 11.5));

        let cnv_back = cs.cnv_out_of_space(svo_pos);
        assert_eq!(cnv_back, world_pos);
    }
}
