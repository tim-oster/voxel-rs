use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

use crate::systems::jobs::{JobHandle, JobSystemHandle};
use crate::world::chunk::{Chunk, ChunkPos};
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
    world_center: Option<ChunkPos>,
    render_distance: u32,
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
            world_center: None,
            render_distance,
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

    pub fn update(&mut self, world_center: &ChunkPos) -> Option<&mut Svo<SerializedChunk>> {
        if let Some(last_center) = self.world_center {
            if last_center != *world_center {
                self.shift_chunks(&last_center, world_center);
            }
        }
        self.world_center = Some(*world_center);

        for _ in 0..50 {
            if let Ok(chunk) = self.rx.try_recv() {
                let chunk_pos = chunk.pos;
                self.chunk_jobs.remove(&chunk_pos);

                let svo_pos = Manager::world_to_svo_pos(&chunk_pos, &world_center, self.render_distance);
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
            return None;
        }

        self.has_changed = false;
        self.svo.serialize();

        Some(&mut self.svo)
    }

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
        let dcx = pos.0 as i32 - self.render_distance as i32;
        let dcz = pos.2 as i32 - self.render_distance as i32;
        let r = self.render_distance as i32;
        dcx * dcx + dcz * dcz > r * r
    }

    fn shift_chunks(&mut self, last_center: &ChunkPos, new_center: &ChunkPos) {
        let r = self.render_distance as i32;
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

                    let new_svo_pos = Manager::world_to_svo_pos(&chunk_pos, new_center, self.render_distance);
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
