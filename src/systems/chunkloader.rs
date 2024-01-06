use std::cmp;

use cgmath::Point3;
use rustc_hash::FxHashMap;

use crate::world::chunk::ChunkPos;

pub struct ChunkLoader {
    radius: u32,
    start_y: i32,
    end_y: i32,

    last_pos: Option<ChunkPos>,
    loaded_chunks: FxHashMap<ChunkPos, u8>,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Ord, PartialOrd)]
pub enum ChunkEvent {
    Load { pos: ChunkPos, lod: u8 },
    Unload { pos: ChunkPos },
    LodChange { pos: ChunkPos, lod: u8 },
}

impl ChunkEvent {
    pub fn get_pos(&self) -> &ChunkPos {
        match self {
            ChunkEvent::Load { pos, .. } => pos,
            ChunkEvent::Unload { pos, .. } => pos,
            ChunkEvent::LodChange { pos, .. } => pos,
        }
    }
}

impl ChunkLoader {
    pub fn new(radius: u32, start_y: i32, end_y: i32) -> ChunkLoader {
        assert!(start_y < end_y);
        ChunkLoader {
            radius,
            start_y,
            end_y,

            last_pos: None,
            loaded_chunks: FxHashMap::default(),
        }
    }

    pub fn get_radius(&self) -> u32 {
        self.radius
    }

    /// Returns a list of chunk events that occurred due to changes to the target position.
    /// Might be empty if the position did not change.
    pub fn update(&mut self, pos: Point3<f32>) -> Vec<ChunkEvent> {
        let mut events = Vec::new();

        let current_pos = ChunkPos::from_block_pos(pos.x as i32, pos.y as i32, pos.z as i32);
        if self.last_pos == Some(current_pos) {
            return events;
        }

        // checks chunks in radius around current position and create events to load new chunks
        // or update LODs of loaded ones
        let r = self.radius as i32;
        for dx in -r..=r {
            for dz in -r..=r {
                // ensure that chunks are only loaded inside the given radius
                if dx * dx + dz * dz > r * r {
                    continue;
                }

                let mut pos = ChunkPos::new(current_pos.x + dx, 0, current_pos.z + dz);
                let lod = ChunkLoader::calculate_lod(&current_pos, &pos);

                for y in self.start_y..self.end_y {
                    // ensure that y is still within loading radius
                    let dy = y - current_pos.y;
                    if dy < -r || dy > r {
                        continue;
                    }

                    pos.y = y;

                    if let Some(old_lod) = self.loaded_chunks.get(&pos) {
                        if *old_lod != lod {
                            events.push(ChunkEvent::LodChange { pos, lod });
                            self.loaded_chunks.insert(pos, lod);
                        }
                    } else {
                        events.push(ChunkEvent::Load { pos, lod });
                        self.loaded_chunks.insert(pos, lod);
                    }
                }
            }
        }

        // create delete events for chunks outside the loading radius
        let mut delete_list = Vec::new();
        for pos in self.loaded_chunks.keys() {
            let dx = (pos.x - current_pos.x).abs();
            let dy = (pos.y - current_pos.y).abs();
            let dz = (pos.z - current_pos.z).abs();

            if (dy < -r || dy > r) || dx * dx + dz * dz > r * r {
                delete_list.push(*pos);
                events.push(ChunkEvent::Unload { pos: *pos });
            }
        }
        for pos in delete_list {
            self.loaded_chunks.remove(&pos);
        }

        // sort events by the targeted chunk's distance to the current position
        events.sort_by(|a, b| {
            let da = a.get_pos().dst_sq(&current_pos);
            let db = b.get_pos().dst_sq(&current_pos);
            da.partial_cmp(&db).unwrap_or(cmp::Ordering::Equal)
        });

        events
    }

    fn calculate_lod(center: &ChunkPos, pos: &ChunkPos) -> u8 {
        match pos.dst_2d_sq(center).sqrt() as i32 {
            0..=6 => 5,
            7..=12 => 4,
            13..=19 => 3,
            _ => 2,
        }
    }

    pub fn is_loaded(&self, pos: &ChunkPos) -> bool {
        self.loaded_chunks.contains_key(pos)
    }

    pub fn add_loaded_chunk(&mut self, pos: ChunkPos, lod: u8) {
        self.loaded_chunks.insert(pos, lod);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use cgmath::Point3;

    use crate::systems::chunkloader::{ChunkEvent, ChunkLoader};
    use crate::world::chunk::ChunkPos;

    /// Asserts that chunks inside the specified radius are properly loaded with an accurate LOD
    /// and also unloaded.
    #[test]
    fn load_and_unload() {
        let mut cl = ChunkLoader::new(1, 0, 1);

        let mut events = cl.update(Point3::new(0.0, 0.0, 0.0));
        events.sort();
        assert_eq!(events, vec![
            ChunkEvent::Load { pos: ChunkPos { x: -1, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 0, y: 0, z: -1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 0, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 0, y: 0, z: 1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 1, y: 0, z: 0 }, lod: 5 },
        ]);

        // stay inside the same chunk
        let events = cl.update(Point3::new(16.0, 16.0, 16.0));
        assert!(events.is_empty());

        // change to neighbor chunk causes partial unloading of old chunks and additional loading
        // of new chunks
        let mut events = cl.update(Point3::new(32.0, 0.0, 0.0));
        events.sort();
        assert_eq!(events, vec![
            ChunkEvent::Load { pos: ChunkPos { x: 1, y: 0, z: -1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 1, y: 0, z: 1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 2, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Unload { pos: ChunkPos { x: -1, y: 0, z: 0 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 0, y: 0, z: -1 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 0, y: 0, z: 1 } },
        ]);

        // change to a chunk outside the current radius to cause a full unload/load
        let mut events = cl.update(Point3::new(128.0, 0.0, 0.0));
        events.sort();
        assert_eq!(events, vec![
            ChunkEvent::Load { pos: ChunkPos { x: 3, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 4, y: 0, z: -1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 4, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 4, y: 0, z: 1 }, lod: 5 },
            ChunkEvent::Load { pos: ChunkPos { x: 5, y: 0, z: 0 }, lod: 5 },
            ChunkEvent::Unload { pos: ChunkPos { x: 0, y: 0, z: 0 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 1, y: 0, z: -1 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 1, y: 0, z: 0 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 1, y: 0, z: 1 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 2, y: 0, z: 0 } },
        ]);

        // changing y above current loading radius to cause a full unload
        let mut events = cl.update(Point3::new(128.0, 64.0, 0.0));
        events.sort();
        assert_eq!(events, vec![
            ChunkEvent::Unload { pos: ChunkPos { x: 3, y: 0, z: 0 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 4, y: 0, z: -1 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 4, y: 0, z: 0 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 4, y: 0, z: 1 } },
            ChunkEvent::Unload { pos: ChunkPos { x: 5, y: 0, z: 0 } },
        ]);

        // staying at unloaded y and changing to a different position does nothing
        let events = cl.update(Point3::new(0.0, 64.0, 0.0));
        assert!(events.is_empty());
    }

    /// Asserts that already loaded chunks are changing their LOD depending on their distance
    /// to the current position.
    #[test]
    fn changing_lod() {
        let mut cl = ChunkLoader::new(25, 0, 1);

        // scale is comprised of all chunk load LOD values
        let events = cl.update(Point3::new(0.0, 0.0, 0.0));
        let z0 = vec![2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2];
        let z1 = vec![2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2];
        assert_eq!(get_lod_scale_on_x_axis(&events, -1), z1);
        assert_eq!(get_lod_scale_on_x_axis(&events, 0), z0);
        assert_eq!(get_lod_scale_on_x_axis(&events, 1), z1);

        // When moving one chunk in positive x, only one chunk per lod level is expected to change
        // as well, as everything shifts one chunk to the left. Additionally, one new chunk is added
        // for z=0.
        let events = cl.update(Point3::new(32.0, 0.0, 0.0));
        let change = vec![2, 3, 4, 5, 4, 3, 2];
        assert_eq!(get_lod_scale_on_x_axis(&events, -1), change);
        assert_eq!(get_lod_scale_on_x_axis(&events, 0), change);
        assert_eq!(get_lod_scale_on_x_axis(&events, 1), change);
    }

    fn get_lod_scale_on_x_axis(events: &Vec<ChunkEvent>, z: i32) -> Vec<u8> {
        let mut columns = HashMap::new();

        for evt in events {
            if let Some((pos, lod)) = match evt {
                ChunkEvent::Load { pos, lod } => Some((pos, lod)),
                ChunkEvent::LodChange { pos, lod } => Some((pos, lod)),
                _ => None,
            } {
                if pos.z != z {
                    continue;
                }
                columns.insert(pos.x, *lod);
            }
        }

        let mut scale = Vec::new();
        let mut xs = columns.keys().map(|x| *x).collect::<Vec<i32>>();
        xs.sort();
        for x in xs {
            scale.push(columns[&x]);
        }
        scale
    }
}
