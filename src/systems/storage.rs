use std::fs;
use std::fs::File;
use std::rc::Rc;
use std::sync::Arc;

use fastanvil::{Chunk, CurrentJavaChunk, Region};
use rustc_hash::FxHashMap;

use crate::gamelogic::content::blocks;
use crate::systems::jobs::{ChunkProcessor, ChunkResult, JobSystem};
use crate::world::chunk::{Chunk as EngineChunk, ChunkPos, ChunkStorageAllocator};

pub trait Storage {
    fn load(&mut self, pos: &ChunkPos, lod: u8);
    fn get_load_results(&mut self, limit: u32) -> Vec<ChunkResult<(Option<EngineChunk>, u8)>>;
    fn has_pending_jobs(&self) -> bool;
    fn dequeue_chunk(&mut self, pos: &ChunkPos);
}

pub struct NopStorage {
    loads: Vec<(ChunkPos, u8)>,
}

impl NopStorage {
    pub fn new() -> NopStorage {
        NopStorage { loads: Vec::new() }
    }
}

impl Storage for NopStorage {
    fn load(&mut self, pos: &ChunkPos, lod: u8) {
        self.loads.push((*pos, lod));
    }

    fn get_load_results(&mut self, limit: u32) -> Vec<ChunkResult<(Option<EngineChunk>, u8)>> {
        let mut result = Vec::<ChunkResult<(Option<EngineChunk>, u8)>>::new();
        for (pos, lod) in self.loads.drain(..) {
            result.push(ChunkResult { pos, value: (None, lod) });
        }
        result
    }

    fn has_pending_jobs(&self) -> bool {
        false
    }

    fn dequeue_chunk(&mut self, pos: &ChunkPos) {}
}

pub struct MinecraftStorage {
    alloc: Arc<ChunkStorageAllocator>,
    loaded_chunks: Arc<FxHashMap<(i32, i32), CurrentJavaChunk>>,
    processor: ChunkProcessor<(Option<EngineChunk>, u8)>,
}

#[allow(clippy::pedantic)]
impl MinecraftStorage {
    pub fn new(job_system: Rc<JobSystem>, alloc: Arc<ChunkStorageAllocator>, region_path: &str) -> Self {
        let mut loaded_chunks = FxHashMap::default();

        for entry in fs::read_dir(region_path).unwrap() {
            let path = entry.unwrap().path();
            let metadata = fs::metadata(&path).unwrap();
            if !metadata.is_file() {
                continue;
            }

            let parts = path.file_name().unwrap().to_str().unwrap().split(".").collect::<Vec<&str>>();
            if parts.len() != 4 || parts[0] != "r" || parts[3] != "mca" {
                continue;
            }

            let x = parts[1].parse::<i32>().unwrap();
            let z = parts[2].parse::<i32>().unwrap();

            let file = File::open(path).unwrap();
            let mut region = Region::from_stream(file).unwrap();

            for chunk in region.iter() {
                if chunk.is_err() {
                    continue;
                }
                let chunk = chunk.unwrap();
                let data: CurrentJavaChunk = fastnbt::from_bytes(chunk.data.as_slice()).unwrap();
                loaded_chunks.insert((x * 32 + chunk.x as i32, z * 32 + chunk.z as i32), data);
            }

            println!("preloaded region {x} {z}");
        }

        Self {
            alloc,
            loaded_chunks: Arc::new(loaded_chunks),
            processor: ChunkProcessor::new(job_system),
        }
    }
}

impl Storage for MinecraftStorage {
    fn load(&mut self, pos: &ChunkPos, lod: u8) {
        let pos = *pos;
        let alloc = self.alloc.clone();
        let loaded_chunks = self.loaded_chunks.clone();

        self.processor.enqueue(pos, false, move || {
            let region_x = (pos.x * 2) >> 5;
            let region_z = (pos.z * 2) >> 5;
            let stack_x = region_x * 32 + ((pos.x * 2) & 31);
            let stack_z = region_z * 32 + ((pos.z * 2) & 31);
            let data = [
                loaded_chunks.get(&(stack_x + 0, stack_z + 0)),
                loaded_chunks.get(&(stack_x + 1, stack_z + 0)),
                loaded_chunks.get(&(stack_x + 0, stack_z + 1)),
                loaded_chunks.get(&(stack_x + 1, stack_z + 1)),
            ];

            let mut storage = alloc.allocate();
            storage.construct_octants_with(5, |block_pos| {
                let data = &data[(block_pos.0 / 16 + (block_pos.2 / 16) * 2) as usize];
                let data = data.as_ref();
                if data.is_none() {
                    return None;
                }

                let actual_height = pos.y * 32 + block_pos.1 as i32;
                let block = data.unwrap().block(block_pos.0 as usize % 16, actual_height as isize, block_pos.2 as usize % 16);
                if let Some(block) = block {
                    if block.name().contains("_ore") {
                        return None;
                    }
                    if block.name().contains("_leaves") {
                        return Some(blocks::OAK_LEAVES);
                    }
                    if block.name().contains("_log") {
                        return Some(blocks::OAK_LOG);
                    }
                    if block.name().contains("_planks") {
                        return Some(blocks::OAK_PLANKS);
                    }
                    return match block.name() {
                        "minecraft:air" | "minecraft:cave_air" | "minecraft:tall_seagrass" | "minecraft:seagrass" | "minecraft:kelp" | "minecraft:kelp_plant" => None,
                        "minecraft:dirt" => Some(blocks::DIRT),
                        "minecraft:grass_block" => Some(blocks::GRASS),
                        "minecraft:gravel" | "minecraft:clay" => Some(blocks::GRAVEL),
                        "minecraft:sand" | "minecraft:sandstone" => Some(blocks::SAND),
                        "minecraft:water" => Some(blocks::WATER),
                        "minecraft:stone" | "minecraft:andesite" | "minecraft:diorite" | "minecraft:deepslate" | "minecraft:tuff" | "minecraft:granite" => Some(blocks::STONE),
                        "minecraft:cobblestone" => Some(blocks::COBBLESTONE),
                        _ => {
                            //println!("{}", block.name());
                            None
                        }
                    };
                }
                None
            });

            let chunk = EngineChunk::new(pos, lod, storage);
            (Some(chunk), lod)
        });
    }

    fn get_load_results(&mut self, limit: u32) -> Vec<ChunkResult<(Option<EngineChunk>, u8)>> {
        self.processor.get_results(limit)
    }

    fn has_pending_jobs(&self) -> bool {
        self.processor.has_pending()
    }

    fn dequeue_chunk(&mut self, pos: &ChunkPos) {
        self.processor.dequeue(pos);
    }
}
