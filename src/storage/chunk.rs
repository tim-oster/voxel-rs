// TODO does it make sense to use 16 instead of 32 for faster octree rebuilds?

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ops::Add;
use std::rc::Rc;

use crate::storage::svo::SVO;

pub type BlockId = u32;

pub const NO_BLOCK: BlockId = 0;

pub struct Chunk {
    blocks: [BlockId; 32 * 32 * 32],
}

// TODO make positions unsigned?

impl Chunk {
    pub fn new() -> Chunk {
        Chunk {
            blocks: [NO_BLOCK; 32 * 32 * 32],
        }
    }

    fn get_block_index(x: i32, y: i32, z: i32) -> usize {
        // TODO optimize by using morton codes?
        (x + y * 32 + z * 32 * 32) as usize
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.blocks[Chunk::get_block_index(x, y, z)]
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockId) {
        self.blocks[Chunk::get_block_index(x, y, z)] = block;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunk_get_and_set_blocks() {
        let mut chunk = super::Chunk::new();

        let block = chunk.get_block(10, 20, 30);
        assert_eq!(block, super::NO_BLOCK);

        chunk.set_block(10, 20, 30, 99);
        assert_eq!(chunk.blocks[10 + 20 * 32 + 30 * 32 * 32], 99);

        let block = chunk.get_block(10, 20, 30);
        assert_eq!(block, 99);
    }
}

// TODO move SVO code to svo.rs

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
struct NodePos {
    x: u8,
    y: u8,
    z: u8,
}

struct Node {
    child_mask: u8,
    leaf_mask: u8,
    children: [Option<Rc<RefCell<Node>>>; 8],
    block_id: Option<BlockId>,
}

impl Chunk {
    pub fn build_svo(&self) -> SVO {
        let mut leaf_map = HashMap::new();

        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let block = self.blocks[Chunk::get_block_index(x, y, z)];
                    if block == NO_BLOCK {
                        continue;
                    }

                    let pos = NodePos { x: x as u8, y: y as u8, z: z as u8 };
                    leaf_map.insert(pos, Rc::new(RefCell::new(Node {
                        child_mask: 1,
                        leaf_mask: 1,
                        children: Default::default(),
                        block_id: Some(block),
                    })));
                }
            }
        }

        // build an octree by merging child octrees five times into one (log2(32) = 5)
        let mut merged_map = leaf_map;
        let mut octant_count = 0;

        let max_depth = 32f32.log2().ceil() as i32;
        for _ in 0..max_depth {
            let result = Chunk::merge_nodes(&merged_map);
            octant_count += result.len();
            merged_map = result;
        }

        let root_node = merged_map.get(merged_map.keys().next().unwrap()).unwrap().borrow();
        let mut descriptors = Vec::with_capacity((4 + 8) * octant_count);

        // create fake root node
        descriptors.push(1 << 8);
        descriptors.push(0);
        descriptors.push(0);
        descriptors.push(0);
        descriptors.push(5); // absolute pointer to where the actual octree starts

        // add actual descriptors from octree
        let svo_descriptors = Chunk::build_descriptors(root_node);
        descriptors.extend(svo_descriptors);

        SVO {
            max_depth,
            max_depth_exp2: (-max_depth as f32).exp2(),
            descriptors,
        }
    }

    fn merge_nodes(nodes: &HashMap<NodePos, Rc<RefCell<Node>>>) -> HashMap<NodePos, Rc<RefCell<Node>>> {
        let mut merged_map = HashMap::new();

        for (pos, node) in nodes.iter() {
            let merged_pos = NodePos { x: pos.x / 2, y: pos.y / 2, z: pos.z / 2 };
            if !merged_map.contains_key(&merged_pos) {
                merged_map.insert(merged_pos, Rc::new(RefCell::new(Node {
                    child_mask: 0,
                    leaf_mask: 0,
                    children: Default::default(),
                    block_id: None,
                })));
            }
            let merged_node = merged_map.get_mut(&merged_pos).unwrap();

            let idx = (pos.x % 2) + (pos.y % 2) * 2 + (pos.z % 2) * 4;
            merged_node.borrow_mut().child_mask |= 1 << idx;
            if node.borrow().block_id.is_some() {
                merged_node.borrow_mut().leaf_mask |= 1 << idx;
            }
            merged_node.borrow_mut().children[idx as usize] = Some(Rc::clone(node));
        }

        merged_map
    }

    fn build_descriptors(node: Ref<Node>) -> Vec<u32> {
        let mut svo = Vec::<u32>::with_capacity(12);
        svo.extend(std::iter::repeat(0).take(12));

        for (idx, node) in node.children.iter().enumerate() {
            if node.is_none() {
                continue;
            }

            let node = node.as_ref().unwrap().borrow();

            if let Some(block) = node.block_id {
                svo[4 + idx] = block;
            } else {
                let mut mask = ((node.child_mask as u32) << 8) | node.leaf_mask as u32;
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                svo[(idx / 2) as usize] |= mask;

                let child_svo = Chunk::build_descriptors(node);
                svo[4 + idx] = 8 - idx as u32;
                svo[4 + idx] |= 1 << 31; // flag as relative pointer
                svo.extend(child_svo);
            }
        }

        svo
    }
}

// struct Node {
//     ptr: usize,
//     child_mask: u8,
//     leaf_mask: u8,
// }
//
// impl Chunk {
//     pub fn build_svo(&self) -> SVO {
//         // build all leaf nodes from non-empty blocks in the chunk
//         let mut leaf_map = HashMap::new();
//         let mut leaf_svo = Vec::with_capacity((4 + 8) * (16 * 16 * 16));
//
//         for z in 0..32 {
//             for y in 0..32 {
//                 for x in 0..32 {
//                     let block = self.blocks[Chunk::get_block_index(x, y, z)];
//                     if block == NO_BLOCK {
//                         continue;
//                     }
//
//                     let pos = NodePos { x: (x / 2) as u8, y: (y / 2) as u8, z: (z / 2) as u8 };
//                     if !leaf_map.contains_key(&pos) {
//                         leaf_map.insert(pos, Node {
//                             ptr: leaf_svo.len(),
//                             child_mask: 0,
//                             leaf_mask: 0,
//                         });
//                         leaf_svo.extend(std::iter::repeat(0).take(12));
//                     }
//                     let node = leaf_map.get_mut(&pos).unwrap();
//
//                     let idx = (x % 2) + (y % 2) * 2 + (z % 2) * 4;
//                     node.child_mask |= 1 << idx;
//                     node.leaf_mask |= 1 << idx;
//                     leaf_svo[node.ptr + 4 + idx as usize] = block as u32;
//                 }
//             }
//         }
//
//         // build an octree by merging child octrees five times into one (log2(32) = 5)
//         let mut input_map = leaf_map;
//         for i in 0..5 {
//             let (merged_map, svo) = Chunk::merge_nodes(&input_map);
//             input_map = merged_map;
//         }
//
//         // // merge all leaf nodes into an octree structure
//         // let mut svo_stack = vec![leaf_svo];
//         // let mut input_map = leaf_map;
//         // while input_map.len() > 1 {
//         //     let (merged_map, svo) = Chunk::merge_nodes(&input_map);
//         //     svo_stack.push(svo);
//         //     input_map = merged_map;
//         // }
//         //
//         // let mut fake_svo = Vec::new();
//         // fake_svo.push((input_map.values().next().unwrap().child_mask as u32) << 8);
//         // fake_svo.push(0);
//         // fake_svo.push(0);
//         // fake_svo.push(0);
//         // fake_svo.push(fake_svo.len() as u32 + 1);
//         // svo_stack.push(fake_svo);
//         //
//         // let final_svo_size = svo_stack.iter().fold(0, |acc, x| acc.add(x.len()));
//         // let mut final_svo = Vec::with_capacity(final_svo_size);
//         // for svo in svo_stack.into_iter().rev() {
//         //     final_svo.extend(svo);
//         // }
//
//         let max_depth = 32f32.log2().ceil() as i32;
//         SVO {
//             max_depth,
//             max_depth_exp2: (-max_depth as f32).exp2(),
//             descriptors: vec![],
//         }
//     }
//
//     fn merge_nodes(nodes: &HashMap<NodePos, Node>) -> (HashMap<NodePos, Node>, Vec<u32>) {
//         let mut merged_map = HashMap::new();
//         // Over-allocate the svo vector instead of trying to precisely estimate its size.
//         // In the worst case scenario (in which no nodes can be merged) the svo will exactly
//         // big enough.
//         let mut svo = Vec::with_capacity((4 + 8) * nodes.len());
//
//         struct MissingPointer {
//             index: usize,
//             ptr: u32,
//         }
//         let mut missing_pointers = Vec::new();
//
//         for (pos, node) in nodes.iter() {
//             // TODO duplicated code
//             let pos = NodePos { x: pos.x / 2, y: pos.y / 2, z: pos.z / 2 };
//             if !merged_map.contains_key(&pos) {
//                 merged_map.insert(pos, Node {
//                     ptr: svo.len(),
//                     child_mask: 0,
//                     leaf_mask: 0,
//                 });
//                 svo.extend(std::iter::repeat(0).take(12));
//             }
//             let merged_node = merged_map.get_mut(&pos).unwrap();
//
//             let idx = (pos.x % 2) + (pos.y % 2) * 2 + (pos.z % 2) * 4;
//             merged_node.child_mask |= 1 << idx;
//
//             let mut mask = ((node.child_mask as u32) << 8) | node.leaf_mask as u32;
//             if (idx % 2) != 0 {
//                 mask <<= 16;
//             }
//             svo[merged_node.ptr + (idx / 2) as usize] |= mask;
//
//             missing_pointers.push(MissingPointer {
//                 index: merged_node.ptr + 4 + idx as usize,
//                 ptr: node.ptr as u32,
//             });
//         }
//
//         for mp in missing_pointers {
//             let offset = svo.len() - mp.index;
//             svo[mp.index] = offset as u32 + mp.ptr;
//             // set relative pointer bit to indicate relative offsets
//             svo[mp.index] |= 1 << 31;
//         }
//
//         (merged_map, svo)
//     }
// }

#[cfg(test)]
mod tests_svo {
    #[test]
    fn chunk_build_svo_one_octant() {
        let mut chunk = super::Chunk::new();
        chunk.set_block(0, 0, 0, 100);
        chunk.set_block(1, 1, 1, 200);

        let svo = chunk.build_svo();
        assert_eq!(svo.descriptors, vec![
            // fake root header
            1 << 8,
            0,
            0,
            0,
            // fake root body
            5,
            // first octant header
            1 << 8,
            0,
            0,
            0,
            // first octant body
            (1 << 31) | 8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // second octant header
            1 << 8,
            0,
            0,
            0,
            // second octant body
            (1 << 31) | 8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // third octant header
            1 << 8,
            0,
            0,
            0,
            // third octant body
            (1 << 31) | 8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // fourth octant header
            ((1 | 128) << 8) | (1 | 128),
            0,
            0,
            0,
            // fourth octant body
            (1 << 31) | 8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // fifth octant header
            0,
            0,
            0,
            0,
            // fifth octant body
            100,
            0,
            0,
            0,
            0,
            0,
            0,
            200,
        ]);
    }
}
