use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::storage::chunk;

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
    block_id: Option<chunk::BlockId>,
}

pub struct Svo {
    pub header_mask: u16,
    pub depth: i32,
    pub descriptors: Vec<u32>,
}

impl Svo {
    pub fn new_from_chunk(chunk: &chunk::Chunk) -> Svo {
        let mut leaf_map = HashMap::new();

        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let block = chunk.blocks[chunk::Chunk::get_block_index(x, y, z)];
                    if block == chunk::NO_BLOCK {
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
            let result = merge_nodes(&merged_map);
            octant_count += result.len();
            merged_map = result;
        }

        let root_node = merged_map.get(merged_map.keys().next().unwrap()).unwrap().borrow();
        // let mut descriptors = Vec::with_capacity((4 + 8) * octant_count);

        // create fake root node
        let mut root_mask = 0;
        for (idx, node) in root_node.children.iter().enumerate() {
            if node.is_none() {
                continue;
            }
            root_mask |= (1 << idx) << 8;

            let node = node.as_ref().unwrap().borrow();
            if node.block_id.is_some() {
                root_mask |= 1 << idx;
            }
        }
        // descriptors.push(root_mask);
        // descriptors.push(0);
        // descriptors.push(0);
        // descriptors.push(0);
        // descriptors.push(5); // absolute pointer to where the actual octree starts

        // add actual descriptors from octree
        let descriptors = build_descriptors(root_node);
        // descriptors.extend(svo_descriptors);

        let depth = max_depth + 1; // plus one because of fake root node
        Svo { header_mask: root_mask, depth, descriptors }
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

            let child_svo = build_descriptors(node);
            svo[4 + idx] = svo.len() as u32 - 4 - idx as u32;
            svo[4 + idx] |= 1 << 31; // flag as relative pointer
            svo.extend(child_svo);
        }
    }

    svo
}

#[cfg(test)]
mod tests_svo {
    use super::chunk::Chunk;

    #[test]
    fn chunk_build_svo_one_sub_tree() {
        let mut chunk = Chunk::new();
        chunk.set_block(0, 0, 0, 100);
        chunk.set_block(1, 1, 1, 200);

        let svo = super::Svo::new_from_chunk(&chunk);
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
            (1 << 31) | 8, 0, 0, 0,
            0, 0, 0, 0,
            // second octant header
            1 << 8,
            0,
            0,
            0,
            // second octant body
            (1 << 31) | 8, 0, 0, 0,
            0, 0, 0, 0,
            // third octant header
            1 << 8,
            0,
            0,
            0,
            // third octant body
            (1 << 31) | 8, 0, 0, 0,
            0, 0, 0, 0,
            // fourth octant header
            ((1 | 128) << 8) | (1 | 128),
            0,
            0,
            0,
            // fourth octant body
            (1 << 31) | 8, 0, 0, 0,
            0, 0, 0, 0,
            // fifth octant header
            0,
            0,
            0,
            0,
            // fifth octant body
            100, 0, 0, 0,
            0, 0, 0, 200,
        ]);
    }

    #[test]
    fn chunk_build_svo_multiple_sub_trees() {
        let mut chunk = Chunk::new();
        chunk.set_block(31, 0, 0, 1);
        chunk.set_block(0, 31, 0, 2);
        chunk.set_block(0, 0, 31, 3);

        let svo = super::Svo::new_from_chunk(&chunk);
        assert_eq!(svo.descriptors, vec![
            // fake root header
            (2 | 4 | 16) << 8,
            0,
            0,
            0,
            // fake root body
            5,

            // core octant header
            (2 << 8) << 16,
            4 << 8,
            16 << 8,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 4 * 12), 0,
            (1 << 31) | (4 + 8 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // header 1
            2 << 8 << 16,
            0,
            0,
            0,
            // body 1
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 2
            2 << 8 << 16,
            0,
            0,
            0,
            // body 2
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 3
            ((2 << 8) | 2) << 16,
            0,
            0,
            0,
            // body 3
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // header 1
            0,
            4 << 8,
            0,
            0,
            // body 1
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 2
            0,
            4 << 8,
            0,
            0,
            // body 2
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 3
            0,
            4 << 8 | 4,
            0,
            0,
            // body 3
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // header 1
            0,
            0,
            16 << 8,
            0,
            // body 1
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 2
            0,
            0,
            16 << 8,
            0,
            // body 2
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 3
            0,
            0,
            16 << 8 | 16,
            0,
            // body 3
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,
        ]);
    }
}