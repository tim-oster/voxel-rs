use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::rc::Rc;

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
struct NodePos {
    x: u8,
    y: u8,
    z: u8,
}

struct TreeNode {
    id: u64,
    nodes: [Option<Rc<RefCell<TreeNode>>>; 8],
    color: Option<i32>,
}

struct OctantBlock {
    id: u64,
    encoded: [u32; 12],
    children: [Option<Box<OctantBlock>>; 8],
    unpopulated_pointers: Vec<UnpopulatedPointer>,
}

struct UnpopulatedPointer {
    octant_index: i8,
    octant_block_ref: u64,
}

impl TreeNode {
    fn new() -> Rc<RefCell<TreeNode>> {
        Rc::new(RefCell::new(TreeNode {
            id: 0,
            nodes: Default::default(),
            color: None,
        }))
    }

    fn get_children_mask(&self) -> u16 {
        if self.color.is_some() {
            return 0;
        }

        let mut child_mask = 0;
        let mut leaf_mask = 0;

        for (i, child) in self.nodes.iter().enumerate() {
            if child.is_none() {
                continue;
            }
            let child = child.as_ref().unwrap();
            child_mask |= 1 << i;
            if child.borrow().color.is_some() {
                leaf_mask |= 1 << i;
            }
        }

        (child_mask << 8) | leaf_mask
    }

    fn build_octant_blocks(&self) -> Box<OctantBlock> {
        let mut block = Box::new(OctantBlock {
            id: self.id,
            encoded: Default::default(),
            children: Default::default(),
            unpopulated_pointers: Vec::new(),
        });

        let mut children_masks = Vec::new();

        for (i, child) in self.nodes.iter().enumerate() {
            if child.is_none() {
                children_masks.push(0);
                continue;
            }

            let child = child.as_ref().unwrap().borrow();
            if let Some(color) = child.color {
                children_masks.push(0);
                block.encoded[4 + i] = color as u32;
                continue;
            }

            children_masks.push(child.get_children_mask());
            block.children[i] = Some(child.build_octant_blocks());
            block.unpopulated_pointers.push(UnpopulatedPointer {
                octant_index: i as i8,
                octant_block_ref: child.id,
            });
        }

        // encode children masks: combine to u16 masks of two children into one u32 int
        for i in 0..8 {
            let mut mask = children_masks[i] as u32;
            if (i % 2) != 0 {
                mask <<= 16;
            }
            block.encoded[i / 2] |= mask;
        }

        block
    }
}

pub struct SVO {
    pub max_depth: i32,
    pub max_depth_exp2: f32,
    pub descriptors: Vec<u32>,
}

pub fn build_voxel_model(filename: &str) -> SVO {
    println!("loading model");
    let data = dot_vox::load(filename).unwrap();
    let model = &data.models[0];

    println!("collecting leaves");
    let mut leaves = HashMap::new();
    for voxel in &model.voxels {
        let pos = NodePos { x: voxel.x, y: voxel.z, z: voxel.y };
        let leaf = TreeNode::new();
        leaf.borrow_mut().color = Some(data.palette[voxel.i as usize] as i32);
        leaves.insert(pos, leaf);
    }

    let mut slot_counts = model.voxels.len() + leaves.len();

    println!("assembling tree");
    let mut input = leaves;
    while input.len() > 1 {
        let output = merge_tree_nodes(input);
        slot_counts += output.len();
        input = output;
    }

    println!("total of {} bytes = {} MB", slot_counts * 4, slot_counts as f32 * 4.0 / 1000.0 / 1000.0);

    println!("building SVO");
    let key = *input.keys().next().unwrap();
    let descriptors = build_svo_buffer(input.remove(&key).unwrap());
    println!("entries in SVO: {}", descriptors.len());
    println!("SVO size: {} MB", descriptors.len() as f32 * 4.0 / 1000.0 / 1000.0);

    let max_depth = (model.size.x.max(model.size.y.max(model.size.z)) as f32).log2().ceil() as i32;
    SVO {
        max_depth,
        max_depth_exp2: (-max_depth as f32).exp2(),
        descriptors,
    }
}

fn merge_tree_nodes(input: HashMap<NodePos, Rc<RefCell<TreeNode>>>) -> HashMap<NodePos, Rc<RefCell<TreeNode>>> {
    let mut output = HashMap::new();
    for (pos, node) in input.into_iter() {
        let parent_pos = NodePos { x: pos.x / 2, y: pos.y / 2, z: pos.z / 2 };
        let parent_node = output.entry(parent_pos).or_insert_with(|| TreeNode::new());
        let idx = (pos.x % 2 + (pos.y % 2) * 2 + (pos.z % 2) * 4) as usize;
        parent_node.borrow_mut().nodes[idx] = Some(node);
    }
    output
}

fn build_svo_buffer(root_node: Rc<RefCell<TreeNode>>) -> Vec<u32> {
    // TODO remove RefCell somehow?

    // assign unique ids to every TreeNode & create a fast lookup map
    let mut next_id = 1;
    let mut node_map = HashMap::new();

    let mut nodes_flat = VecDeque::new();
    nodes_flat.push_back(Rc::clone(&root_node)); // TODO

    while !nodes_flat.is_empty() {
        let node = nodes_flat.pop_front().unwrap();
        node.borrow_mut().id = next_id;
        node_map.insert(next_id, Rc::clone(&node));
        next_id += 1;

        for child in node.borrow().nodes.iter() {
            if let Some(child) = child {
                nodes_flat.push_back(Rc::clone(child));
            }
        }
    }

    // build octant block tree
    let root_block = root_node.borrow().build_octant_blocks();

    // build result buffer
    let estimated_size = next_id * 12 + 5; // each block takes up 12 ints
    let mut svo = Vec::with_capacity(estimated_size as usize);
    let mut block_ptr_map = HashMap::new();

    // fake root block
    svo.push(root_node.borrow().get_children_mask() as u32);
    svo.push(0);
    svo.push(0);
    svo.push(0);
    svo.push(svo.len() as u32 + 1);

    // push all child blocks
    let mut blocks_flat = VecDeque::new();
    blocks_flat.push_back(root_block);

    let mut unpopulated_pointers = Vec::new();

    while !blocks_flat.is_empty() {
        let block = blocks_flat.pop_front().unwrap();

        let current_pointer = svo.len() as u32;

        block_ptr_map.insert(block.id, current_pointer);
        svo.extend_from_slice(&block.encoded);

        for ptr in block.unpopulated_pointers {
            let ref_ptr = current_pointer + 4 + ptr.octant_index as u32;
            unpopulated_pointers.push((ref_ptr, ptr.octant_block_ref));
        }

        for child in block.children {
            if let Some(child) = child {
                blocks_flat.push_back(child);
            }
        }
    }

    for (ptr, block_ref) in unpopulated_pointers {
        svo[ptr as usize] = block_ptr_map[&block_ref];
    }

    svo
}
