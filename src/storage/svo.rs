use std::collections::HashMap;

#[derive(PartialEq, Eq, Hash)]
struct NodePos {
    x: u8,
    y: u8,
    z: u8,
}

struct TreeNode {
    nodes: [Option<Box<TreeNode>>; 8],
    color: Option<i32>,
}

impl TreeNode {
    fn new() -> Box<TreeNode> {
        Box::new(TreeNode {
            nodes: Default::default(),
            color: None,
        })
    }

    fn build_descriptor(&self) -> i32 {
        if let Some(color) = self.color {
            return color;
        }

        let mut child_mask = 0;
        let mut leaf_mask = 0;

        for (i, child) in self.nodes.iter().enumerate() {
            if child.is_none() {
                continue;
            }
            let child = child.as_ref().unwrap();
            child_mask |= 1 << i;
            if child.color.is_some() {
                leaf_mask |= 1 << i;
            }
        }

        (child_mask << 8) | leaf_mask
    }

    fn build(&self) -> Vec<i32> {
        self.build_with_far_pointer_threshold(0x7fff)
    }

    fn build_with_far_pointer_threshold(&self, far_pointer_threshold: usize) -> Vec<i32> {
        let mut result = Vec::new();
        result.push(self.build_descriptor());
        if ((result[0] >> 8) & 0xff) == 0 {
            return result;
        }

        result[0] |= 1 << 17;

        struct ChildDesc {
            index: usize,
            desc: Vec<i32>,
            far_ptr_index: Option<usize>,
        }
        let mut full_child_descriptors = Vec::new();

        let mut last_child_index = None;
        for child in self.nodes.iter() {
            if child.is_none() {
                // skip leading empty children
                if result.len() > 1 {
                    result.push(0);
                }
                continue;
            }
            let child = child.as_ref().unwrap();
            result.push(child.build_descriptor());
            last_child_index = Some(result.len() - 1);
            if child.color.is_some() {
                continue;
            }
            full_child_descriptors.push(ChildDesc {
                index: result.len() - 1,
                desc: child.build_with_far_pointer_threshold(far_pointer_threshold),
                far_ptr_index: None,
            });
        }
        // strip trailing empty children
        if let Some(index) = last_child_index {
            result.truncate(index + 1);
        }

        full_child_descriptors.sort_by(|a, b| {
            let len_a = a.desc.len();
            let len_b = b.desc.len();
            if len_a == len_b {
                return a.index.partial_cmp(&b.index).unwrap();
            }
            len_a.partial_cmp(&len_b).unwrap()
        });

        let mut space_count = 0;
        let start_index = result.len();
        for child in full_child_descriptors.iter_mut() {
            let offset = (start_index + space_count) - child.index;
            if offset > far_pointer_threshold {
                result.push(0);
                child.far_ptr_index = Some(result.len() - 1);
                space_count += 1;
            }

            space_count += child.desc.len() - 1; // remove descriptor i32 from space
        }

        for child in full_child_descriptors.iter() {
            match child.far_ptr_index {
                None => {
                    let offset = result.len() - child.index;
                    result[child.index] |= (offset as i32) << 17;
                }
                Some(ptr_index) => {
                    let offset = ptr_index - child.index;
                    result[child.index] |= 1 << 16;
                    result[child.index] |= (offset as i32) << 17;

                    let offset = result.len() - child.index;
                    result[ptr_index] = offset as i32;
                }
            }
            result.extend_from_slice(&child.desc[1..]);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::svo::TreeNode;

    #[test]
    fn build_leaf_node() {
        let mut node = TreeNode::new();
        node.color = Some(0xff);

        let desc = node.build();
        assert_eq!(desc, vec![255]);
    }

    #[test]
    fn build_parent_with_only_leaves() {
        let mut parent = TreeNode::new();
        parent.nodes[1] = Some(TreeNode::new());
        parent.nodes[1].as_mut().unwrap().color = Some(0xa);
        parent.nodes[4] = Some(TreeNode::new());
        parent.nodes[4].as_mut().unwrap().color = Some(0xb);

        let desc = parent.build();
        assert_eq!(desc, vec![
            0b00000000_00000010_00010010_00010010,
            0xa,
            0,
            0,
            0xb,
        ]);
    }

    #[test]
    fn build_parent_with_children_with_leaves() {
        let mut child = TreeNode::new();
        child.nodes[3] = Some(TreeNode::new());
        child.nodes[3].as_mut().unwrap().color = Some(0xa);

        let mut root = TreeNode::new();
        root.nodes[1] = Some(child);
        root.nodes[4] = Some(TreeNode::new());
        root.nodes[4].as_mut().unwrap().color = Some(0xb);

        let desc = root.build();
        assert_eq!(desc, vec![
            0b00000000_00000010_00010010_00010000,
            0b00000000_00001000_00001000_00001000,
            0,
            0,
            0xb,
            //
            0xa,
        ]);
    }

    #[test]
    fn build_parent_with_far_pointers() {
        let mut child_a_a = TreeNode::new();
        child_a_a.nodes[3] = Some(TreeNode::new());
        child_a_a.nodes[3].as_mut().unwrap().color = Some(0xa);

        let mut child_a = TreeNode::new();
        child_a.nodes[3] = Some(child_a_a);

        let mut child_b = TreeNode::new();
        child_b.nodes[3] = Some(TreeNode::new());
        child_b.nodes[3].as_mut().unwrap().color = Some(0xb);

        let mut root = TreeNode::new();
        root.nodes[1] = Some(child_a);
        root.nodes[3] = Some(child_b);

        let desc = root.build_with_far_pointer_threshold(1);
        assert_eq!(desc, vec![
            0b00000000_00000010_00001010_00000000,
            0b00000000_00000111_00001000_00000000,
            0,
            0b00000000_00000100_00001000_00001000,
            //
            5,
            //
            0xb,
            //
            0b00000000_00000010_00001000_00001000,
            //
            0xa,
        ]);
    }
}

pub struct SVO {
    pub max_depth: i32,
    pub max_depth_exp2: f32,
    pub descriptors: Vec<i32>,
}

pub fn build_voxel_model(filename: &str) -> SVO {
    println!("loading model");
    let data = dot_vox::load(filename).unwrap();
    let model = &data.models[0];

    println!("collecting leaves");
    let mut leaves = HashMap::new();
    for voxel in &model.voxels {
        let pos = NodePos { x: voxel.x, y: voxel.z, z: voxel.y };
        let mut leaf = TreeNode::new();
        leaf.color = Some(data.palette[voxel.i as usize] as i32);
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
    let descriptors = input[input.keys().next().unwrap()].build();
    println!("entries in SVO: {}", descriptors.len());
    println!("SVO size: {} MB", descriptors.len() as f32 * 4.0 / 1000.0 / 1000.0);

    let max_depth = (model.size.x.max(model.size.y.max(model.size.z)) as f32).log2().ceil() as i32;
    SVO {
        max_depth,
        max_depth_exp2: (-max_depth as f32).exp2(),
        descriptors,
    }
}

fn merge_tree_nodes(input: HashMap<NodePos, Box<TreeNode>>) -> HashMap<NodePos, Box<TreeNode>> {
    let mut output = HashMap::new();
    for (pos, node) in input.into_iter() {
        let parent_pos = NodePos { x: pos.x / 2, y: pos.y / 2, z: pos.z / 2 };
        let parent_node = output.entry(parent_pos).or_insert_with(|| TreeNode::new());
        let idx = (pos.x % 2 + (pos.y % 2) * 2 + (pos.z % 2) * 4) as usize;
        parent_node.nodes[idx] = Some(node);
    }
    output
}
