// TODO find a way to not copy new sub SVOs into the buffer but allocate them in place directly (memory pooling)

use std::cell::{Ref, RefCell};
use std::cmp::max;
use std::collections::HashMap;
use std::rc::Rc;

use cgmath::num_traits::Pow;

use crate::storage::svo::Svo;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct GridPos {
    x: i32,
    y: i32,
    z: i32,
}

type NodePtr = Rc<RefCell<Node>>;

struct Node {
    parent: Option<NodePtr>,
    children: [Option<NodePtr>; 8],
    grid_pos: Option<GridPos>,
}

struct GridSlot {
    node: NodePtr,
    svo: Svo,
}

pub struct SvoGrid {
    root_node: Option<NodePtr>,
    grid: HashMap<GridPos, GridSlot>,
    depth: i32,
}

impl SvoGrid {
    pub fn new() -> SvoGrid {
        SvoGrid {
            root_node: None,
            grid: HashMap::new(),
            depth: 0,
        }
    }

    pub fn add_svo(&mut self, x: i32, y: i32, z: i32, svo: Svo) {
        // replace svo if the slot already exists
        let pos = GridPos { x, y, z };
        if let Some(value) = self.grid.get_mut(&pos) {
            value.svo = svo;
            return;
        }

        // create a new leaf node and occupy the new slot
        let new_node = Rc::new(RefCell::new(Node {
            parent: None,
            children: Default::default(),
            grid_pos: Some(pos),
        }));
        self.grid.insert(pos, GridSlot { node: Rc::clone(&new_node), svo });

        // rebuild tree structure
        let required_depth = max(1, max(x, max(y, z)));
        let required_depth = (required_depth as f32).log2().floor() as i32 + 1;

        // expand octree to fit all potential new sub octrees
        if required_depth > self.depth {
            for _ in self.depth..required_depth {
                let new_root = Rc::new(RefCell::new(Node {
                    parent: None,
                    children: Default::default(),
                    grid_pos: None,
                }));
                if self.root_node.is_some() {
                    let old_root = self.root_node.take().unwrap();
                    old_root.borrow_mut().parent = Some(Rc::clone(&new_root));
                    new_root.borrow_mut().children[0] = Some(old_root);
                }
                self.root_node = Some(new_root);
            }

            self.depth = required_depth;
        }

        // decent until leaf level to insert new node and create necessary trees along the way
        let mut pos = (x, y, z);
        let mut size = 2f32.pow(self.depth) as i32;
        let mut node = self.root_node.clone().unwrap();
        while size > 0 {
            size /= 2;

            let idx_x = pos.0 / size;
            let idx_y = pos.1 / size;
            let idx_z = pos.2 / size;
            let idx = (idx_x + idx_y * 2 + idx_z * 4) as usize;

            if size == 1 {
                new_node.borrow_mut().parent = Some(Rc::clone(&node));
                node.borrow_mut().children[idx] = Some(new_node);
                break;
            }

            if node.borrow().children[idx].is_none() {
                let child = Rc::new(RefCell::new(Node {
                    parent: Some(Rc::clone(&node)),
                    children: Default::default(),
                    grid_pos: None,
                }));
                node.borrow_mut().children[idx] = Some(child);
            }

            pos.0 %= size;
            pos.1 %= size;
            pos.2 %= size;

            let next = node.borrow().children[idx].clone().unwrap();
            node = next;
        }
    }

    // TODO support svo removal?

    pub fn build_svo(&self) -> Svo {
        if self.root_node.is_none() {
            return Svo { header_mask: 0, depth: 0, descriptors: vec![] };
        }

        let root_node = self.root_node.as_ref().unwrap();

        // TODO figure out correct size
        let mut descriptors = Vec::new();

        // create fake root node
        let mut root_mask = 0;
        for (idx, node) in root_node.borrow().children.iter().enumerate() {
            if node.is_none() {
                continue;
            }
            root_mask |= (1 << idx) << 8;
        }
        descriptors.push(root_mask as u32);
        descriptors.push(0);
        descriptors.push(0);
        descriptors.push(0);
        descriptors.push(5); // absolute pointer to where the actual octree starts

        let (ctrl, missing_pointers) = self.build_descriptors(self.root_node.as_ref().unwrap().borrow());
        descriptors.extend(ctrl);

        for (pos, node) in self.grid.iter() {
            let ptr = descriptors.len();
            descriptors.extend(&node.svo.descriptors);

            let idx = missing_pointers[pos] + 5;
            descriptors[idx] = ptr as u32;
        }

        // TODO better way to calculate depth
        Svo { header_mask: root_mask, depth: self.depth + 5, descriptors }
    }

    fn build_descriptors(&self, node: Ref<Node>) -> (Vec<u32>, HashMap<GridPos, usize>) {
        let mut svo = Vec::<u32>::with_capacity(12);
        svo.extend(std::iter::repeat(0).take(12));

        let mut missing_pointers = HashMap::new();

        for (idx, node) in node.children.iter().enumerate() {
            if node.is_none() {
                continue;
            }

            let node = node.as_ref().unwrap().borrow();

            if let Some(pos) = node.grid_pos {
                let slot = self.grid.get(&pos).unwrap();

                let mut mask = slot.svo.header_mask as u32;
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                svo[(idx / 2) as usize] |= mask;

                svo[4 + idx] = 0; // will be populated later
                missing_pointers.insert(pos, 4 + idx);
            } else {
                let mut mask = 0;
                for (idx, child) in node.children.iter().enumerate() {
                    if child.is_none() {
                        continue;
                    }
                    mask |= (1 << idx) << 8;
                }
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                svo[(idx / 2) as usize] |= mask;

                let (child_svo, child_pointers) = self.build_descriptors(node);
                svo[4 + idx] = svo.len() as u32 - 4 - idx as u32;
                svo[4 + idx] |= 1 << 31; // flag as relative pointer

                let base_ptr = svo.len();
                svo.extend(child_svo);

                for (pos, ptr) in child_pointers {
                    missing_pointers.insert(pos, base_ptr + ptr);
                }
            }
        }

        (svo, missing_pointers)
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::svo::Svo;

    #[test]
    fn svo_grid_add_and_build() {
        let mut grid = super::SvoGrid::new();
        grid.add_svo(0, 0, 0, Svo {
            header_mask: 1,
            depth: 0,
            descriptors: vec![100],
        });
        grid.add_svo(3, 3, 3, Svo {
            header_mask: 2,
            depth: 0,
            descriptors: vec![200],
        });

        let svo = grid.build_svo();
        assert_eq!(svo.descriptors, vec![
            // fake root header
            (1 | (1 << 7)) << 8,
            0,
            0,
            0,
            // fake root body
            5,

            // core octant header
            1 << 8,
            0,
            0,
            (1 << 7) << 8 << 16,
            // core octant body
            (1 << 31) | 8, 0, 0, 0,
            0, 0, 0, (1 << 31) | (12 + 1),

            // header for (0,0,0)
            1, // use svo header here
            0,
            0,
            0,
            // body for (0,0,0)
            3 * 12 + 5, 0, 0, 0, // use svo pointer here
            0, 0, 0, 0,

            // header for (3,3,3)
            0,
            0,
            0,
            2 << 16, // use svo header here
            // body for (3,3,3)
            0, 0, 0, 0,
            0, 0, 0, 3 * 12 + 5 + 1, // use svo pointer here

            // leaf data
            100, // svo 1
            200, // svo 2
        ]);
    }
}
