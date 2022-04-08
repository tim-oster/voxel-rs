use std::borrow::{Borrow, BorrowMut};
use std::cmp::max;

use cgmath::num_traits::Pow;

type OctantId = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Position(u32, u32, u32);

impl Position {
    #[inline]
    fn idx(&self) -> usize {
        (self.0 + self.1 * 2 + self.2 * 4) as usize
    }

    fn required_depth(&self) -> u32 {
        let depth = max(1, max(self.0, max(self.1, self.2)));
        (depth as f32).log2().floor() as u32 + 1
    }
}

#[derive(Debug, PartialEq)]
struct Octree<T> {
    octants: Vec<Octant<T>>,
    root: Option<OctantId>,
}

impl<T> Octree<T> {
    fn new() -> Octree<T> {
        Octree { octants: Vec::new(), root: None }
    }

    fn with_capacity(capacity: usize) -> Octree<T> {
        Octree { octants: Vec::with_capacity(capacity), root: None }
    }

    fn depth(&self) -> u32 {
        if self.root.is_none() {
            return 0;
        }
        self.octants[self.root.unwrap()].size_power
    }

    fn add_leaf(&mut self, pos: Position, leaf: T) {
        let current_depth = self.depth();
        let required_depth = pos.required_depth();
        let difference = required_depth as i32 - current_depth as i32;
        if difference > 0 { self.expand(difference as u32); }

        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = self.octants[it].size();

        while size > 0 {
            size /= 2;

            let idx = Position(pos.0 / size, pos.1 / size, pos.2 / size).idx();

            pos.0 %= size;
            pos.1 %= size;
            pos.2 %= size;

            if let Some(child) = self.octants[it].children[idx] {
                it = child;

                // if the child is a leaf node, convert it to a normal child node and remove its content
                let current = &self.octants[child];
                if current.content.is_some() {
                    let mut parent_power = 0;

                    if let Some(parent) = current.parent {
                        parent_power = self.octants[parent].size_power;
                    }

                    let current = &mut self.octants[child];
                    current.size_power = parent_power + 1;
                    current.content = None;
                }

                // if this is the end of the tree, insert the content
                if size == 1 {
                    let current = &mut self.octants[child];
                    current.size_power = 0;
                    current.content = Some(leaf);
                    break;
                }
            } else {
                let prev_id = it;
                let next_id = self.new_octant(Some(prev_id));
                it = next_id;

                let mut prev_octant = &mut self.octants[prev_id];
                prev_octant.add_child(idx, next_id);

                if size == 1 {
                    let current = &mut self.octants[next_id];
                    current.content = Some(leaf);
                    break;
                }

                let mut size_counter = 1;
                let mut it = Some(next_id);
                while it.is_some() {
                    let current = &mut self.octants[it.unwrap()];
                    if current.size_power < size_counter {
                        current.size_power = size_counter;
                    }
                    size_counter += 1;
                    it = current.parent;
                }
            }
        }
    }

    fn remove_leaf(&mut self, pos: Position) {
        // TODO
    }

    fn expand(&mut self, by: u32) {
        for _ in 0..by {
            let new_root_id = self.new_octant(None);

            if let Some(root_id) = self.root {
                let mut root = &mut self.octants[root_id];
                let size_power = root.size_power;
                root.parent = Some(new_root_id);

                let mut new_root = &mut self.octants[new_root_id];
                new_root.add_child(0, root_id);
                new_root.size_power = size_power + 1;
            } else {
                self.octants[new_root_id].size_power = 1;
            }

            self.root = Some(new_root_id);
        }
    }

    fn new_octant(&mut self, parent: Option<OctantId>) -> OctantId {
        let octant = Octant {
            parent,
            size_power: 0,
            children: Default::default(),
            children_count: 0,
            content: None,
        };

        let id = self.octants.len() as OctantId;
        self.octants.push(octant);
        id
    }

    // TODO how efficiently remove empty octants? .compact() method?
}

#[derive(Debug, PartialEq)]
struct Octant<T> {
    parent: Option<OctantId>,
    size_power: u32,

    children: [Option<OctantId>; 8],
    children_count: u8,

    content: Option<T>,
}

impl<T> Octant<T> {
    fn size(&self) -> u32 {
        2f32.pow(self.size_power as i32) as u32
    }

    fn add_child(&mut self, idx: usize, child: OctantId) {
        self.children[idx] = Some(child);
        self.children_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::octree::{Octant, Octree, Position};

    #[test]
    fn octree_add_leaf_single() {
        let mut octree = Octree::new();

        octree.add_leaf(Position(1, 1, 3), 20);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    size_power: 1,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: None,
                },
                Octant {
                    parent: None,
                    size_power: 2,
                    children: [Some(0), None, None, None, Some(2), None, None, None],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(1),
                    size_power: 1,
                    children: [None, None, None, None, None, None, None, Some(3)],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(2),
                    size_power: 0,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
            ],
            root: Some(1),
        });
    }

    #[test]
    fn octree_add_leaf_multiple() {
        let mut octree = Octree::new();

        octree.add_leaf(Position(6, 7, 5), 10);
        octree.add_leaf(Position(0, 0, 0), 20);
        octree.add_leaf(Position(1, 0, 6), 30);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    size_power: 1,
                    children: [Some(6), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(2),
                    size_power: 2,
                    children: [Some(0), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant { // root node
                    parent: None,
                    size_power: 3,
                    children: [Some(1), None, None, None, Some(7), None, None, Some(3)],
                    children_count: 3,
                    content: None,
                },
                // first node start
                Octant {
                    parent: Some(2),
                    size_power: 2,
                    children: [None, None, None, Some(4), None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(3),
                    size_power: 1,
                    children: [None, None, None, None, None, None, Some(5), None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(4),
                    size_power: 0,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
                // first node end
                Octant {
                    parent: Some(0),
                    size_power: 0,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
                // third node start
                Octant {
                    parent: Some(2),
                    size_power: 2,
                    children: [None, None, None, None, Some(8), None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(7),
                    size_power: 1,
                    children: [None, Some(9), None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(8),
                    size_power: 0,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(30),
                },
                // third node end
            ],
            root: Some(2),
        });
    }

    // TODO test edge cases: leaf override & replacement
}
