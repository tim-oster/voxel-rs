use std::cmp::max;

use cgmath::num_traits::Pow;

type OctantId = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Position(u32, u32, u32);

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
pub struct Octree<T> {
    octants: Vec<Octant<T>>,
    free_list: Vec<OctantId>,
    root: Option<OctantId>,
    depth: u32,
}

impl<T> Octree<T> {
    pub fn new() -> Octree<T> {
        Octree { octants: Vec::new(), free_list: Vec::new(), root: None, depth: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Octree<T> {
        Octree { octants: Vec::with_capacity(capacity), free_list: Vec::new(), root: None, depth: 0 }
    }

    /// Adds the given leaf value at the given position. If the tree is not big enough yet,
    /// it expands it until it can successfully insert. Children along the path are overridden,
    /// if any exist.
    pub fn add_leaf(&mut self, pos: Position, leaf: T) {
        let current_depth = self.depth;
        let required_depth = pos.required_depth();
        let difference = required_depth as i32 - current_depth as i32;
        if difference > 0 {
            self.expand(difference as u32);
        }

        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

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
                    self.octants[child].content = None;
                }

                // if this is the end of the tree, insert the content
                if size == 1 {
                    self.octants[child].content = Some(leaf);
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
            }
        }
    }

    /// Removes the leaf at the given position if it exists. Empty children are *not* removed from
    /// the tree. Look at [`Octree::compact`] for removing empty children.
    pub fn remove_leaf(&mut self, pos: Position) {
        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

        while size > 0 {
            size /= 2;

            let idx = Position(pos.0 / size, pos.1 / size, pos.2 / size).idx();

            pos.0 %= size;
            pos.1 %= size;
            pos.2 %= size;

            let child = self.octants[it].children[idx];
            if child.is_none() {
                break;
            }

            let parent = it;
            it = child.unwrap();

            if size == 1 {
                self.octants[parent].remove_child(idx);
                self.delete_octant(it);
                break;
            }
        }
    }

    fn expand(&mut self, by: u32) {
        for _ in 0..by {
            let new_root_id = self.new_octant(None);

            if let Some(root_id) = self.root {
                self.octants[root_id].parent = Some(new_root_id);
                self.octants[new_root_id].add_child(0, root_id);
            }

            self.root = Some(new_root_id);
        }

        self.depth += by
    }

    fn new_octant(&mut self, parent: Option<OctantId>) -> OctantId {
        let octant = Octant {
            parent,
            children: Default::default(),
            children_count: 0,
            content: None,
        };

        if let Some(free_id) = self.free_list.pop() {
            self.octants[free_id] = octant;
            return free_id;
        }

        let id = self.octants.len() as OctantId;
        self.octants.push(octant);
        id
    }

    fn delete_octant(&mut self, id: OctantId) {
        self.free_list.push(id);
    }
}

#[derive(Debug, PartialEq)]
struct Octant<T> {
    parent: Option<OctantId>,

    children: [Option<OctantId>; 8],
    children_count: u8,

    content: Option<T>,
}

impl<T> Octant<T> {
    fn add_child(&mut self, idx: usize, child: OctantId) {
        self.children[idx] = Some(child);
        self.children_count += 1;
    }

    fn remove_child(&mut self, idx: usize) {
        self.children[idx] = None;
        self.children_count -= 1;
    }
}

//noinspection DuplicatedCode
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
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: None,
                },
                Octant {
                    parent: None,
                    children: [Some(0), None, None, None, Some(2), None, None, None],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, None, Some(3)],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(2),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
            ],
            free_list: vec![],
            root: Some(1),
            depth: 2,
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
                    children: [Some(6), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(2),
                    children: [Some(0), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant { // root node
                    parent: None,
                    children: [Some(1), None, None, None, Some(7), None, None, Some(3)],
                    children_count: 3,
                    content: None,
                },
                // first node start
                Octant {
                    parent: Some(2),
                    children: [None, None, None, Some(4), None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(3),
                    children: [None, None, None, None, None, None, Some(5), None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(4),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
                // first node end
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
                // third node start
                Octant {
                    parent: Some(2),
                    children: [None, None, None, None, Some(8), None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(7),
                    children: [None, Some(9), None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(8),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(30),
                },
                // third node end
            ],
            free_list: vec![],
            root: Some(2),
            depth: 3,
        });
    }

    #[test]
    fn octree_leaf_replacement() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [Some(1), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        };

        octree.add_leaf(Position(0, 0, 0), 20);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [Some(1), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });
    }

    #[test]
    fn octree_remove_and_add_leaf() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [Some(1), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        };

        octree.remove_leaf(Position(0, 0, 0));

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
            ],
            free_list: vec![1],
            root: Some(0),
            depth: 1,
        });

        octree.add_leaf(Position(0, 0, 0), 20);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [Some(1), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });
    }
}
