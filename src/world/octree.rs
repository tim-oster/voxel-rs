use std::cmp::max;

use cgmath::num_traits::Pow;

pub type OctantId = usize;

// TODO probably requires refactoring in the future

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Position(pub u32, pub u32, pub u32);

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
    pub(in crate::world) octants: Vec<Octant<T>>,
    pub(in crate::world) free_list: Vec<OctantId>,
    pub(in crate::world) root: Option<OctantId>,
    pub(in crate::world) depth: u32,
}

impl<T> Octree<T> {
    pub fn new() -> Octree<T> {
        Octree { octants: Vec::new(), free_list: Vec::new(), root: None, depth: 0 }
    }

    pub fn with_size(size: u32) -> Octree<T> {
        let mut octree = Self::new();
        octree.expand_to(size);
        octree
    }

    /// Adds the given leaf value at the given position. If the tree is not big enough yet,
    /// it expands it until it can successfully insert. Children along the path are overridden,
    /// if any exist.
    pub fn add_leaf(&mut self, pos: Position, leaf: T) -> OctantId {
        self.expand_to(pos.required_depth());

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
                    return child;
                }
            } else {
                let prev_id = it;
                let next_id = self.new_octant(Some(prev_id));
                it = next_id;

                let prev_octant = &mut self.octants[prev_id];
                prev_octant.add_child(idx, next_id);

                if size == 1 {
                    let current = &mut self.octants[next_id];
                    current.content = Some(leaf);
                    return next_id;
                }
            }
        }

        panic!("could not reach end of tree");
    }

    pub fn replace_leaf(&mut self, pos: Position, leaf: OctantId) -> Option<OctantId> {
        self.expand_to(pos.required_depth());

        if let Some(old_parent_id) = self.octants[leaf].parent {
            self.octants[old_parent_id].remove_child_by_octant_id(leaf);
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

            if size == 1 {
                let previous_child = self.octants[it].children[idx];
                if previous_child == Some(leaf) {
                    return None;
                }

                if let Some(previous_child) = previous_child {
                    self.octants[it].remove_child(idx);
                    self.octants[previous_child].parent = None;
                }

                self.octants[it].add_child(idx, leaf);
                self.octants[leaf].parent = Some(it);

                return previous_child;
            }

            if let Some(child) = self.octants[it].children[idx] {
                it = child;

                // if the child is a leaf node, convert it to a normal child node and remove its content
                let current = &self.octants[child];
                if current.content.is_some() {
                    self.octants[child].content = None;
                }
            } else {
                let prev_id = it;
                let next_id = self.new_octant(Some(prev_id));
                it = next_id;

                let prev_octant = &mut self.octants[prev_id];
                prev_octant.add_child(idx, next_id);
            }
        }

        panic!("could not reach end of tree");
    }

    /// Removes the leaf at the given position if it exists. Empty parents are *not* removed from
    /// the tree. Look at [`Octree::compact`] for removing empty parents.
    pub fn remove_leaf(&mut self, pos: Position) -> Option<OctantId> {
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

            it = child.unwrap();

            if size == 1 {
                self.delete_octant(it);
                break;
            }
        }

        None
    }

    pub fn get_leaf(&self, pos: Position) -> Option<&T> {
        self.find_leaf(pos).map_or(None, |v| self.octants[v].content.as_ref())
    }

    pub fn find_leaf(&self, pos: Position) -> Option<OctantId> {
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

            it = child.unwrap();

            if size == 1 {
                return Some(it);
            }
        }

        None
    }

    pub fn expand(&mut self, by: u32) {
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

    pub fn expand_to(&mut self, to: u32) {
        if self.depth > to {
            return;
        }
        let diff = to - self.depth;
        if diff > 0 {
            self.expand(diff);
        }
    }

    /// Removes all octants from the tree that have no children and no content. The algorithm
    /// is depth first, so removal cascades through the whole tree removing all empty parents
    /// as well.
    pub fn compact(&mut self) {
        if self.root.is_none() {
            return;
        }
        // TODO can it happen that the root is removed?
        self.compact_octant(self.root.unwrap());
    }

    fn compact_octant(&mut self, octant_id: OctantId) {
        let octant = &self.octants[octant_id];
        let mut children = Vec::new();
        for (i, child) in octant.children.iter().enumerate() {
            if let Some(id) = child {
                children.push((i, *id));
            }
        }

        for child in children {
            self.compact_octant(child.1);

            let octant = &self.octants[child.1];
            if octant.children_count == 0 && octant.content.is_none() {
                self.delete_octant(child.1);
                self.octants[octant_id].remove_child(child.0);
            }
        }
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

    pub fn delete_octant(&mut self, id: OctantId) {
        let octant = &self.octants[id];
        if let Some(parent) = octant.parent {
            self.octants[parent].remove_child_by_octant_id(id);
            self.octants[id].parent = None;
        }
        self.free_list.push(id);
    }
}

#[derive(Debug, PartialEq)]
pub(in crate::world) struct Octant<T> {
    pub(in crate::world) parent: Option<OctantId>,

    pub(in crate::world) children: [Option<OctantId>; 8],
    pub(in crate::world) children_count: u8,

    pub(in crate::world) content: Option<T>,
}

impl<T> Octant<T> {
    fn add_child(&mut self, idx: usize, child: OctantId) {
        if self.children[idx].is_none() {
            self.children_count += 1;
        }
        self.children[idx] = Some(child);
    }

    fn remove_child(&mut self, idx: usize) {
        if self.children[idx].is_some() {
            self.children_count -= 1;
        }
        self.children[idx] = None;
    }

    fn remove_child_by_octant_id(&mut self, child: OctantId) {
        if let Some(idx) = self.children.iter().position(|&x| x == Some(child)) {
            self.remove_child(idx);
        }
    }
}

//noinspection DuplicatedCode
#[cfg(test)]
mod tests {
    use crate::world::octree::{Octant, Octree, Position};

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

        assert_eq!(octree.get_leaf(Position(1, 1, 3)), Some(&20));
        assert_eq!(octree.get_leaf(Position(1, 1, 1)), None);
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
    fn octree_add_leaf_replacing() {
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
                    parent: None,
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

    #[test]
    fn octree_replace_leaf() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [Some(1), None, None, None, None, None, None, Some(2)],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
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
        };

        // replace at empty slot
        let previous = octree.replace_leaf(Position(1, 0, 0), 1);
        assert_eq!(previous, None);
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [None, Some(1), None, None, None, None, None, Some(2)],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
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

        // replace with itself
        let previous = octree.replace_leaf(Position(1, 0, 0), 1);
        assert_eq!(previous, None);
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [None, Some(1), None, None, None, None, None, Some(2)],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
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

        // replace with existing
        let previous = octree.replace_leaf(Position(1, 1, 1), 1);
        assert_eq!(previous, Some(2));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [None, None, None, None, None, None, None, Some(1)],
                    children_count: 1,
                    content: None,
                },
                Octant {
                    parent: Some(0),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
                Octant {
                    parent: None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });

        // replace in new parent
        let previous = octree.replace_leaf(Position(2, 0, 0), 1);
        assert_eq!(previous, None);
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(3),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: None,
                },
                Octant {
                    parent: Some(4),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(10),
                },
                Octant {
                    parent: None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: Some(20),
                },
                Octant {
                    parent: None,
                    children: [Some(0), Some(4), None, None, None, None, None, None],
                    children_count: 2,
                    content: None,
                },
                Octant {
                    parent: Some(3),
                    children: [Some(1), None, None, None, None, None, None, None],
                    children_count: 1,
                    content: None,
                },
            ],
            free_list: vec![],
            root: Some(3),
            depth: 2,
        });
    }

    #[test]
    fn octree_compact() {
        let mut octree = Octree::new();

        octree.add_leaf(Position(1, 1, 3), 20);
        octree.compact();

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                    content: None,
                },
                Octant {
                    parent: None,
                    children: [None, None, None, None, Some(2), None, None, None],
                    children_count: 1,
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
            free_list: vec![0],
            root: Some(1),
            depth: 2,
        });
    }
}
