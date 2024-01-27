use std::cmp::max;
use std::mem;

use cgmath::num_traits::Pow;

pub type OctantId = u32;

/// LeafId describes a leaf's position inside the octree by storing the child's `idx` inside its `parent` octant.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct LeafId {
    pub parent: OctantId,
    pub idx: u8,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Position(pub u32, pub u32, pub u32);

impl Position {
    fn idx(&self) -> u8 {
        (self.0 + self.1 * 2 + self.2 * 4) as u8
    }

    fn required_depth(&self) -> u8 {
        let depth = max(1, max(self.0, max(self.1, self.2)));
        (depth as f32).log2().floor() as u8 + 1
    }
}

impl std::ops::Div<u32> for Position {
    type Output = Position;

    fn div(self, rhs: u32) -> Self::Output {
        Position(
            self.0 / rhs,
            self.1 / rhs,
            self.2 / rhs,
        )
    }
}

impl std::ops::RemAssign<u32> for Position {
    fn rem_assign(&mut self, rhs: u32) {
        self.0 %= rhs;
        self.1 %= rhs;
        self.2 %= rhs;
    }
}

/// Octree is a data structure that subdivides three-dimensional space into octants. One octant
/// can contain up to 8 leaf nodes, or 8 child octants which further subdivide their parent octant
/// to contain 8 children/leaves.
/// The data structure is allocated in linearly without any nested pointer structs.
#[derive(Debug, PartialEq)]
pub struct Octree<T> {
    pub(super) root: Option<OctantId>,
    pub(super) octants: Vec<Octant<T>>,
    free_list: Vec<OctantId>,
    depth: u8,
}

impl<T> Octree<T> {
    pub fn new() -> Octree<T> {
        Octree { root: None, octants: Vec::new(), free_list: Vec::new(), depth: 0 }
    }

    pub fn with_size(size: u8) -> Octree<T> {
        let mut octree = Self::new();
        octree.expand_to(size);
        octree
    }

    pub fn reset(&mut self) {
        self.root = None;
        self.octants.clear();
        self.free_list.clear();
        self.depth = 0;
    }

    /// Adds the given leaf value at the given position. If the tree is not big enough yet,
    /// it is expanded. Children along the path are overridden, if any exist. Returns the
    /// new LeafId, that holds the leaf value, as well as any previous value that was overridden.
    pub fn set_leaf(&mut self, pos: Position, leaf: T) -> (LeafId, Option<T>) {
        self.expand_to(pos.required_depth());

        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

        while size >= 1 {
            size /= 2;
            let idx = (pos / size).idx();
            pos %= size;

            if size == 1 {
                let prev = self.octants[it as usize].set_child(idx, Child::Leaf(leaf));
                return (LeafId { parent: it, idx }, prev.into_leaf_value());
            }

            it = self.step_into_or_create_octant_at(it, idx);
        }

        unreachable!("could not reach end of tree");
    }

    /// Moves the leaf at `leaf_id` to the given position. The original leaf will be set to an
    /// empty octant. It returns the new LeafId at the given position, as well as the overridden
    /// leaf value at the target position, if any was present.
    pub fn move_leaf(&mut self, leaf_id: LeafId, to_pos: Position) -> (LeafId, Option<T>) {
        self.expand_to(to_pos.required_depth());

        let mut it = self.root.unwrap();
        let mut pos = to_pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

        while size >= 1 {
            size /= 2;
            let idx = (pos / size).idx();
            pos %= size;

            if size == 1 {
                // do nothing if leaf is replaced with itself
                if it == leaf_id.parent && idx == leaf_id.idx {
                    return (leaf_id, None);
                }

                // remove current leaf value, if any
                let old_leaf = self.octants[it as usize].set_child(idx, Child::None);

                // remove leaf from its previous parent
                let new_leaf = self.octants[leaf_id.parent as usize].set_child(leaf_id.idx, Child::None);

                // attach new leaf octant
                if new_leaf.get_leaf_value().is_some() {
                    self.octants[it as usize].set_child(idx, new_leaf);
                }

                let new_leaf_id = LeafId { parent: it, idx };
                match old_leaf {
                    Child::None => return (new_leaf_id, None),
                    Child::Octant(_) => unreachable!("found unexpected octant"),
                    Child::Leaf(value) => return (new_leaf_id, Some(value)),
                }
            }

            it = self.step_into_or_create_octant_at(it, idx);
        }

        unreachable!("could not reach end of tree");
    }

    fn step_into_or_create_octant_at(&mut self, it: OctantId, idx: u8) -> OctantId {
        match &self.octants[it as usize].children[idx as usize] {
            Child::None => {
                let prev_id = it;
                let next_id = self.new_octant(Some(prev_id));

                let prev_octant = &mut self.octants[prev_id as usize];
                prev_octant.set_child(idx, Child::Octant(next_id));

                next_id
            }
            Child::Octant(id) => *id,
            Child::Leaf(_) => unreachable!("found unexpected leaf"),
        }
    }

    /// Removes the leaf at the given position, if it exists. Empty parents are *not* removed from
    /// the tree. [`Octree::compact`] can be used for that. Returns the removed value and its
    /// LeafId.
    pub fn remove_leaf(&mut self, pos: Position) -> (Option<T>, Option<LeafId>) {
        if pos.required_depth() > self.depth {
            return (None, None);
        }

        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

        while size >= 1 {
            size /= 2;
            let idx = (pos / size).idx();
            pos %= size;

            match &self.octants[it as usize].children[idx as usize] {
                Child::None => break,
                Child::Octant(id) => it = *id,
                Child::Leaf(_) => {
                    match self.octants[it as usize].set_child(idx, Child::None) {
                        Child::None => return (None, None),
                        Child::Octant(_) => unreachable!("found unexpected octant"),
                        Child::Leaf(value) => return (Some(value), Some(LeafId { parent: it, idx })),
                    }
                }
            }
        }

        (None, None)
    }

    /// Removes the leaf for the given LeafId and returns its value.
    pub fn remove_leaf_by_id(&mut self, leaf_id: LeafId) -> Option<T> {
        match &self.octants[leaf_id.parent as usize].children[leaf_id.idx as usize] {
            Child::None => None,
            Child::Octant(_) => None,
            Child::Leaf(_) => {
                match self.octants[leaf_id.parent as usize].set_child(leaf_id.idx, Child::None) {
                    Child::None => None,
                    Child::Octant(_) => unreachable!("found unexpected octant"),
                    Child::Leaf(value) => Some(value),
                }
            }
        }
    }

    /// Returns a reference to the value of the leaf at the given position, if it exists.
    pub fn get_leaf(&self, pos: Position) -> Option<&T> {
        let mut it = self.root.unwrap();
        let mut pos = pos;
        let mut size = 2f32.pow(self.depth as i32) as u32;

        while size > 0 {
            size /= 2;
            let idx = (pos / size).idx();
            pos %= size;

            let child = &self.octants[it as usize].children[idx as usize];
            if child.is_none() {
                break;
            }

            match &self.octants[it as usize].children[idx as usize] {
                Child::None => break,
                Child::Octant(id) => it = *id,
                Child::Leaf(value) => return Some(value),
            }
        }

        None
    }

    /// Expands the octant's depth by the given value. If necessary, the existing root octant
    /// is wrapped in new parent octants.
    pub fn expand(&mut self, by: u8) {
        for _ in 0..by {
            let new_root_id = self.new_octant(None);

            if let Some(root_id) = self.root {
                self.octants[root_id as usize].parent = Some(new_root_id);
                self.octants[new_root_id as usize].set_child(0, Child::Octant(root_id));
            }

            self.root = Some(new_root_id);
        }

        self.depth += by
    }

    /// Expands the octant's depth to by equal to the given value. If the depth is already larger,
    /// nothing happens. For shrinking empty octant space, [`Octree::compact`] can be used.
    pub fn expand_to(&mut self, to: u8) {
        if self.depth > to {
            return;
        }
        let diff = to - self.depth;
        if diff > 0 {
            self.expand(diff);
        }
    }

    /// Removes all octants from the tree that have no children and no content. The algorithm
    /// is depth first, so removal cascades through the whole tree removing all empty children
    /// as well.
    pub fn compact(&mut self) {
        if self.root.is_none() {
            return;
        }

        self.compact_octant(self.root.unwrap());

        if self.octants[self.root.unwrap() as usize].children_count != 0 {
            return;
        }

        self.reset();
    }

    fn compact_octant(&mut self, octant_id: OctantId) {
        let octant = &self.octants[octant_id as usize];
        let mut children = Vec::new();
        for (i, child) in octant.children.iter().enumerate() {
            if let Child::Octant(id) = child {
                children.push((i, *id));
            }
        }

        for child in children {
            self.compact_octant(child.1);

            let octant = &self.octants[child.1 as usize];
            if octant.children_count == 0 {
                self.delete_octant(child.1);
                self.octants[octant_id as usize].set_child(child.0 as u8, Child::None);
            }
        }
    }

    /// Returns either an available octant from the octree's free list, or allocates a new one.
    fn new_octant(&mut self, parent: Option<OctantId>) -> OctantId {
        if let Some(free_id) = self.free_list.pop() {
            self.octants[free_id as usize].parent = parent;
            return free_id;
        }

        let id = self.octants.len() as OctantId;
        self.octants.push(Octant {
            parent,
            children_count: 0,
            children: Default::default(),
        });
        id
    }

    /// Resets the given octant and adds it to the octree's free list.
    fn delete_octant(&mut self, id: OctantId) {
        if let Some(parent) = self.octants[id as usize].parent {
            let children = &self.octants[parent as usize].children;
            let idx = children.iter().position(|x| x.get_octant_value() == Some(id));
            if let Some(idx) = idx {
                self.octants[parent as usize].set_child(idx as u8, Child::None);
            }
        }

        let octant = &mut self.octants[id as usize];
        octant.parent = None;
        octant.children_count = 0;
        for ch in &mut octant.children {
            *ch = Child::None;
        }

        self.free_list.push(id);
    }

    /// Returns the octree's depth.
    pub fn depth(&self) -> u8 {
        self.depth
    }
}

/// Child represents possible states for an octant in the octree.
#[derive(Debug, Default)]
pub(super) enum Child<T> {
    #[default]
    None,
    Octant(OctantId),
    Leaf(T),
}

impl<T> Child<T> {
    pub fn is_none(&self) -> bool {
        matches!(self, Child::None)
    }

    pub fn is_octant(&self) -> bool {
        matches!(self, Child::Octant(_))
    }

    pub fn get_octant_value(&self) -> Option<OctantId> {
        match self {
            Child::Octant(id) => Some(*id),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Child::Leaf(_))
    }

    pub fn get_leaf_value(&self) -> Option<&T> {
        match self {
            Child::Leaf(value) => Some(value),
            _ => None,
        }
    }

    pub fn get_leaf_value_mut(&mut self) -> Option<&mut T> {
        match self {
            Child::Leaf(value) => Some(value),
            _ => None,
        }
    }

    pub fn into_leaf_value(self) -> Option<T> {
        match self {
            Child::Leaf(value) => Some(value),
            _ => None,
        }
    }
}

impl<T: PartialEq> PartialEq for Child<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Child::None, Child::None) => true,
            (Child::Octant(l), Child::Octant(r)) => l == r,
            (Child::Leaf(l), Child::Leaf(r)) => l == r,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(super) struct Octant<T> {
    parent: Option<OctantId>,
    children_count: u8,
    pub(super) children: [Child<T>; 8],
}

impl<T> Octant<T> {
    fn set_child(&mut self, idx: u8, child: Child<T>) -> Child<T> {
        let idx = idx as usize;

        if self.children[idx].is_none() && !child.is_none() {
            self.children_count += 1;
        }
        if !self.children[idx].is_none() && child.is_none() {
            self.children_count -= 1;
        }

        let mut child = child;
        mem::swap(&mut child, &mut self.children[idx]);
        child
    }
}

//noinspection DuplicatedCode
#[cfg(test)]
mod tests {
    use Child::*;

    use crate::world::octree::{Child, LeafId, Octant, Octree, Position};

    /// Tests that adding a leaf at a depth > 1 results in the correct octree state.
    #[test]
    fn octree_add_leaf_single() {
        let mut octree = Octree::new();

        assert_eq!(octree.set_leaf(Position(1, 1, 3), 20), (LeafId { parent: 2, idx: 7 }, Option::None));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                },
                Octant {
                    parent: Option::None,
                    children: [Octant(0), None, None, None, Octant(2), None, None, None],
                    children_count: 2,
                },
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, None, Leaf(20)],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(1),
            depth: 2,
        });

        assert_eq!(octree.get_leaf(Position(1, 1, 3)), Some(&20));
        assert_eq!(octree.get_leaf(Position(1, 1, 1)), Option::None);
    }

    /// Tests that adding multiple leaves at different depths results in the correct octree state.
    #[test]
    fn octree_add_leaf_multiple() {
        let mut octree = Octree::new();

        assert_eq!(octree.set_leaf(Position(6, 7, 5), 10), (LeafId { parent: 4, idx: 6 }, Option::None));
        assert_eq!(octree.set_leaf(Position(0, 0, 0), 20), (LeafId { parent: 0, idx: 0 }, Option::None));
        assert_eq!(octree.set_leaf(Position(1, 0, 6), 30), (LeafId { parent: 6, idx: 1 }, Option::None));

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    children: [Leaf(20), None, None, None, None, None, None, None],
                    children_count: 1,
                },
                Octant {
                    parent: Some(2),
                    children: [Octant(0), None, None, None, None, None, None, None],
                    children_count: 1,
                },
                Octant { // root node
                    parent: Option::None,
                    children: [Octant(1), None, None, None, Octant(5), None, None, Octant(3)],
                    children_count: 3,
                },
                Octant {
                    parent: Some(2),
                    children: [None, None, None, Octant(4), None, None, None, None],
                    children_count: 1,
                },
                Octant {
                    parent: Some(3),
                    children: [None, None, None, None, None, None, Leaf(10), None],
                    children_count: 1,
                },
                Octant {
                    parent: Some(2),
                    children: [None, None, None, None, Octant(6), None, None, None],
                    children_count: 1,
                },
                Octant {
                    parent: Some(5),
                    children: [None, Leaf(30), None, None, None, None, None, None],
                    children_count: 1,
                },
                // third node end
            ],
            free_list: vec![],
            root: Some(2),
            depth: 3,
        });

        assert_eq!(octree.get_leaf(Position(6, 7, 5)), Some(&10));
        assert_eq!(octree.get_leaf(Position(0, 0, 0)), Some(&20));
        assert_eq!(octree.get_leaf(Position(1, 0, 6)), Some(&30));
        assert_eq!(octree.get_leaf(Position(1, 1, 1)), Option::None);

        // replace by adding
        assert_eq!(octree.set_leaf(Position(0, 0, 0), 40), (LeafId { parent: 0, idx: 0 }, Some(20)));
        assert_eq!(octree.get_leaf(Position(0, 0, 0)), Some(&40));
    }

    /// Tests that setting a leaf can override an already existing leaf.
    #[test]
    fn octree_add_leaf_replacing() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [Leaf(10), None, None, None, None, None, None, None],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        };

        octree.set_leaf(Position(0, 0, 0), 20);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [Leaf(20), None, None, None, None, None, None, None],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });
    }

    /// Tests that removal and addition of leaves at the same position works.
    #[test]
    fn octree_remove_and_add_leaf() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [Leaf(10), Leaf(20), None, None, None, None, None, None],
                    children_count: 2,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        };

        assert_eq!(octree.remove_leaf(Position(0, 0, 0)), (Some(10), Some(LeafId { parent: 0, idx: 0 })));
        assert_eq!(octree.remove_leaf_by_id(LeafId { parent: 0, idx: 1 }), Some(20));

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });

        octree.set_leaf(Position(0, 0, 0), 30);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [Leaf(30), None, None, None, None, None, None, None],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });
    }

    /// Tests that moving a leaf around in the octree results in the correct state. Also tests that
    /// all expected edge cases are covered.
    #[test]
    fn octree_move_leaf() {
        let mut octree = Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [Leaf(10), None, None, None, None, None, None, Leaf(20)],
                    children_count: 2,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        };

        // replace at empty slot
        let previous = octree.move_leaf(LeafId { parent: 0, idx: 0 }, Position(1, 0, 0));
        assert_eq!(previous, (LeafId { parent: 0, idx: 1 }, Option::None));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [None, Leaf(10), None, None, None, None, None, Leaf(20)],
                    children_count: 2,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });

        // replace with itself
        let previous = octree.move_leaf(LeafId { parent: 0, idx: 1 }, Position(1, 0, 0));
        assert_eq!(previous, (LeafId { parent: 0, idx: 1 }, Option::None));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [None, Leaf(10), None, None, None, None, None, Leaf(20)],
                    children_count: 2,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });

        // replace with existing
        let previous = octree.move_leaf(LeafId { parent: 0, idx: 1 }, Position(1, 1, 1));
        assert_eq!(previous, (LeafId { parent: 0, idx: 7 }, Some(20)));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [None, None, None, None, None, None, None, Leaf(10)],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(0),
            depth: 1,
        });

        // replace in new parent
        let previous = octree.move_leaf(LeafId { parent: 0, idx: 7 }, Position(2, 0, 0));
        assert_eq!(previous, (LeafId { parent: 2, idx: 0 }, Option::None));
        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                },
                Octant {
                    parent: Option::None,
                    children: [Octant(0), Octant(2), None, None, None, None, None, None],
                    children_count: 2,
                },
                Octant {
                    parent: Some(1),
                    children: [Leaf(10), None, None, None, None, None, None, None],
                    children_count: 1,
                },
            ],
            free_list: vec![],
            root: Some(1),
            depth: 2,
        });
    }

    /// Tests that compacting an octree after removing all leaves works as expected.
    #[test]
    fn octree_compact() {
        let mut octree = Octree::new();

        octree.set_leaf(Position(0, 1, 3), 10);
        octree.set_leaf(Position(1, 1, 3), 20);

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                },
                Octant {
                    parent: Option::None,
                    children: [Octant(0), None, None, None, Octant(2), None, None, None],
                    children_count: 2,
                },
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, Leaf(10), Leaf(20)],
                    children_count: 2,
                },
            ],
            free_list: vec![],
            root: Some(1),
            depth: 2,
        });

        octree.compact();

        assert_eq!(octree, Octree {
            octants: vec![
                Octant {
                    parent: Option::None,
                    children: [None, None, None, None, None, None, None, None],
                    children_count: 0,
                },
                Octant {
                    parent: Option::None,
                    children: [None, None, None, None, Octant(2), None, None, None],
                    children_count: 1,
                },
                Octant {
                    parent: Some(1),
                    children: [None, None, None, None, None, None, Leaf(10), Leaf(20)],
                    children_count: 2,
                },
            ],
            free_list: vec![0],
            root: Some(1),
            depth: 2,
        });

        octree.remove_leaf(Position(0, 1, 3));
        octree.remove_leaf(Position(1, 1, 3));
        octree.compact();

        assert_eq!(octree, Octree {
            octants: vec![],
            free_list: vec![],
            root: Option::None,
            depth: 0,
        });
    }
}
