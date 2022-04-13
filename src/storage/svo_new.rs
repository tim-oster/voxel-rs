use std::collections::HashMap;

use crate::chunk::BlockId;
use crate::storage::octree::{Octant, OctantId, Octree, Position};

pub struct Svo<T: SvoSerializable> {
    octree: Octree<T>,
}

impl<T: SvoSerializable> Svo<T> {
    fn new() -> Svo<T> {
        Svo {
            octree: Octree::new(),
        }
    }
}

impl<T: SvoSerializable> Svo<T> {
    pub fn set(&mut self, pos: Position, leaf: Option<T>) {
        if let Some(leaf) = leaf {
            self.octree.add_leaf(pos, leaf);
        } else {
            self.octree.remove_leaf(pos);
        }
    }

    pub fn serialize(&self) -> SvoBuffer {
        if self.octree.root.is_none() {
            return SvoBuffer { header_mask: 0, depth: 0, bytes: vec![] };
        }

        // TODO figure out correct sizes
        let mut bytes = Vec::new();
        bytes.push(0); // will become header mask
        bytes.push(0);
        bytes.push(0);
        bytes.push(0);
        bytes.push(5); // absolute pointer to where the actual octree starts

        let (header_mask, depth) = serialize_octree(&self.octree, &mut bytes);
        bytes[0] = header_mask as u32;

        SvoBuffer { header_mask, depth, bytes }
    }
}

fn serialize_octree<T: SvoSerializable>(octree: &Octree<T>, dst: &mut Vec<u32>) -> (u16, u32) {
    let root_id = octree.root.unwrap();
    let root = &octree.octants[root_id];

    let (header_mask, _, ptrs) = serialize_octant(octree, root, dst);
    let header_mask = (header_mask as u16) << 8; // convert from leaf to child mask

    // TODO the leaf now points to the spot containing the value, this has to be changed in the shader as well

    // TODO this must be deterministic, currently map access is in random order
    for (octant_id, index) in ptrs {
        let octant = &octree.octants[octant_id];
        dst[index] = dst.len() as u32;
        octant.content.as_ref().unwrap().serialize_to(dst);
    }

    (header_mask, octree.depth) // TODO plus size of deepest child
}

fn serialize_octant<T: SvoSerializable>(octree: &Octree<T>, octant: &Octant<T>, dst: &mut Vec<u32>) -> (u32, u32, HashMap<OctantId, usize>) {
    let start_offset = dst.len();

    dst.reserve(12);
    dst.extend(std::iter::repeat(0).take(12));

    let mut child_mask = 0u32;
    let mut leaf_mask = 0u32;
    let mut pointers = HashMap::new();

    for (idx, child) in octant.children.iter().enumerate() {
        if child.is_none() {
            continue;
        }

        child_mask |= 1 << idx;

        let child_id = child.unwrap();
        let child = &octree.octants[child_id];

        if child.content.is_some() {
            leaf_mask |= 1 << idx;
            pointers.insert(child_id, start_offset + 4 + idx);
        } else {
            let (child_mask, leaf_mask, child_ptrs) =
                serialize_octant(octree, &octree.octants[child_id], dst);

            let mut mask = ((child_mask as u32) << 8) | leaf_mask as u32;
            if (idx % 2) != 0 {
                mask <<= 16;
            }
            dst[start_offset + (idx / 2) as usize] |= mask;

            dst[start_offset + 4 + idx] = 12 - 4 - idx as u32; // offset from pointer to end of octant block
            dst[start_offset + 4 + idx] |= 1 << 31; // flag as relative pointer

            pointers.extend(child_ptrs);
        }
    }

    (child_mask, leaf_mask, pointers)
}

#[derive(Debug, PartialEq)]
pub struct SvoBuffer {
    pub header_mask: u16,
    pub depth: u32,
    pub bytes: Vec<u32>,
}

pub trait SvoSerializable {
    fn serialize_to(&self, dst: &mut Vec<u32>);
}

impl<T: SvoSerializable> SvoSerializable for Octree<T> {
    fn serialize_to(&self, dst: &mut Vec<u32>) {
        serialize_octree(self, dst);
    }
}

impl SvoSerializable for BlockId {
    fn serialize_to(&self, dst: &mut Vec<u32>) {
        dst.push(*self as u32);
    }
}

#[cfg(test)]
mod tests {
    use crate::chunk::BlockId;
    use crate::storage::octree::{Octree, Position};
    use crate::storage::svo_new::{Svo, SvoBuffer};

    #[test]
    fn svo_serialize() {
        let mut svo = Svo::new();
        svo.set(Position(0, 0, 0), Some(100 as BlockId));
        svo.set(Position(1, 1, 1), Some(200 as BlockId));
        svo.octree.expand_to(5);

        assert_eq!(svo.serialize(), SvoBuffer {
            header_mask: 1 << 8,
            depth: 5,
            bytes: vec![
                // svo header
                1 << 8,
                0,
                0,
                0,
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
                65, 0, 0, 0,
                0, 0, 0, 66,
                // content
                100,
                200,
            ],
        });
    }

    // TODO test with multiple octrees in svo
    // #[test]
    // fn octree_serialize_2() {
    //     let mut octree = Octree::new();
    //     octree.add_leaf(Position(31, 0, 0), 1 as BlockId);
    //     octree.add_leaf(Position(0, 31, 0), 2 as BlockId);
    //     octree.add_leaf(Position(0, 0, 31), 3 as BlockId);
    //
    //     let svo = octree.serialize();
    //     assert_eq!(svo, SvoBuffer {
    //         header_mask: (2 | 4 | 16) << 8,
    //         depth: 5,
    //         bytes: vec![
    //             // core octant header
    //             (2 << 8) << 16,
    //             4 << 8,
    //             16 << 8,
    //             0,
    //             // core octant body
    //             0, (1 << 31) | 7, (1 << 31) | (6 + 4 * 12), 0,
    //             (1 << 31) | (4 + 8 * 12), 0, 0, 0,
    //
    //             // subtree for (1,0,0)
    //             // header 1
    //             2 << 8 << 16,
    //             0,
    //             0,
    //             0,
    //             // body 1
    //             0, (1 << 31) | 7, 0, 0,
    //             0, 0, 0, 0,
    //             // header 2
    //             2 << 8 << 16,
    //             0,
    //             0,
    //             0,
    //             // body 2
    //             0, (1 << 31) | 7, 0, 0,
    //             0, 0, 0, 0,
    //             // header 3
    //             ((2 << 8) | 2) << 16,
    //             0,
    //             0,
    //             0,
    //             // body 3
    //             0, (1 << 31) | 7, 0, 0,
    //             0, 0, 0, 0,
    //             // leaf header
    //             0,
    //             0,
    //             0,
    //             0,
    //             // leaf body
    //             0, 1, 0, 0,
    //             0, 0, 0, 0,
    //
    //             // subtree for (0,1,0)
    //             // header 1
    //             0,
    //             4 << 8,
    //             0,
    //             0,
    //             // body 1
    //             0, 0, (1 << 31) | 6, 0,
    //             0, 0, 0, 0,
    //             // header 2
    //             0,
    //             4 << 8,
    //             0,
    //             0,
    //             // body 2
    //             0, 0, (1 << 31) | 6, 0,
    //             0, 0, 0, 0,
    //             // header 3
    //             0,
    //             4 << 8 | 4,
    //             0,
    //             0,
    //             // body 3
    //             0, 0, (1 << 31) | 6, 0,
    //             0, 0, 0, 0,
    //             // leaf header
    //             0,
    //             0,
    //             0,
    //             0,
    //             // leaf body
    //             0, 0, 2, 0,
    //             0, 0, 0, 0,
    //
    //             // subtree for (0,0,1)
    //             // header 1
    //             0,
    //             0,
    //             16 << 8,
    //             0,
    //             // body 1
    //             0, 0, 0, 0,
    //             (1 << 31) | 4, 0, 0, 0,
    //             // header 2
    //             0,
    //             0,
    //             16 << 8,
    //             0,
    //             // body 2
    //             0, 0, 0, 0,
    //             (1 << 31) | 4, 0, 0, 0,
    //             // header 3
    //             0,
    //             0,
    //             16 << 8 | 16,
    //             0,
    //             // body 3
    //             0, 0, 0, 0,
    //             (1 << 31) | 4, 0, 0, 0,
    //             // leaf header
    //             0,
    //             0,
    //             0,
    //             0,
    //             // leaf body
    //             0, 0, 0, 0,
    //             3, 0, 0, 0,
    //         ],
    //     });
    // }
}
