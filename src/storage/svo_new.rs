use std::cmp::max;
use std::collections::HashMap;

use crate::chunk::BlockId;
use crate::storage::octree::{Octant, OctantId, Octree, Position};

pub struct Svo<T: SvoSerializable> {
    octree: Octree<T>,
}

impl<T: SvoSerializable> Svo<T> {
    pub fn new() -> Svo<T> {
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

        // add svo "hull" to allow the raytracer to enter
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

    let (child_mask, leaf_mask, ptrs) = serialize_octant(octree, root, dst);
    let mask = ((child_mask as u16) << 8) | leaf_mask as u16;

    let mut max_depth = octree.depth;
    for ptr in ptrs.iter() {
        let octant = &octree.octants[ptr.octant];
        dst[ptr.buffer_index] = dst.len() as u32;
        let depth = octant.content.as_ref().unwrap().serialize_to(ptr, dst);
        max_depth = max(max_depth, octree.depth + depth);
    }

    (mask, max_depth)
}

fn serialize_octant<T: SvoSerializable>(octree: &Octree<T>, octant: &Octant<T>, dst: &mut Vec<u32>) -> (u32, u32, Vec<MissingPointer>) {
    let start_offset = dst.len();

    dst.reserve(12);
    dst.extend(std::iter::repeat(0).take(12));

    let mut child_mask = 0u32;
    let mut leaf_mask = 0u32;
    let mut pointers = Vec::new();

    for (idx, child) in octant.children.iter().enumerate() {
        if child.is_none() {
            continue;
        }

        child_mask |= 1 << idx;

        let child_id = child.unwrap();
        let child = &octree.octants[child_id];

        if let Some(content) = &child.content {
            if !content.is_nested() {
                leaf_mask |= 1 << idx;
            }
            pointers.push(MissingPointer {
                octant: child_id,
                child_idx: idx,
                buffer_index: start_offset + 4 + idx,
            });
        } else {
            let child_offset = (dst.len() - start_offset) as u32;
            let (child_mask, leaf_mask, child_ptrs) =
                serialize_octant(octree, &octree.octants[child_id], dst);

            let mut mask = ((child_mask as u32) << 8) | leaf_mask as u32;
            if (idx % 2) != 0 {
                mask <<= 16;
            }
            dst[start_offset + (idx / 2) as usize] |= mask;

            dst[start_offset + 4 + idx] = child_offset - 4 - idx as u32; // offset from pointer to start of next block
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

pub struct MissingPointer {
    pub octant: OctantId,
    pub child_idx: usize,
    pub buffer_index: usize,
}

pub trait SvoSerializable {
    // TODO is is_nested clean?
    fn is_nested(&self) -> bool;
    fn serialize_to(&self, at: &MissingPointer, dst: &mut Vec<u32>) -> u32;
}

impl<T: SvoSerializable> SvoSerializable for Octree<T> {
    fn is_nested(&self) -> bool {
        true
    }

    fn serialize_to(&self, at: &MissingPointer, dst: &mut Vec<u32>) -> u32 {
        let (header_mask, depth) = serialize_octree(self, dst);

        // TODO is there a cleaner way to implement this header replacement?
        // encode header information in parent octant (it is empty because this looks like a leaf node to the parent)
        let block_start = at.buffer_index - at.child_idx - 4;
        let header_pos = block_start + (at.child_idx / 2) as usize;

        let mut mask = 0xffff0000 as u32;
        let mut header_mask = header_mask as u32;
        if (at.child_idx % 2) != 0 {
            mask = !mask;
            header_mask <<= 16;
        }
        dst[header_pos] = (dst[header_pos] & mask) | header_mask;

        depth
    }
}

impl SvoSerializable for BlockId {
    fn is_nested(&self) -> bool {
        false
    }

    fn serialize_to(&self, at: &MissingPointer, dst: &mut Vec<u32>) -> u32 {
        dst.push(*self as u32);
        0
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
        svo.octree.compact();

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

    #[test]
    fn svo_serialize_nested() {
        let mut octree = Octree::new();
        octree.add_leaf(Position(31, 0, 0), 1 as BlockId);
        octree.add_leaf(Position(0, 31, 0), 2 as BlockId);
        octree.add_leaf(Position(0, 0, 31), 3 as BlockId);
        octree.expand_to(5);
        octree.compact();

        let mut svo = Svo::new();
        svo.set(Position(1, 0, 0), Some(octree));

        let svo = svo.serialize();
        assert_eq!(svo, SvoBuffer {
            header_mask: 2 << 8,
            depth: 6,
            bytes: vec![
                // svo header
                2 << 8,
                0,
                0,
                0,
                5,

                // outer octree, first node header
                (2 | 4 | 16) << 8 << 16,
                0,
                0,
                0,
                // outer octree, first node body
                0, 17, 0, 0,
                0, 0, 0, 0,

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
                0, 173, 0, 0,
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
                0, 0, 174, 0,
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
                175, 0, 0, 0,

                // leaf values
                1,
                2,
                3,
            ],
        });
    }
}
