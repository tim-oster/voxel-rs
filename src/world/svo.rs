use std::collections::{HashMap, HashSet};
use std::ptr;
use std::rc::Rc;

use crate::{BlockId, ChunkStorage, Octree, Position};
use crate::world::octree::{Octant, OctantId};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum OctantChange {
    Add(OctantId),
    Remove(OctantId),
}

pub trait SvoSerializable {
    fn serialize(&self, dst: &mut Vec<u32>) -> SerializationResult;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SerializationResult {
    child_mask: u16,
    leaf_mask: u16,
    depth: u32,
}

pub struct Svo<T: SvoSerializable> {
    octree: Octree<T>,
    change_set: HashSet<OctantChange>,

    buffer: SvoBuffer,
    octant_info: HashMap<OctantId, OctantInfo>,
    root_octant_info: Option<OctantInfo>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct OctantInfo {
    buf_offset: usize,
    serialization: SerializationResult,
}

impl<T: SvoSerializable> Svo<T> {
    pub fn new() -> Svo<T> {
        Svo {
            octree: Octree::new(),
            change_set: HashSet::new(),
            buffer: SvoBuffer::new(0), // TODO find a good size
            octant_info: HashMap::new(),
            root_octant_info: None,
        }
    }

    pub fn set(&mut self, pos: Position, leaf: Option<T>) {
        if let Some(leaf) = leaf {
            let id = self.octree.add_leaf(pos, leaf);
            self.change_set.insert(OctantChange::Add(id));
        } else {
            if let Some(id) = self.octree.remove_leaf(pos) {
                self.change_set.insert(OctantChange::Remove(id));
            }
        }
    }

    // TODO also returns what has changed inside the buffer
    pub fn serialize(&mut self) {
        if self.octree.root.is_none() || self.change_set.is_empty() {
            return;
        }

        // rebuild & remove all changed leaf octants
        let mut octant_buffer = Vec::with_capacity(0); // TODO figure out good size
        for change in self.change_set.drain() {
            match change {
                OctantChange::Add(id) => {
                    // TODO is there any case where this might be None?
                    let octant = self.octree.octants[id].content.as_ref().unwrap();
                    let result = octant.serialize(&mut octant_buffer);
                    if result.depth > 0 {
                        let offset = self.buffer.insert(id, &octant_buffer);
                        octant_buffer.clear();

                        self.octant_info.insert(id, OctantInfo { buf_offset: offset, serialization: result });
                    }
                }
                OctantChange::Remove(id) => {
                    self.buffer.remove(id);
                    self.octant_info.remove(&id);
                }
            }
        }

        // rebuild root octree
        let root_id = self.octree.root.unwrap();
        let root = &self.octree.octants[root_id];
        let result = self.serialize_octant(root, &mut octant_buffer);

        let offset = self.buffer.insert(usize::MAX, &octant_buffer);
        self.root_octant_info = Some(OctantInfo { buf_offset: offset, serialization: result });
    }

    // TODO code duplication
    fn serialize_octant(&self, octant: &Octant<T>, dst: &mut Vec<u32>) -> SerializationResult {
        let start_offset = dst.len();

        dst.reserve(12);
        dst.extend(std::iter::repeat(0).take(12));

        let mut result = SerializationResult {
            child_mask: 0,
            leaf_mask: 0,
            depth: 0,
        };

        for (idx, child) in octant.children.iter().enumerate() {
            if child.is_none() {
                continue;
            }

            result.child_mask |= 1 << idx;

            let child_id = child.unwrap();
            let child = &self.octree.octants[child_id];

            if let Some(content) = &child.content {
                // TODO is there any case where this might be None?
                let info = self.octant_info.get(&child_id).unwrap();

                let mut mask = ((info.serialization.child_mask as u32) << 8) | info.serialization.leaf_mask as u32;
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                dst[start_offset + (idx / 2) as usize] |= mask;

                dst[start_offset + 4 + idx] = info.buf_offset as u32 + 5; // TODO hardcoded offset

                result.depth = result.depth.max(info.serialization.depth + 1);
            } else {
                let child_offset = (dst.len() - start_offset) as u32;
                let child_result = self.serialize_octant(&self.octree.octants[child_id], dst);

                let mut mask = ((child_result.child_mask as u32) << 8) | child_result.leaf_mask as u32;
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                dst[start_offset + (idx / 2) as usize] |= mask;

                dst[start_offset + 4 + idx] = child_offset - 4 - idx as u32; // offset from pointer to start of next block
                dst[start_offset + 4 + idx] |= 1 << 31; // flag as relative pointer

                result.depth = result.depth.max(child_result.depth + 1);
            }
        }

        result
    }

    pub unsafe fn write_to(&self, dst: *mut u32) -> usize {
        if self.root_octant_info.is_none() {
            return 0;
        }

        let start = dst as usize;
        let info = self.root_octant_info.unwrap();

        // TODO do that here?
        let max_depth_exp2 = (-(info.serialization.depth as f32)).exp2();
        dst.write(max_depth_exp2.to_bits());

        let dst = dst.offset(1);
        dst.offset(0).write((info.serialization.child_mask as u32) << 8);
        dst.offset(1).write(0);
        dst.offset(2).write(0);
        dst.offset(3).write(0);
        dst.offset(4).write(info.buf_offset as u32 + 5); // TODO hardcoded offset

        // TODO only copy changed ranges
        // for id in &svo_buffer.changed_octants {
        //     let range = svo_buffer.buffer.octant_to_range.get(id).unwrap();
        //     unsafe {
        //         ptr::copy(svo_buffer.buffer.bytes.as_ptr().offset(range.start as isize),
        //                   world_buffer.offset(1 + range.start as isize),
        //                   range.length);
        //     }
        // }

        let dst = dst.offset(5);
        let len = self.buffer.bytes.len();
        ptr::copy(self.buffer.bytes.as_ptr(), dst, len);

        let dst = dst.offset(len as isize);
        ((dst as usize) - start) / 4
    }
}

impl SvoSerializable for Rc<ChunkStorage> {
    fn serialize(&self, dst: &mut Vec<u32>) -> SerializationResult {
        self.get_octree_ref().serialize(dst)
    }
}

impl SvoSerializable for Octree<BlockId> {
    fn serialize(&self, dst: &mut Vec<u32>) -> SerializationResult {
        if self.root.is_none() {
            return SerializationResult { child_mask: 0, leaf_mask: 0, depth: 0 };
        }

        let root_id = self.root.unwrap();
        let root = &self.octants[root_id];
        self.serialize_octant(root, dst)
    }
}

impl Octree<BlockId> {
    fn serialize_octant(&self, octant: &Octant<BlockId>, dst: &mut Vec<u32>) -> SerializationResult {
        let start_offset = dst.len();

        dst.reserve(12);
        dst.extend(std::iter::repeat(0).take(12));

        let mut result = SerializationResult {
            child_mask: 0,
            leaf_mask: 0,
            depth: 0,
        };

        for (idx, child) in octant.children.iter().enumerate() {
            if child.is_none() {
                continue;
            }

            result.child_mask |= 1 << idx;

            let child_id = child.unwrap();
            let child = &self.octants[child_id];

            if let Some(content) = &child.content {
                result.leaf_mask |= 1 << idx;
                dst[start_offset + 4 + idx] = *content as u32;
                result.depth = 1;
            } else {
                let child_offset = (dst.len() - start_offset) as u32;
                let child_result = self.serialize_octant(&self.octants[child_id], dst);

                let mut mask = ((child_result.child_mask as u32) << 8) | child_result.leaf_mask as u32;
                if (idx % 2) != 0 {
                    mask <<= 16;
                }
                dst[start_offset + (idx / 2) as usize] |= mask;

                dst[start_offset + 4 + idx] = child_offset - 4 - idx as u32; // offset from pointer to start of next block
                dst[start_offset + 4 + idx] |= 1 << 31; // flag as relative pointer

                result.depth = result.depth.max(child_result.depth + 1);
            }
        }

        result
    }
}

// TODO write more tests
#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::chunk::BlockId;
    use crate::Svo;
    use crate::world::octree::{Octree, Position};
    use crate::world::svo::{OctantInfo, Range, SerializationResult, SvoBuffer};

// #[test]
    // fn svo_serialize() {
    //     let mut svo = Svo::new();
    //     svo.set(Position(0, 0, 0), Some(100 as BlockId));
    //     svo.set(Position(1, 1, 1), Some(200 as BlockId));
    //     svo.octree.expand_to(5);
    //     svo.octree.compact();
    //
    //     assert_eq!(svo.serialize(), SerializedSvo {
    //         header_mask: 1 << 8,
    //         depth: 5,
    //         buffer: SvoBuffer {
    //             root_mask: 1 << 8,
    //             depth: 5,
    //             bytes: vec![
    //                 // svo header
    //                 1 << 8,
    //                 0,
    //                 0,
    //                 0,
    //                 5,
    //                 // first octant header
    //                 1 << 8,
    //                 0,
    //                 0,
    //                 0,
    //                 // first octant body
    //                 (1 << 31) | 8, 0, 0, 0,
    //                 0, 0, 0, 0,
    //                 // second octant header
    //                 1 << 8,
    //                 0,
    //                 0,
    //                 0,
    //                 // second octant body
    //                 (1 << 31) | 8, 0, 0, 0,
    //                 0, 0, 0, 0,
    //                 // third octant header
    //                 1 << 8,
    //                 0,
    //                 0,
    //                 0,
    //                 // third octant body
    //                 (1 << 31) | 8, 0, 0, 0,
    //                 0, 0, 0, 0,
    //                 // fourth octant header
    //                 ((1 | 128) << 8) | (1 | 128),
    //                 0,
    //                 0,
    //                 0,
    //                 // fourth octant body
    //                 (1 << 31) | 8, 0, 0, 0,
    //                 0, 0, 0, 0,
    //                 // fifth octant header
    //                 0,
    //                 0,
    //                 0,
    //                 0,
    //                 // fifth octant body
    //                 65, 0, 0, 0,
    //                 0, 0, 0, 66,
    //                 // content
    //                 100,
    //                 200,
    //             ],
    //             free_ranges: vec![],
    //             octant_to_range: HashMap::from([
    //                 (1, Range { start: 65, length: 1, ptr: MissingPointer { octant: 1, child_idx: 0, buffer_index: 57 } }),
    //                 (2, Range { start: 66, length: 1, ptr: MissingPointer { octant: 2, child_idx: 7, buffer_index: 64 } }),
    //                 (6, Range { start: 5, length: 60, ptr: MissingPointer { octant: 0, child_idx: 0, buffer_index: 5 } }),
    //             ]),
    //             rebuild_targets: Default::default(),
    //         },
    //         changed_octants: HashSet::from([1, 2, 6]),
    //     });
    // }

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
        svo.serialize();

        assert_eq!(svo.root_octant_info, Some(OctantInfo {
            buf_offset: 156,
            serialization: SerializationResult {
                child_mask: 2,
                leaf_mask: 0,
                depth: 6,
            },
        }));

        let expected = vec![
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

            // outer octree, first node header
            (2 | 4 | 16) << 8 << 16,
            0,
            0,
            0,
            // outer octree, first node body
            0, 5 /*points to first octree*/, 0, 0, // TODO hardcoded offset
            0, 0, 0, 0,
        ];
        assert_eq!(svo.buffer, SvoBuffer {
            bytes: expected.clone(),
            free_ranges: vec![],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 156 }),
                (usize::MAX, Range { start: 156, length: 12 }),
            ]),
        });

        let mut buffer = Vec::new();
        buffer.resize(200, 0);
        let size = unsafe { svo.write_to(buffer.as_mut_ptr()) };
        assert_eq!(buffer[..size], [
            vec![
                // max depth exponent
                (-6f32).exp2().to_bits(),

                // svo header
                2 << 8,
                0,
                0,
                0,
                156 + 5,// TODO hardcoded offset
            ],
            expected,
        ].concat());
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct Range {
    start: usize,
    length: usize,
}

#[derive(Debug, PartialEq)]
pub struct SvoBuffer {
    bytes: Vec<u32>,
    free_ranges: Vec<Range>,
    octant_to_range: HashMap<OctantId, Range>,
}

impl SvoBuffer {
    fn new(initial_capacity: usize) -> SvoBuffer {
        let mut bytes = Vec::new();
        bytes.resize(initial_capacity, 0);

        let mut buffer = SvoBuffer {
            bytes,
            free_ranges: Vec::new(),
            octant_to_range: HashMap::new(),
        };
        if initial_capacity > 0 {
            buffer.free_ranges.push(Range { start: 0, length: initial_capacity })
        }
        buffer
    }

    fn insert(&mut self, id: OctantId, buf: &Vec<u32>) -> usize {
        self.remove(id);

        let mut ptr = self.bytes.len();
        let length = buf.len();

        if let Some(pos) = self.free_ranges.iter().position(|x| length <= x.length) {
            let range = &mut self.free_ranges[pos];

            ptr = range.start;

            if length < range.length {
                range.start += length;
                range.length -= length;
            } else {
                self.free_ranges.remove(pos);
            }

            unsafe {
                ptr::copy(buf.as_ptr(), self.bytes.as_mut_ptr().offset(ptr as isize), length);
            }
        } else {
            self.bytes.extend(buf);
        }

        self.octant_to_range.insert(id, Range { start: ptr, length });
        ptr
    }

    fn remove(&mut self, id: OctantId) {
        let range = self.octant_to_range.remove(&id);
        if range.is_none() {
            return;
        }

        let range = range.unwrap();
        self.free_ranges.push(range);
        self.free_ranges.sort_by(|lhs, rhs| lhs.start.cmp(&rhs.start));

        let mut i = 1;
        while i < self.free_ranges.len() {
            let rhs = self.free_ranges[i];
            let lhs = &mut self.free_ranges[i - 1];

            if (lhs.start + lhs.length) == rhs.start {
                lhs.length += rhs.length;
                self.free_ranges.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod svo_buffer_tests {
    use std::collections::HashMap;

    use crate::world::svo::{Range, SvoBuffer};

    #[test]
    fn buffer_insert_remove() {
        // create empty buffer with initial capacity
        let mut buffer = SvoBuffer::new(10);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            free_ranges: vec![Range { start: 0, length: 10 }],
            octant_to_range: Default::default(),
        });

        // insert data until initial capacity is full
        buffer.insert(1, &vec![0, 1, 2, 3, 4]);
        buffer.insert(2, &vec![5, 6]);
        buffer.insert(3, &vec![7, 8, 9]);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            free_ranges: vec![],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 3 }),
            ]),
        });

        // exceed initial capacity
        buffer.insert(4, &vec![10]);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            free_ranges: vec![],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 3 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // replace already existing data
        buffer.insert(3, &vec![11]);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![Range { start: 8, length: 2 }],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 1 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // remove existing data
        buffer.remove(2);
        buffer.remove(3);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![Range { start: 5, length: 5 }],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 5 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // insert into free space
        buffer.insert(5, &vec![12, 13, 14]);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![Range { start: 8, length: 2 }],
            octant_to_range: HashMap::from([
                (1, Range { start: 0, length: 5 }),
                (4, Range { start: 10, length: 1 }),
                (5, Range { start: 5, length: 3 }),
            ]),
        });

        // remove all and revert back to beginning
        buffer.remove(5);
        buffer.remove(4);
        buffer.remove(1);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: Default::default(),
        });
    }
}
