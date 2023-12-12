#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::ptr;

use crate::world::chunk::{BlockId, ChunkPos};
use crate::world::octree::{Octant, OctantId, Octree, Position};
use crate::world::world::BorrowedChunk;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum OctantChange {
    Add(OctantId),
    Remove(OctantId),
}

pub trait SvoSerializable {
    fn serialize(&self, dst: &mut Vec<u32>, lod: u8) -> SerializationResult;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SerializationResult {
    pub child_mask: u16,
    pub leaf_mask: u16,
    pub depth: u32,
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
    const PREAMBLE_LENGTH: u32 = 5;

    pub fn new() -> Svo<T> {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Svo<T> {
        Svo {
            octree: Octree::new(),
            change_set: HashSet::new(),
            buffer: SvoBuffer::new(capacity),
            octant_info: HashMap::new(),
            root_octant_info: None,
        }
    }

    pub fn clear(&mut self) {
        self.octree.reset();
        self.change_set.clear();
        self.buffer.clear();
        self.octant_info.clear();
        self.root_octant_info = None;
    }

    pub fn set(&mut self, pos: Position, leaf: Option<T>) -> Option<OctantId> {
        if let Some(leaf) = leaf {
            let id = self.octree.add_leaf(pos, leaf);
            self.change_set.insert(OctantChange::Add(id));
            return Some(id);
        }
        if let Some(id) = self.octree.remove_leaf(pos) {
            self.change_set.insert(OctantChange::Remove(id));
            return Some(id);
        }
        return None;
    }

    /// replace sets the leaf octant at the given position and returns the previously present leaf
    /// octant id.
    pub fn replace(&mut self, pos: Position, leaf: OctantId) -> Option<OctantId> {
        // TODO should this be marked in change_set?
        self.octree.replace_leaf(pos, leaf)
    }

    pub fn remove_octant(&mut self, octant_id: OctantId) {
        // TODO should this be marked in change_set?
        self.octree.delete_octant(octant_id);
    }

    // TODO should this be the normal get operation and the current one get_at_position instead?
    pub fn get_at_pos(&self, pos: Position) -> Option<&T> {
        self.octree.get_leaf(pos)
    }

    pub fn octant_count(&self) -> usize {
        self.octree.octant_count()
    }

    pub fn serialize(&mut self) {
        if self.octree.root.is_none() {
            return;
        }

        // rebuild & remove all changed leaf octants
        let mut octant_buffer = Vec::with_capacity(56172); // size of a full chunk
        for change in self.change_set.drain() {
            match change {
                OctantChange::Add(id) => {
                    let octant = self.octree.octants[id].content.as_ref().unwrap();
                    let result = octant.serialize(&mut octant_buffer, 0);
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
        let result = self.serialize_root(&mut octant_buffer);
        let offset = self.buffer.insert(usize::MAX, &octant_buffer);
        self.root_octant_info = Some(OctantInfo { buf_offset: offset, serialization: result });
    }

    fn serialize_root(&self, dst: &mut Vec<u32>) -> SerializationResult {
        let root_id = self.octree.root.unwrap();
        let root = &self.octree.octants[root_id];

        serialize_octant(&self.octree, root, dst, 0, &|params| {
            let info = self.octant_info.get(&params.id);
            if info.is_none() {
                return;
            }
            let info = info.unwrap();

            let mut mask = ((info.serialization.child_mask as u32) << 8) | info.serialization.leaf_mask as u32;
            if (params.idx % 2) != 0 {
                mask <<= 16;
            }

            params.dst[(params.idx / 2) as usize] |= mask;
            params.dst[4 + params.idx] = info.buf_offset as u32 + Self::PREAMBLE_LENGTH;
            params.result.depth = params.result.depth.max(info.serialization.depth + 1);
        })
    }

    pub fn size_in_bytes(&self) -> usize {
        self.buffer.bytes.len() * 4
    }

    pub fn depth(&self) -> u32 {
        if self.root_octant_info.is_none() {
            return 0;
        }
        self.root_octant_info.unwrap().serialization.depth
    }

    pub unsafe fn write_to(&self, dst: *mut u32) -> usize {
        if self.root_octant_info.is_none() {
            return 0;
        }

        let start = dst as usize;
        let info = self.root_octant_info.unwrap();
        let dst = Self::write_preamble(info, dst);

        let len = self.buffer.bytes.len();
        ptr::copy(self.buffer.bytes.as_ptr(), dst, len);

        let dst = dst.offset(len as isize);
        ((dst as usize) - start) / 4
    }

    // TODO write test
    /// Writes all changes after the last reset to the given buffer. The implementation assumes
    /// that the buffer passed has the same size and contains the same data as buffers in past
    /// writes. Changes are **NOT** reset, a call to `reset_changes` is necessary to do so.
    pub unsafe fn write_changes_to(&self, dst: *mut u32) {
        if self.root_octant_info.is_none() {
            return;
        }

        let info = self.root_octant_info.unwrap();
        let dst = Self::write_preamble(info, dst);

        for changed_range in &self.buffer.updated_ranges {
            let offset = changed_range.start as isize;
            let src = self.buffer.bytes.as_ptr().offset(offset);
            ptr::copy(src, dst.offset(offset), changed_range.length);
        }
    }

    pub fn reset_changes(&mut self) {
        self.buffer.updated_ranges.clear();
    }

    unsafe fn write_preamble(info: OctantInfo, dst: *mut u32) -> *mut u32 {
        dst.offset(0).write((info.serialization.child_mask as u32) << 8);
        dst.offset(1).write(0);
        dst.offset(2).write(0);
        dst.offset(3).write(0);
        dst.offset(4).write(info.buf_offset as u32 + Self::PREAMBLE_LENGTH);
        dst.offset(Self::PREAMBLE_LENGTH as isize)
    }
}

impl SvoSerializable for SerializedChunk {
    fn serialize(&self, dst: &mut Vec<u32>, _lod: u8) -> SerializationResult {
        if self.buffer.is_some() {
            // TODO is this fast enough?
            // TODO how to free memory after swap
            dst.extend(self.buffer.as_ref().unwrap());
        }
        self.result
    }
}

impl SvoSerializable for Octree<BlockId> {
    fn serialize(&self, dst: &mut Vec<u32>, lod: u8) -> SerializationResult {
        if self.root.is_none() {
            return SerializationResult { child_mask: 0, leaf_mask: 0, depth: 0 };
        }

        let root_id = self.root.unwrap();
        let root = &self.octants[root_id];
        serialize_octant(self, root, dst, lod, &|params| {
            params.result.leaf_mask |= 1 << params.idx;
            params.dst[4 + params.idx] = *params.content as u32;
            params.result.depth = 1;
        })
    }
}

pub struct SerializedChunk {
    pub pos: ChunkPos,
    pub lod: u8,
    pub borrowed_chunk: Option<BorrowedChunk>,
    buffer: Option<Vec<u32>>,
    result: SerializationResult,
}

impl SerializedChunk {
    pub fn new(chunk: BorrowedChunk) -> SerializedChunk {
        let pos = chunk.pos;
        let lod = chunk.lod;

        // TODO use memory pool
        let mut buffer = Vec::with_capacity(56172); // size of a full chunk
        let result = chunk.storage.as_ref().unwrap().serialize(&mut buffer, chunk.lod);
        if result.depth > 0 {
            return SerializedChunk { pos, lod, borrowed_chunk: Some(chunk), buffer: Some(buffer), result };
        }
        SerializedChunk { pos, lod, borrowed_chunk: Some(chunk), buffer: None, result }
    }
}

struct ChildEncodeParams<'a, T> {
    id: OctantId,
    idx: usize,
    result: &'a mut SerializationResult,
    dst: &'a mut [u32],
    content: &'a T,
}

fn serialize_octant<T, F>(octree: &Octree<T>, octant: &Octant<T>, dst: &mut Vec<u32>, lod: u8, child_encoder: &F) -> SerializationResult
    where F: Fn(ChildEncodeParams<T>) {
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
        let child = &octree.octants[child_id];

        if child.content.is_some() || lod == 1 {
            let mut content = child.content.as_ref();
            if content.is_none() {
                content = breadth_first(&octree, &child);
            }
            if content.is_none() {
                continue;
            }

            let content = content.unwrap();
            child_encoder(ChildEncodeParams {
                id: child_id,
                idx,
                result: &mut result,
                dst: &mut dst[start_offset..],
                content,
            });
        } else {
            let child_lod = if lod > 0 { lod - 1 } else { 0 };
            let child_offset = (dst.len() - start_offset) as u32;
            let child_result = serialize_octant(octree, &octree.octants[child_id], dst, child_lod, child_encoder);

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

fn breadth_first<'a, T>(octree: &'a Octree<T>, parent: &'a Octant<T>) -> Option<&'a T> {
    for child in parent.children.iter() {
        if child.is_none() {
            continue;
        }
        let child = &octree.octants[child.unwrap()];
        if child.content.is_some() {
            return child.content.as_ref();
        }
    }
    for child in parent.children.iter() {
        if child.is_none() {
            continue;
        }
        let child = &octree.octants[child.unwrap()];
        let result = breadth_first(octree, child);
        if result.is_some() {
            return result;
        }
    }
    return None;
}

#[cfg(test)]
mod svo_tests {
    use std::collections::HashMap;

    use crate::world::chunk::BlockId;
    use crate::world::octree::{Octree, Position};
    use crate::world::svo::{OctantInfo, Range, SerializationResult, Svo, SvoBuffer};

    #[test]
    fn svo_serialize() {
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

        let preamble_length = 5;
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
            0, 0 + preamble_length, 0, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(svo.buffer, SvoBuffer {
            bytes: expected.clone(),
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 168 }],
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
                // preamble
                2 << 8,
                0,
                0,
                0,
                156 + preamble_length,
            ],
            expected,
        ].concat());
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Range {
    pub start: usize,
    pub length: usize,
}

#[derive(Debug, PartialEq)]
pub struct SvoBuffer {
    bytes: Vec<u32>,
    free_ranges: Vec<Range>,
    updated_ranges: Vec<Range>,
    octant_to_range: HashMap<OctantId, Range>,
}

impl SvoBuffer {
    fn new(initial_capacity: usize) -> SvoBuffer {
        let mut bytes = Vec::new();
        bytes.resize(initial_capacity, 0);

        let mut buffer = SvoBuffer {
            bytes,
            free_ranges: Vec::new(),
            updated_ranges: Vec::new(),
            octant_to_range: HashMap::new(),
        };
        if initial_capacity > 0 {
            buffer.free_ranges.push(Range { start: 0, length: initial_capacity })
        }
        buffer
    }

    fn clear(&mut self) {
        self.free_ranges.clear();
        self.free_ranges.push(Range { start: 0, length: self.bytes.capacity() });
        self.updated_ranges.clear();
        self.octant_to_range.clear();
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

        self.updated_ranges.push(Range { start: ptr, length });
        Self::merge_ranges(&mut self.updated_ranges);

        ptr
    }

    fn remove(&mut self, id: OctantId) {
        let range = self.octant_to_range.remove(&id);
        if range.is_none() {
            return;
        }

        let range = range.unwrap();
        self.free_ranges.push(range);
        Self::merge_ranges(&mut self.free_ranges);
    }

    fn merge_ranges(ranges: &mut Vec<Range>) {
        ranges.sort_by(|lhs, rhs| lhs.start.cmp(&rhs.start));

        let mut i = 1;
        while i < ranges.len() {
            let rhs = ranges[i];
            let lhs = &mut ranges[i - 1];

            if rhs.start <= lhs.start + lhs.length {
                let diff = lhs.start + lhs.length - rhs.start;
                if rhs.length > diff {
                    lhs.length += rhs.length - diff;
                }
                ranges.remove(i);
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
            updated_ranges: vec![],
            octant_to_range: Default::default(),
        });

        // insert data until initial capacity is full
        buffer.insert(1, &vec![0, 1, 2, 3, 4]);
        buffer.insert(2, &vec![5, 6]);
        buffer.insert(3, &vec![7, 8, 9]);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 10 }],
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
            updated_ranges: vec![Range { start: 0, length: 11 }],
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
            updated_ranges: vec![Range { start: 0, length: 11 }],
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
            updated_ranges: vec![Range { start: 0, length: 11 }],
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
            updated_ranges: vec![Range { start: 0, length: 11 }],
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
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: Default::default(),
        });
    }

    #[test]
    fn merge_ranges() {
        struct TestCase {
            name: &'static str,
            input: Vec<Range>,
            expected: Vec<Range>,
        }
        for case in vec![
            TestCase {
                name: "join adjacent ranges",
                input: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 1, length: 1 },
                    Range { start: 2, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 3 },
                ],
            },
            TestCase {
                name: "ignore non-adjacent ranges",
                input: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 2, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 2, length: 1 },
                ],
            },
            TestCase {
                name: "remove fully contained ranges",
                input: vec![
                    Range { start: 0, length: 5 },
                    Range { start: 3, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 5 },
                ],
            },
            TestCase {
                name: "remove and extend contained ranges",
                input: vec![
                    Range { start: 0, length: 5 },
                    Range { start: 3, length: 5 },
                ],
                expected: vec![
                    Range { start: 0, length: 8 },
                ],
            },
            TestCase {
                name: "works in inverse order",
                input: vec![
                    Range { start: 3, length: 5 },
                    Range { start: 0, length: 5 },
                ],
                expected: vec![
                    Range { start: 0, length: 8 },
                ],
            },
        ] {
            let mut input = case.input;
            SvoBuffer::merge_ranges(&mut input);
            assert_eq!(input, case.expected, "{}", case.name);
        }
    }
}
