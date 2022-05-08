use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::mem::swap;
use std::ptr::copy;

use crate::chunk::BlockId;
use crate::world::octree::{Octant, OctantId, Octree, Position};

// TODO refactor whole implementation

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Range {
    pub start: usize,
    pub length: usize,

    ptr: MissingPointer, // TODO improve
}

#[derive(Debug, PartialEq)]
pub struct SerializedSvo {
    pub header_mask: u16,
    pub depth: u32,
    pub buffer: SvoBuffer,
    pub changed_octants: HashSet<OctantId>,
}

#[derive(Debug, PartialEq)]
pub struct SvoBuffer {
    root_mask: u16,
    depth: u32,

    pub bytes: Vec<u32>,
    free_ranges: Vec<Range>,
    pub octant_to_range: HashMap<OctantId, Range>,

    // TODO should this be stored in the buffer?
    rebuild_targets: HashMap<OctantId, MissingPointer>,
}

impl SvoBuffer {
    fn insert(&mut self, buf: Vec<u32>) -> usize {
        let mut ptr = self.bytes.len();

        if let Some(pos) = self.free_ranges.iter().position(|x| buf.len() <= x.length) {
            let range = &mut self.free_ranges[pos];

            ptr = range.start;

            if buf.len() < range.length {
                range.start += buf.len();
                range.length -= buf.len();
            } else {
                self.free_ranges.remove(pos);
            }

            unsafe {
                copy(buf.as_ptr(), self.bytes.as_mut_ptr().offset(ptr as isize), buf.len());
            }
        } else {
            self.bytes.extend(buf);
        }

        ptr
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum OctantChange {
    Add(OctantId),
    Remove(OctantId),
}

pub struct Svo<T: SvoSerializable> {
    octree: Octree<T>,
    change_set: HashSet<OctantChange>,
}

impl<T: SvoSerializable> Svo<T> {
    pub fn new() -> Svo<T> {
        Svo {
            octree: Octree::new(),
            change_set: HashSet::new(),
        }
    }
}

impl<T: SvoSerializable> Svo<T> {
    pub fn set(&mut self, pos: Position, leaf: Option<T>) {
        if let Some(leaf) = leaf {
            let octant_id = self.octree.add_leaf(pos, leaf);
            self.change_set.insert(OctantChange::Add(octant_id));
        } else {
            if let Some(id) = self.octree.remove_leaf(pos) {
                self.change_set.insert(OctantChange::Remove(id));
            }
        }
    }

    pub fn serialize(&mut self) -> SerializedSvo {
        self.change_set.clear(); // TODO use internal mutability for this?

        let dst = SerializedSvo {
            header_mask: 0,
            depth: 0,
            buffer: SvoBuffer {
                // TODO figure out correct sizes
                root_mask: 0,
                depth: 0,
                bytes: Vec::new(),
                free_ranges: vec![],
                octant_to_range: Default::default(),
                rebuild_targets: Default::default(),
            },
            changed_octants: HashSet::new(),
        };
        self.serialize_to(dst)
    }

    pub fn serialize_to(&self, dst: SerializedSvo) -> SerializedSvo {
        if self.octree.root.is_none() {
            return dst;
        }

        let mut dst = dst;
        if dst.buffer.bytes.is_empty() {
            // if buffer is empty, insert empty preamble that contains space for the "hull" header
            // mask and the initial absolute pointer to the root of the octree
            dst.buffer.bytes.reserve(5);
            dst.buffer.bytes.extend(std::iter::repeat(0).take(5));
        }

        let (header_mask, depth, _, changed_octants) = serialize_octree(&self.octree, &mut dst.buffer);
        dst.buffer.bytes[0] = header_mask as u32;
        dst.buffer.bytes[4] = dst.buffer.octant_to_range.get(&self.octree.root.unwrap()).unwrap().start as u32;

        dst.header_mask = header_mask;
        dst.depth = depth;
        dst.changed_octants = changed_octants;
        dst
    }

    pub fn serialize_delta(&mut self, previous: SerializedSvo) -> SerializedSvo {
        if self.change_set.is_empty() {
            return previous;
        }

        let changes = self.change_set.drain().collect::<Vec<OctantChange>>();
        let mut changed_ids = Vec::new();

        let mut rebuild_root = false;
        for change in changes {
            match change {
                OctantChange::Add(id) => {
                    changed_ids.push(id);
                    if !previous.buffer.octant_to_range.contains_key(&id) {
                        rebuild_root = true;
                        break;
                    }
                }
                OctantChange::Remove(id) => {
                    changed_ids.push(id);
                    rebuild_root = true;
                    break;
                }
            }
        }
        if rebuild_root {
            changed_ids.push(self.octree.root.unwrap());
        }

        let mut previous = previous;
        for change in changed_ids {
            if let Some(range) = previous.buffer.octant_to_range.remove(&change) {
                previous.buffer.rebuild_targets.insert(change, range.ptr);
                previous.buffer.free_ranges.push(range);
            }
        }

        // Order by start pointer to allow merging adjacent ranges.
        previous.buffer.free_ranges.sort_by(|lhs, rhs| lhs.start.cmp(&rhs.start));

        // Merge adjacent ranges into one bigger range.
        let mut i = 1;
        while i < previous.buffer.free_ranges.len() {
            let rhs = previous.buffer.free_ranges[i];
            let lhs = &mut previous.buffer.free_ranges[i - 1];

            if (lhs.start + lhs.length) == rhs.start {
                lhs.length += rhs.length;
                previous.buffer.free_ranges.remove(i);
            } else {
                i += 1;
            }
        }

        // Order by length so that smallest ranges come first. Allows buffer insertion to always
        // pick the shortest range first by simply iterating through the free ranges vector.
        previous.buffer.free_ranges.sort_by(|lhs, rhs| lhs.length.cmp(&rhs.length));

        self.serialize_to(previous)
    }
}

fn serialize_octree<T: SvoSerializable>(octree: &Octree<T>, dst: &mut SvoBuffer) -> (u16, u32, Vec<MissingPointer>, HashSet<OctantId>) {
    let root_id = octree.root.unwrap();
    let root = &octree.octants[root_id];

    let mut changed_octants = HashSet::new();

    if !dst.octant_to_range.contains_key(&root_id) {
        let mut staging_buf = Vec::new(); // TODO use memory pool?
        let (child_mask, leaf_mask, ptrs) = serialize_octant(octree, root, &mut staging_buf);
        let len = staging_buf.len();
        let ptr = dst.insert(staging_buf);

        dst.root_mask = ((child_mask as u16) << 8) | leaf_mask as u16;
        dst.octant_to_range.insert(root_id, Range {
            start: ptr,
            length: len,
            ptr: MissingPointer {
                octant: 0, // TODO not correct, is it?
                child_idx: 0,
                buffer_index: ptr,
            },
        });
        changed_octants.insert(root_id);

        dst.depth = octree.depth;
        for missing_ptr in ptrs {
            let mut missing_ptr = missing_ptr;
            missing_ptr.buffer_index += ptr;

            if let Some(range) = dst.octant_to_range.get(&missing_ptr.octant) {
                dst.bytes[missing_ptr.buffer_index] = range.start as u32;
                continue;
            }

            dst.rebuild_targets.insert(missing_ptr.octant, missing_ptr);
        }
    }

    let mut rebuild_targets = dst.rebuild_targets.drain()
        .map(|x| x.1)
        .collect::<Vec<MissingPointer>>();
    rebuild_targets.sort_by(|lhs, rhs| lhs.octant.cmp(&rhs.octant));
    for missing_ptr in &rebuild_targets {
        let octant = &octree.octants[missing_ptr.octant as usize];

        let mut staging_buf = Vec::new(); // TODO use memory pool?
        let (depth, child_ptrs) = octant.content.as_ref().unwrap().serialize_to(&missing_ptr, dst, &mut staging_buf);
        let len = staging_buf.len();
        let ptr = dst.insert(staging_buf);

        for child_ptr in &child_ptrs {
            dst.bytes[ptr + child_ptr.buffer_index] += ptr as u32;
        }

        dst.bytes[missing_ptr.buffer_index] = ptr as u32;
        dst.depth = max(dst.depth, octree.depth + depth);
        dst.octant_to_range.insert(missing_ptr.octant, Range {
            start: ptr,
            length: len,
            ptr: *missing_ptr,
        });

        changed_octants.insert(missing_ptr.octant);
        changed_octants.insert(root_id);
    }

    // TODO still return?
    (dst.root_mask, dst.depth, rebuild_targets, changed_octants)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MissingPointer {
    pub octant: OctantId,
    pub child_idx: usize,
    pub buffer_index: usize,
}

pub trait SvoSerializable {
    // TODO is is_nested clean?
    fn is_nested(&self) -> bool;
    fn serialize_to(&self, at: &MissingPointer, dst: &mut SvoBuffer, staging_buffer: &mut Vec<u32>) -> (u32, Vec<MissingPointer>);
}

impl<T: SvoSerializable> SvoSerializable for Octree<T> {
    fn is_nested(&self) -> bool {
        true
    }

    fn serialize_to(&self, at: &MissingPointer, dst: &mut SvoBuffer, staging_buffer: &mut Vec<u32>) -> (u32, Vec<MissingPointer>) {
        // TODO do not track changed octants here

        let mut wrapped_dst = SvoBuffer {
            root_mask: 0,
            depth: 0,
            bytes: Vec::new(),
            free_ranges: vec![],
            octant_to_range: Default::default(),
            rebuild_targets: Default::default(),
        };
        swap(staging_buffer, &mut wrapped_dst.bytes);
        let (header_mask, depth, ptrs, _) = serialize_octree(self, &mut wrapped_dst);
        swap(staging_buffer, &mut wrapped_dst.bytes);

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
        dst.bytes[header_pos] = (dst.bytes[header_pos] & mask) | header_mask;

        (depth, ptrs)
    }
}

impl SvoSerializable for BlockId {
    fn is_nested(&self) -> bool {
        false
    }

    fn serialize_to(&self, _: &MissingPointer, _: &mut SvoBuffer, staging_buffer: &mut Vec<u32>) -> (u32, Vec<MissingPointer>) {
        // TODO set inplace instead of appending?
        staging_buffer.push(*self as u32);
        (0, Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::chunk::BlockId;
    use crate::world::octree::{Octree, Position};
    use crate::world::svo::{MissingPointer, Range, SerializedSvo, Svo, SvoBuffer};

    #[test]
    fn svo_serialize() {
        let mut svo = Svo::new();
        svo.set(Position(0, 0, 0), Some(100 as BlockId));
        svo.set(Position(1, 1, 1), Some(200 as BlockId));
        svo.octree.expand_to(5);
        svo.octree.compact();

        assert_eq!(svo.serialize(), SerializedSvo {
            header_mask: 1 << 8,
            depth: 5,
            buffer: SvoBuffer {
                root_mask: 1 << 8,
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
                free_ranges: vec![],
                octant_to_range: HashMap::from([
                    (1, Range { start: 65, length: 1, ptr: MissingPointer { octant: 1, child_idx: 0, buffer_index: 57 } }),
                    (2, Range { start: 66, length: 1, ptr: MissingPointer { octant: 2, child_idx: 7, buffer_index: 64 } }),
                    (6, Range { start: 5, length: 60, ptr: MissingPointer { octant: 0, child_idx: 0, buffer_index: 5 } }),
                ]),
                rebuild_targets: Default::default(),
            },
            changed_octants: HashSet::from([1, 2, 6]),
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
        assert_eq!(svo, SerializedSvo {
            header_mask: 2 << 8,
            depth: 6,
            buffer: SvoBuffer {
                root_mask: 2 << 8,
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
                free_ranges: vec![],
                octant_to_range: HashMap::from([
                    (0, Range { start: 5, length: 12, ptr: MissingPointer { octant: 0, child_idx: 0, buffer_index: 5 } }),
                    (1, Range { start: 17, length: 159, ptr: MissingPointer { octant: 1, child_idx: 1, buffer_index: 10 } }),
                ]),
                rebuild_targets: Default::default(),
            },
            changed_octants: HashSet::from([0, 1]),
        });
    }
}
