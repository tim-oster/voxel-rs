use std::alloc::{Allocator, Global};
use std::ptr;

use rustc_hash::FxHashMap;

use crate::world::memory::{Pool, StatsAllocator};

pub type ChunkBufferPool<T, A = StatsAllocator> = Pool<ChunkBuffer<T, A>, A>;

/// `ChunkBuffer` abstracts the temporary storage used for serializing octants into the GPU format.
pub struct ChunkBuffer<T: Bits, A: Allocator = Global> {
    pub data: Vec<T, A>,
}

impl<T: Bits> ChunkBuffer<T> {
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T: Bits, A: Allocator> ChunkBuffer<T, A> {
    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self { data: Vec::with_capacity_in(capacity, alloc) }
    }

    pub fn reset(&mut self) {
        self.data.clear();
    }
}

// -------------------------------------------------------------------------------------------------

pub trait Bits: Clone + Copy {
    type T: Clone + Copy;

    const ZERO: Self::T;
    const BYTES: usize;
}

impl Bits for u8 {
    type T = u8;

    const ZERO: Self::T = 0u8;
    const BYTES: usize = 1;
}

impl Bits for u16 {
    type T = u16;

    const ZERO: Self::T = 0u16;
    const BYTES: usize = 2;
}

impl Bits for u32 {
    type T = u32;

    const ZERO: Self::T = 0u32;
    const BYTES: usize = 4;
}

// -------------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Range {
    pub start: usize,
    pub length: usize,
}

/// [`RangeBuffer`] allows for copying data to an internal buffer and keeping track of the range the data was copied to using
/// unique ids. Those ids can be used to remove data from the buffer again.
///
/// Removing data does not free up the already allocated memory but instead marks the range inside the buffer as free.
/// If two adjacent ranges are removed, they are merged into one. Inserting into the buffer will prioritize reusing
/// memory over appending to the end of the internal buffer.
#[derive(Debug)]
pub struct RangeBuffer<T, A: Allocator> {
    pub bytes: Vec<T, A>,
    pub free_ranges: Vec<Range>,
    pub updated_ranges: Vec<Range>,
    pub octant_to_range: FxHashMap<u64, Range>,
}

impl<T: PartialEq, A: Allocator> PartialEq for RangeBuffer<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.free_ranges == other.free_ranges
            && self.updated_ranges == other.updated_ranges
            && self.octant_to_range == other.octant_to_range
    }
}

impl<T: Bits<T=T>, A: Allocator> RangeBuffer<T, A> {
    pub fn with_capacity_in(initial_capacity: usize, alloc: A) -> Self {
        let mut bytes = Vec::with_capacity_in(initial_capacity, alloc);
        bytes.extend(std::iter::repeat(T::ZERO).take(initial_capacity));

        let mut buffer = Self {
            bytes,
            free_ranges: Vec::new(),
            updated_ranges: Vec::new(),
            octant_to_range: FxHashMap::default(),
        };
        if initial_capacity > 0 {
            buffer.free_ranges.push(Range { start: 0, length: initial_capacity });
        }
        buffer
    }

    pub fn clear(&mut self) {
        self.free_ranges.clear();
        self.free_ranges.push(Range { start: 0, length: self.bytes.capacity() });
        self.updated_ranges.clear();
        self.octant_to_range.clear();
    }

    /// Inserts by copying data from the given buffer to the first free range or to the end of the internal buffer.
    pub fn insert(&mut self, id: u64, buf: &ChunkBuffer<T>) -> usize {
        self.remove(id);

        let mut ptr = self.bytes.len();
        let length = buf.data.len();

        // try to find the first free range that is at least as long as the buffer
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
                ptr::copy(buf.data.as_ptr(), self.bytes.as_mut_ptr().add(ptr), length);
            }
        } else {
            // otherwise, extend at the end
            self.bytes.extend(buf.data.iter());
        }

        self.octant_to_range.insert(id, Range { start: ptr, length });

        self.updated_ranges.push(Range { start: ptr, length });
        Self::merge_ranges(&mut self.updated_ranges);

        ptr
    }

    /// Frees the corresponding range for the given id.
    pub fn remove(&mut self, id: u64) {
        let range = self.octant_to_range.remove(&id);
        if range.is_none() {
            return;
        }

        let range = range.unwrap();
        self.free_ranges.push(range);
        Self::merge_ranges(&mut self.free_ranges);
    }

    /// Orders all free ranges by start index and merges adjacent ranges into one.
    fn merge_ranges(ranges: &mut Vec<Range>) {
        // Unstable is fine here as no equivalent objects can exist. It should be slightly faster
        // and avoids heap allocations.
        ranges.sort_unstable_by(|lhs, rhs| lhs.start.cmp(&rhs.start));

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

    pub fn size_in_bytes(&self) -> usize {
        self.bytes.len() * T::BYTES
    }
}

#[cfg(test)]
mod range_buffer_tests {
    use std::alloc::Global;

    use rustc_hash::FxHashMap;

    use crate::world::hds::internal::{ChunkBuffer, RangeBuffer};

    /// Tests different insert & remove edge cases.
    #[test]
    fn buffer_insert_remove() {
        // create empty buffer with initial capacity
        let mut buffer = RangeBuffer::<u32, Global>::with_capacity_in(10, Global);

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            free_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 10 }],
            updated_ranges: vec![],
            octant_to_range: Default::default(),
        });

        // insert data until initial capacity is full
        buffer.insert(1, &ChunkBuffer { data: vec![0, 1, 2, 3, 4] });
        buffer.insert(2, &ChunkBuffer { data: vec![5, 6] });
        buffer.insert(3, &ChunkBuffer { data: vec![7, 8, 9] });

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            free_ranges: vec![],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 10 }],
            octant_to_range: FxHashMap::from_iter([
                (1, crate::world::hds::internal::Range { start: 0, length: 5 }),
                (2, crate::world::hds::internal::Range { start: 5, length: 2 }),
                (3, crate::world::hds::internal::Range { start: 7, length: 3 }),
            ]),
        });

        // exceed initial capacity
        buffer.insert(4, &ChunkBuffer { data: vec![10] });

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            free_ranges: vec![],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, crate::world::hds::internal::Range { start: 0, length: 5 }),
                (2, crate::world::hds::internal::Range { start: 5, length: 2 }),
                (3, crate::world::hds::internal::Range { start: 7, length: 3 }),
                (4, crate::world::hds::internal::Range { start: 10, length: 1 }),
            ]),
        });

        // replace already existing data
        buffer.insert(3, &ChunkBuffer { data: vec![11] });

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![crate::world::hds::internal::Range { start: 8, length: 2 }],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, crate::world::hds::internal::Range { start: 0, length: 5 }),
                (2, crate::world::hds::internal::Range { start: 5, length: 2 }),
                (3, crate::world::hds::internal::Range { start: 7, length: 1 }),
                (4, crate::world::hds::internal::Range { start: 10, length: 1 }),
            ]),
        });

        // remove existing data
        buffer.remove(2);
        buffer.remove(3);

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![crate::world::hds::internal::Range { start: 5, length: 5 }],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, crate::world::hds::internal::Range { start: 0, length: 5 }),
                (4, crate::world::hds::internal::Range { start: 10, length: 1 }),
            ]),
        });

        // insert into free space
        buffer.insert(5, &ChunkBuffer { data: vec![12, 13, 14] });

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![crate::world::hds::internal::Range { start: 8, length: 2 }],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, crate::world::hds::internal::Range { start: 0, length: 5 }),
                (4, crate::world::hds::internal::Range { start: 10, length: 1 }),
                (5, crate::world::hds::internal::Range { start: 5, length: 3 }),
            ]),
        });

        // remove all and revert back to beginning
        buffer.remove(5);
        buffer.remove(4);
        buffer.remove(1);

        assert_eq!(buffer, RangeBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            updated_ranges: vec![crate::world::hds::internal::Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::default(),
        });
    }

    /// Tests that range merging edge cases work properly.
    #[test]
    fn merge_ranges() {
        struct TestCase {
            name: &'static str,
            input: Vec<crate::world::hds::internal::Range>,
            expected: Vec<crate::world::hds::internal::Range>,
        }
        for case in vec![
            TestCase {
                name: "join adjacent ranges",
                input: vec![
                    crate::world::hds::internal::Range { start: 0, length: 1 },
                    crate::world::hds::internal::Range { start: 1, length: 1 },
                    crate::world::hds::internal::Range { start: 2, length: 1 },
                ],
                expected: vec![
                    crate::world::hds::internal::Range { start: 0, length: 3 },
                ],
            },
            TestCase {
                name: "ignore non-adjacent ranges",
                input: vec![
                    crate::world::hds::internal::Range { start: 0, length: 1 },
                    crate::world::hds::internal::Range { start: 2, length: 1 },
                ],
                expected: vec![
                    crate::world::hds::internal::Range { start: 0, length: 1 },
                    crate::world::hds::internal::Range { start: 2, length: 1 },
                ],
            },
            TestCase {
                name: "remove fully contained ranges",
                input: vec![
                    crate::world::hds::internal::Range { start: 0, length: 5 },
                    crate::world::hds::internal::Range { start: 3, length: 1 },
                ],
                expected: vec![
                    crate::world::hds::internal::Range { start: 0, length: 5 },
                ],
            },
            TestCase {
                name: "remove and extend contained ranges",
                input: vec![
                    crate::world::hds::internal::Range { start: 0, length: 5 },
                    crate::world::hds::internal::Range { start: 3, length: 5 },
                ],
                expected: vec![
                    crate::world::hds::internal::Range { start: 0, length: 8 },
                ],
            },
            TestCase {
                name: "works in inverse order",
                input: vec![
                    crate::world::hds::internal::Range { start: 3, length: 5 },
                    crate::world::hds::internal::Range { start: 0, length: 5 },
                ],
                expected: vec![
                    crate::world::hds::internal::Range { start: 0, length: 8 },
                ],
            },
        ] {
            let mut input = case.input;
            RangeBuffer::<u32, Global>::merge_ranges(&mut input);
            assert_eq!(input, case.expected, "{}", case.name);
        }
    }
}
