use crate::storage::octree::{Octant, Octree, Position};

pub struct Svo<T: SvoSerializable> {
    octree: Octree<T>,
    // TODO cache which octants have changed
}

impl<T: SvoSerializable> Svo<T> {
    pub fn set(&mut self, pos: Position, leaf: Option<T>) {}
    pub fn get(&self, pos: Position) -> Option<T> { None }
    pub fn serialize(&self) -> SvoBuffer {
        SvoBuffer {
            header_mask: 0,
            depth: 0,
            bytes: vec![],
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SvoBuffer {
    pub header_mask: u16,
    pub depth: u32,
    pub bytes: Vec<u32>,
}

pub trait SvoSerializable {
    fn serialize(&self) -> SvoBuffer;
}

impl<T: Copy> SvoSerializable for Octree<T> where u32: From<T> {
    fn serialize(&self) -> SvoBuffer {
        if self.root.is_none() {
            return SvoBuffer { header_mask: 0, depth: 0, bytes: vec![] };
        }

        let root_id = self.root.unwrap();

        // TODO figure out correct size
        let mut bytes = Vec::new();
        let (header_mask, _) = serialize_octant(self, &self.octants[root_id], &mut bytes);
        let header_mask = (header_mask as u16) << 8; // convert from leaf to child mask

        SvoBuffer {
            header_mask,
            depth: self.depth,
            bytes,
        }
    }
}

fn serialize_octant<T: Copy>(octree: &Octree<T>, octant: &Octant<T>, dst: &mut Vec<u32>) -> (u32, u32) where u32: From<T> {
    let start_offset = dst.len();

    dst.reserve(12);
    dst.extend(std::iter::repeat(0).take(12));

    let mut child_mask = 0u32;
    let mut leaf_mask = 0u32;

    for (idx, child) in octant.children.iter().enumerate() {
        if child.is_none() {
            continue;
        }

        child_mask |= 1 << idx;

        let child_id = child.unwrap();
        let child = &octree.octants[child_id];

        if let Some(content) = child.content {
            leaf_mask |= 1 << idx;
            dst[start_offset + 4 + idx] = content.into();
        } else {
            let (child_mask, leaf_mask) = serialize_octant(octree, &octree.octants[child_id], dst);

            let mut mask = ((child_mask as u32) << 8) | leaf_mask as u32;
            if (idx % 2) != 0 {
                mask <<= 16;
            }
            dst[start_offset + (idx / 2) as usize] |= mask;

            dst[start_offset + 4 + idx] = 12 - 4 - idx as u32; // offset from pointer to end of octant block
            dst[start_offset + 4 + idx] |= 1 << 31; // flag as relative pointer
        }
    }

    (child_mask, leaf_mask)
}


#[cfg(test)]
mod tests {
    use crate::chunk::BlockId;
    use crate::storage::octree::{Octree, Position};
    use crate::storage::svo_new::{SvoBuffer, SvoSerializable};

    #[test]
    fn octree_serialize() {
        let mut octree = Octree::new();
        octree.add_leaf(Position(0, 0, 0), 100 as BlockId);
        octree.add_leaf(Position(1, 1, 1), 200 as BlockId);
        octree.expand_to(5);

        let svo = octree.serialize();
        assert_eq!(svo, SvoBuffer {
            header_mask: 1 << 8,
            depth: 5,
            bytes: vec![
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
                100, 0, 0, 0,
                0, 0, 0, 200,
            ],
        });
    }

    // TODO fix or remove?
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
