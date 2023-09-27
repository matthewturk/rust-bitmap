use hilbert::FloatDataRange;
use ndarray_stats::QuantileExt;
use numpy::ndarray::Axis;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use roaring::RoaringTreemap;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_bitmap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ParticleTreemap>()?;
    Ok(())
}
#[pyclass]
struct ParticleTreemap {
    bitmap: RoaringTreemap,
}

#[pymethods]
impl ParticleTreemap {
    #[new]
    fn new() -> Self {
        ParticleTreemap {
            bitmap: RoaringTreemap::new(),
        }
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.bitmap.is_disjoint(&other.bitmap)
    }

    pub fn insert(&mut self, value: u64) -> bool {
        self.bitmap.insert(value)
    }

    pub fn insert_range(&mut self, lower: u64, upper: u64) -> u64 {
        self.bitmap.insert_range(lower..upper)
    }

    pub fn from_array(&mut self, arr: PyReadonlyArray1<u64>) -> bool {
        arr.as_array()
            .iter()
            .for_each(|x| _ = self.bitmap.insert(*x));
        true
    }

    pub fn len(&self) -> u64 {
        self.bitmap.len()
    }

    pub fn union_len(&self, other: &Self) -> u64 {
        self.bitmap.union_len(&other.bitmap)
    }
    pub fn intersection_len(&self, other: &Self) -> u64 {
        self.bitmap.intersection_len(&other.bitmap)
    }

    pub fn difference_len(&self, other: &Self) -> u64 {
        self.bitmap.difference_len(&other.bitmap)
    }

    pub fn serialized_size(&self) -> usize {
        self.bitmap.serialized_size()
    }

    pub fn num_partitions(&self) -> usize {
        self.bitmap.bitmaps().count()
    }

    pub fn partition_info(&self) {
        for (index, bm) in self.bitmap.bitmaps() {
            println!("Partition {} is of rank {}", index, bm.len());
        }
    }

    pub fn from_normalized_coordinates(&mut self, arr: PyReadonlyArray2<f64>, bits: usize) {
        // 2097152 is 21 bits
        println!(
            "Min and max {} {}",
            arr.as_array().min_skipnan(),
            arr.as_array().max_skipnan()
        );
        let range = FloatDataRange::new(
            *arr.as_array().min_skipnan(),
            *arr.as_array().max_skipnan(),
            (1 << 28) as f64,
        );
        for position in arr.as_array().axis_iter(Axis(0)) {
            let mut zindex = 0;
            position.iter().enumerate().for_each(|(i, x)| {
                zindex = zindex | (spread_64bits_by2(range.compress(*x, bits) as u64) << i);
            });
            self.insert(zindex);
        }
    }
}

pub fn encode_morton_64bit(ind: Vec<u32>) -> u64 {
    let mut result = 0u64;
    for (i, var) in ind.iter().enumerate() {
        result = result | spread_64bits_by2(*var as u64) << i;
    }
    return result;
}

pub fn spread_64bits_by2(x: u64) -> u64 {
    let mut y: u64 = x;
    // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---k jihg fedc ba98 7654 3210
    y = y & (0x00000000001FFFFFu64);
    // x = ---- ---- ---- ---- ---- ---k jihg fedc ba-- ---- ---- ---- ---- --98 7654 3210
    y = (y | (y << 20)) & (0x000001FFC00003FFu64);
    // x = ---- ---- ---- -kji hgf- ---- ---- -edc ba-- ---- ---- 9876 5--- ---- ---4 3210
    y = (y | (y << 10)) & (0x0007E007C00F801Fu64);
    // x = ---- ---- -kji h--- -gf- ---- -edc ---- ba-- ---- 987- ---6 5--- ---4 32-- --10
    y = (y | (y << 4)) & (0x00786070C0E181C3u64);
    // x = ---- ---k ji-- h--g --f- ---e d--c --b- -a-- --98 --7- -6-- 5--- -43- -2-- 1--0
    y = (y | (y << 2)) & (0x0199219243248649u64);
    // x = ---- -kj- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    y = (y | (y << 2)) & (0x0649249249249249u64);
    // x = ---k --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    y = (y | (y << 2)) & (0x1249249249249249u64);
    return y;
}

// // The Cython code reads as follows:
//
// #-----------------------------------------------------------------------------
// # 21 bits spread over 64 with 2 bits in between
// @cython.cdivision(True)
// @cython.boundscheck(False)
// @cython.wraparound(False)
// cdef inline np.uint64_t spread_64bits_by2(np.uint64_t x):
//     # This magic comes from http://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
//     # Only reversible up to 2097151
//     # Select highest 21 bits (Required to be reversible to 21st bit)
//     # x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---k jihg fedc ba98 7654 3210
//     x=(x&(<np.uint64_t>0x00000000001FFFFF))
//     # x = ---- ---- ---- ---- ---- ---k jihg fedc ba-- ---- ---- ---- ---- --98 7654 3210
//     x=(x|(x<<20))&(<np.uint64_t>0x000001FFC00003FF)
//     # x = ---- ---- ---- -kji hgf- ---- ---- -edc ba-- ---- ---- 9876 5--- ---- ---4 3210
//     x=(x|(x<<10))&(<np.uint64_t>0x0007E007C00F801F)
//     # x = ---- ---- -kji h--- -gf- ---- -edc ---- ba-- ---- 987- ---6 5--- ---4 32-- --10
//     x=(x|(x<<4))&(<np.uint64_t>0x00786070C0E181C3)
//     # x = ---- ---k ji-- h--g --f- ---e d--c --b- -a-- --98 --7- -6-- 5--- -43- -2-- 1--0
//     x=(x|(x<<2))&(<np.uint64_t>0x0199219243248649)
//     # x = ---- -kj- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
//     x=(x|(x<<2))&(<np.uint64_t>0x0649249249249249)
//     # x = ---k --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
//     x=(x|(x<<2))&(<np.uint64_t>0x1249249249249249)
//     return x
//
// @cython.cdivision(True)
// cdef inline np.uint64_t encode_morton_64bit(np.uint64_t x_ind, np.uint64_t y_ind, np.uint64_t z_ind):
//     cdef np.uint64_t mi
//     mi = 0
//     mi |= spread_64bits_by2(z_ind)<<ZSHIFT
//     mi |= spread_64bits_by2(y_ind)<<YSHIFT
//     mi |= spread_64bits_by2(x_ind)<<XSHIFT
//     return mi
// ZSHIFT is 0, YSHIFT is 1, XSHIFT is 2
