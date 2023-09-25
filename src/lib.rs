use hilbert::{FloatDataRange, Point};
use num_traits::cast::ToPrimitive;
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
        let range = FloatDataRange::new(0.0, 1.0, (1 << 21) as f64);
        let mut coordinates = Vec::new();
        let mut next_id = 0;
        for (i, position) in arr.as_array().axis_iter(Axis(0)).enumerate() {
            coordinates.clear();
            position.for_each(|x| coordinates.push(range.compress(*x, bits)));
            if i % 10000 == 0 {
                println!("Adding ... {} with total length {}", i, self.len())
            }
            match Point::new(next_id, &coordinates)
                .hilbert_transform(bits)
                .to_u64()
            {
                Some(val) => self.insert(val),
                None => continue,
            };
            next_id += 1;
        }
    }
}
