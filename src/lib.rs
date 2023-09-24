use hilbert::{FloatDataRange, Point};
use num_traits::cast::ToPrimitive;
use numpy::ndarray::{ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use roaring::{RoaringBitmap, RoaringTreemap};

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

    pub fn from_normalized_coordinates(&mut self, arr: PyReadonlyArray2<f64>) {
        // 2097152 is 21 bits
        let range = FloatDataRange::new(0.0, 1.0, 2097152.0);
        let mut coordinates = Vec::new();
        let mut next_id = 0;
        for position in arr.as_array().outer_iter() {
            position.for_each(|x| coordinates.push(range.compress(*x, 21)));
            match Point::new(next_id, &coordinates)
                .hilbert_transform(21)
                .to_u64()
            {
                Some(val) => self.insert(val),
                None => continue,
            };
            next_id += 1;
        }
    }
}
