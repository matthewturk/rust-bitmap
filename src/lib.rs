use numpy::ndarray::{ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD};
use numpy::{PyArray1, PyReadonlyArray1};
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

    pub fn from_array(&mut self, arr: PyReadonlyArray1<u64>) -> bool {
        arr.as_array()
            .iter()
            .for_each(|x| _ = self.bitmap.insert(*x));
        true
    }

    pub fn len(&self) -> u64 {
        self.bitmap.len()
    }
}
