use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
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
}
