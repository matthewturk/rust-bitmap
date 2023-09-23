import rust_bitmap
import numpy as np

b1 = rust_bitmap.ParticleTreemap()
b2 = rust_bitmap.ParticleTreemap()

(arr,) = np.where(np.random.random(128**3) > 0.5)

b1.from_array(arr.astype("uint64"))
print(b1.len())
print(arr.size)
