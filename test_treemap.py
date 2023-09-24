import time
import rust_bitmap
import numpy as np

(arr,) = np.where(np.random.random(256**3) > 0.5)
arr = arr.astype("u8")
c = arr.size

N = 10
t1 = time.time()
for i in range(N):
    b1 = rust_bitmap.ParticleTreemap()
    b1.from_array(arr)
t2 = time.time()
print(f"Time 1: {(t2-t1)/N:0.2f}")
print(b1.len())

t1 = time.time()
for i in range(N):
    b2 = rust_bitmap.ParticleTreemap()
    b2.insert_range(0, c)
t2 = time.time()
print(b2.len())

print(f"Time 2: {(t2-t1)/N:0.2f}")
