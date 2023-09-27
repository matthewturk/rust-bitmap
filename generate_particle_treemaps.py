import rust_bitmap
import h5py

bitmaps = {}
with h5py.File("TNGHalo/halo_59.hdf5", "r") as f:
    boxsize = f["/Header"].attrs["BoxSize"]
    for ptype in [0, 1, 4]:
        print(ptype)
        bitmaps[ptype] = b = rust_bitmap.ParticleTreemap()
        c = f[f"/PartType{ptype}/Coordinates"][()] / boxsize
        print(c.max())
        print(f"Adding {c.shape}")
        b.from_normalized_coordinates(c, 8)
        print(f"Length {b.len()}")
f = h5py.File("TNGHalo/halo_59.hdf5", "r")


print(bitmaps[0].intersection_len(bitmaps[1]))
print(bitmaps[0].intersection_len(bitmaps[4]))
print(bitmaps[1].intersection_len(bitmaps[4]))

for k, b in bitmaps.items():
    print(f"{ptype=} {b.len()=} {b.serialized_size()=} {b.num_partitions()=}")
