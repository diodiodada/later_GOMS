import h5py
filename = 'M.949-0.107283.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())

print(a_group_key)
