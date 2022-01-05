import python_bindings.hello as hello
import numpy as np

v = []
hello.ReadDoubleVector(v, b'../tgsl/tools/geometry_processing/output/bc_frame_000000_python.bin')
print(v)

hello.ReadIntVector(v, b'../tgsl/tools/geometry_processing/output/mesh_python.bin')
print(v)