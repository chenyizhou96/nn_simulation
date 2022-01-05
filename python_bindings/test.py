import TGSL
import numpy as np

v = []
filename = '../../tgsl/tools/geometry_processing/output/bc_frame_000000_python.bin'
TGSL.ReadDoubleVector(v, bytes(filename, 'UTF-8'))
print(v)

TGSL.ReadIntVector(v, b'../../tgsl/tools/geometry_processing/output/mesh_python.bin')
print(v)