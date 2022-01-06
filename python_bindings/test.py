import TGSL
import numpy as np

v = []
filename = '../../tgsl/tools/geometry_processing/output/bc_frame_000000_python.bin'
TGSL.ReadDoubleVector(v, bytes(filename, 'UTF-8'))
print(v)

mesh = []
TGSL.ReadIntVector(mesh, b'../../tgsl/tools/geometry_processing/output/mesh_python.bin')

X_mesh = []
TGSL.ReadDoubleVector(X_mesh, b'../../tgsl/tools/geometry_processing/output/x_mesh_python.bin')
boundary_mesh = []
TGSL.ReadIntVector(boundary_mesh, b'../../tgsl/tools/geometry_processing/output/boundary_mesh_python.bin')

TGSL.WriteTrisFrame(X_mesh, int(len(X_mesh)/3), boundary_mesh, len(boundary_mesh), b'./test.geo')

print(boundary_mesh)