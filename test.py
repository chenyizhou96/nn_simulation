import python_bindings.TGSL as TGSL
import numpy as np
import sys, getopt
import config

v = []
TGSL.ReadDoubleVector(v, b'../tgsl/tools/geometry_processing/output/bc_frame_000000_python.bin')
#print(v)

TGSL.ReadIntVector(v, b'../tgsl/tools/geometry_processing/output/mesh_python.bin')
#print(v)

F = [1, 0, 0, 0, 1, 0, 0, 0, 1]
R = []
S = []
TGSL.QRDecomp(F, R, S)
print(R)
print(S)

test_config = config.TrainConfig(sys.argv)

