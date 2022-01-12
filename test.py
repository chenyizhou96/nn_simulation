import python_bindings.TGSL as TGSL
import numpy as np
import sys, getopt
import config
import torch
import unittest
import models
from efem import ElasticFEM
import train

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

x = []
TGSL.ReadDoubleVector(x, b'./test_pos.bin')
x = torch.FloatTensor(x)
mesh = []
TGSL.ReadIntVector(mesh, b'../tgsl/tools/geometry_processing/output/mesh_python.bin')
X_mesh = []
TGSL.ReadDoubleVector(X_mesh, b'../tgsl/tools/geometry_processing/output/x_mesh_python.bin')
X_mesh = torch.FloatTensor(X_mesh)
efem = ElasticFEM(mesh, X_mesh)
e = efem.potential_energy(models.linear_elasticity_psi, x,1000/2.6, 300/0.52)
print('%.10f'%e)


# class ElasticFEMTest(unittest.TestCase):
#   # def __init__(self):
#   #   self.mesh = []
#   #   TGSL.ReadIntVector(self.mesh, b'../tgsl/tools/geometry_processing/output/mesh_python.bin')
#   #   self.X_mesh = []
#   #   TGSL.ReadDoubleVector(self.X_mesh, b'../tgsl/tools/geometry_processing/output/x_mesh_python.bin')
#   #   self.X_mesh = torch.FloatTensor(self.X_mesh)
#   #   self.efem = ElasticFEM(self.mesh, self.X_mesh)


#   def energy_test(self):
#     x = []
#     TGSL.ReadDoubleVector(x, b'./test_pos.bin')
#     x = torch.FloatTensor(x)
#     e = self.efem.potential_energy(models.linear_elasticity_psi, x,1000/2.6, 300/0.52)
#     self.assertAlmostEqual(e,4.54e-3,3)

# if __name__ == '__main__':
#   unittest.main()

