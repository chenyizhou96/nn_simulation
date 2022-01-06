# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string

#include "../../tgsl/library/io/BinaryIO.h"

cdef extern from "<array>" namespace "std" nogil:
  cdef cppclass Particle "std::array<double, 3>":
    Particle() except+
    int& operator[](size_t)

cdef extern from *:
  ctypedef int Particle3 "std::array<double, 3>" 


cdef extern from "../../tgsl/library/io/BinaryIO.h" namespace "TGSL::IO":
  void Deserialize[T](vector[T]& v, string filename)

#cdef extern from "../../tgsl/library/Mesh.h" namespace "TGSL::MESH":
  #void ComputeBoundaryTriMesh(vector[int]& tet_mesh, vector[int]& tri_mesh)

cdef extern from "../../tgsl/library/io/IO.h" namespace "TGSL::IO":
  void WriteTrisGEO[T](vector[T]& positions, int Np, vector[int] mesh, int mesh_size, string filename)

def WriteTrisFrame(positions, Np, mesh, mesh_size, filename):
  WriteTrisGEO[double](positions, Np, mesh, mesh_size, filename)

#def ComputeBoundaryTriMesh(tet_mesh, boundary_mesh):
  #cdef vector[int] boundary_mesh_copy= boundary_mesh
  #ComputeBoundaryTriMesh(tet_mesh, boundary_mesh_copy)
  #boundary_mesh[:] = boundary_mesh_copy

def ReadDoubleVector(v, filename):
  cdef vector[double] vec = v
  Deserialize[double](vec, filename)
  v[:] = vec

def AppendDoubleVector(v, filename):
  cdef vector[double] vec
  Deserialize[double](vec, filename)
  v = v + vec

def ReadIntVector(v, filename):
  cdef vector[int] vec = v
  Deserialize[int](vec, filename)
  v[:] = vec
