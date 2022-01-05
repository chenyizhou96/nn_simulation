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
