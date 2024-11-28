import unittest
from modmesh import SimpleArrayUint64, SimpleArrayFloat64
import numpy as np

class TimeBuffer(unittest.TestCase):
    #a = np.array([[1.,2.,3.,4.,5.], [10.,20.,30.,40.,50.],[100.,200.,300.,400.,500.]])
    #print(a)
    #b = SimpleArrayFloat64(array=a[1, 1:])
    #print(b.ndarray)
    
    np_shape = (5,5,5,5)
    a = np.ndarray(shape= np_shape, dtype=float)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                for l in range(5):
                    a[i,j,k,l] = i*1000 + j*100 + k*10 + l
                #a[i,j,k] = i*100 + j*10 + k

    #np_b = a[1:3, 2:5, 3:5, 1]
    #np_b = a
    #np_b_copy = np_b.copy()
    #print(np_b)
    #print(np_b_copy)

    b = SimpleArrayFloat64(array=a)
    print(b.ndarray)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                for l in range(5):
                    assert b.ndarray[i,j,k,l] == a[i,j,k,l]
    #b = SimpleArrayFloat64(array=np_b)
    #print(b.ndarray)

    