import numpy as np
def periodic_laplacian(values, dx=1.0):
    data=np.asarray(values,dtype=float)
    result=np.zeros_like(data)
    result[1:-1]=(data[:-2]-2*data[1:-1]+data[2:])/(dx*dx)
    return result
