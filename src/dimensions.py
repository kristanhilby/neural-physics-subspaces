import igl
import numpy as np
import jax.numpy as jnp
import os

def make_body(file, density, scale):

    v, f = igl.read_triangle_mesh(file)
    v = scale*v

    vol = igl.massmatrix(v,f).data
    vol = np.nan_to_num(vol) # massmatrix returns Nans in some stewart meshes

    c = np.sum( vol[:,None]*v, axis=0 ) / np.sum(vol) 
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:,None]*W) * density

    x0 = jnp.array( [[1, 0, 0],[0, 1, 0],[0, 0, 1], c] )

    body = {'v': v, 'f': f, 'W':W, 'x0': x0, 'mass': mass }
    return body

body = make_body( os.path.join(".", "data", "klann-red.obj"), 1000, 1.0)