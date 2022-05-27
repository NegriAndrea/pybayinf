#!/usr/bin/env python
# -*- coding: utf-8 -*-
# to import only when cupy is installed

# version of the f_single_schechter that uses cupy
def f_single_schechternumpy(M, alpha, phi, Mo):
    # I use cp.multiply so then I can switch to numpy by simply changing the
    # import statement
    import cupy as cp
    f = cp.multiply(phi , 10.0**(0.4 * cp.multiply((alpha + 1) , (Mo - M))))
    out = cp.multiply(0.4 * cp.log(10.0) * cp.exp(-10.0**(0.4 * (Mo - M))), f)
    return out

