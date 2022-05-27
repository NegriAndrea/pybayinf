
# ================================================================================
# Schechter function converted to magnitudes

def f_single_schechter(M, alpha, phi, Mo):
    import numpy as np
    f = phi * 10.0**(0.4 * (alpha + 1) * (Mo - M))
    out = 0.4 * np.log(10.0) * np.exp(-10.0**(0.4 * (Mo - M))) * f
    return out

# ================================================================================
# Sum of two Schechter functions converted to magnitudes (one M*)

def f_double_schechter(M, alpha1, alpha2, phi1, phi2, Mo):
    import numpy as np
    f1 = phi1 * 10.0**(0.4 * (alpha1 + 1) * (Mo - M))
    f2 = phi2 * 10.0**(0.4 * (alpha2 + 1) * (Mo - M))
    out = 0.4 * np.log(10.0) * np.exp(-10.0**(0.4 * (Mo - M))) * (f1 + f2)
    return out

def f_double_schechter_a_const(M, alpha2, phi1, phi2, Mo):
    import numpy as np
    alpha1=-1.
    f1 = phi1 * 10.0**(0.4 * (alpha1 + 1) * (Mo - M))
    f2 = phi2 * 10.0**(0.4 * (alpha2 + 1) * (Mo - M))
    out = 0.4 * np.log(10.0) * np.exp(-10.0**(0.4 * (Mo - M))) * (f1 + f2)
    return out

# ================================================================================
# Sum of two Schechter functions converted to magnitudes (two M*)

def f_double_schechter_s(M, alpha1, alpha2, phi1, phi2, M1, M2):
    import numpy as np
    f1 = phi1 * 10.0**(0.4 * (alpha1 + 1) * (M1 - M)) * np.exp(-10.0**(0.4 * (M1 - M)))
    f2 = phi2 * 10.0**(0.4 * (alpha2 + 1) * (M2 - M)) * np.exp(-10.0**(0.4 * (M2 - M)))
    out = 0.4 * np.log(10.0) * (f1 + f2)
    return out

# ================================================================================
# Double Schechter function used in Popesso et al. (2006) converted to magnitudes

def f_double_schechter_popesso_2006(M, alpha1, alpha2, phi, M1, M2):
    import numpy as np
    # bright end
    f1 = 10.0**(0.4 * (alpha1 + 1) * (M1 - M)) * np.exp(-10.0**(0.4 * (M1 - M)))
    # faint end
    f2 = 10.0**(0.4 * (alpha2 + 1) * (M2 - M)) * 10.0**(0.4 * (-1) * (M2 - M1)) * np.exp(-10.0**(0.4 * (M2 - M)))
    # combined
    out = 0.4 * np.log(10.0) * phi * (f1 + f2)
    return out

# ================================================================================
