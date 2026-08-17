import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import math


# (time_array, x_array, y_array, vx_array, vy_array, ax_array, ay_array, rho_restricted_to_curve(t), derivative of w, second derivative of w, focal length of camera)
def compute_projective_bending_energy(t, x, y, vx, vy, ax, ay, w, w_dt, w_ddt, f):
    """
    Computes the parameterization-invariant bending energy of the curve embedded in 3D space
    """
    V1_x = -f * vy
    V1_y =  f * vx
    V1_z =  x * vy - y * vx
    
    V2_x = -f * ay
    V2_y =  f * ax
    V2_z =  x * ay - y * ax
    
    V3_z = vx * ay - vy * ax
    
    c1 = 2 * (w_dt**2) - (w * w_ddt)
    c2 = w * w_dt
    c3 = w**2
    
    K_x = c1 * V1_x + c2 * V2_x
    K_y = c1 * V1_y + c2 * V2_y
    K_z = c1 * V1_z + c2 * V2_z + c3 * V3_z
    
    K_norm_sq = K_x**2 + K_y**2 + K_z**2
    
    # Step 4.2: The Speed Squared S(t)
    term1 = (w_dt**2) * (x**2 + y**2 + f**2)
    term2 = 2 * w * w_dt * (x * vx + y * vy)
    term3 = (w**2) * (vx**2 + vy**2)
    
    S = term1 + term2 + term3
    
    # Add epsilon to prevent division by zero in perfectly static segments
    eps = 1e-12
    
    integrand = K_norm_sq / ((S + eps)**2.5)
    
    # Integrate over the parameter t using trapezoidal approximation
    energy = np.trapezoid(integrand, t)
    
    return energy

