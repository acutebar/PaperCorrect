import numpy as np
import math
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    projective_kinematics,
    compute_projective_bending_energy,
    total_energy
)
import torch

USE_CONE_SAMPLING = True

def generate_uniform_rho_cloud(num_points=256, max_phi=None):
    """
    Uniformly samples a spherical cap on S^2_+ and maps to stereographic (u, v) with rho=1.
    If max_phi is None, samples the full upper hemisphere [0, pi/2].
    """
    indices = torch.arange(0, num_points, dtype=torch.float64) + 0.5
    
    # ==========================================================================
    # FIX 2: Sample Only the Image Cone (Logic)
    # ==========================================================================
    if USE_CONE_SAMPLING and max_phi is not None:
        # Uniform area sampling on a spherical cap of half-angle max_phi:
        # cos(phi) runs from 1 down to cos(max_phi)
        cos_max = math.cos(max_phi)
        phi = torch.arccos(1.0 - (indices / num_points) * (1.0 - cos_max))
    else:
        # Revert: Sample full half-sphere [0, pi/2]
        phi = torch.arccos(1.0 - indices / num_points)
    # ==========================================================================
    
    theta = torch.pi * (1.0 + 5.0**0.5) * indices
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    u = x / (1.0 + z)
    v = y / (1.0 + z)
    rho = torch.ones_like(u)
    
    return torch.column_stack([u, v, rho])



def run_gradient_descent(t, curves, f = 50.0, num_points=256, step=lambda n: 0.2, steps=500, eps=1e-5):
    deformation_cloud = generate_uniform_rho_cloud(num_points, torch.pi/3)
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)

    for i in range(steps):
        TE = total_energy(t, curves, cloud_coords, cloud_values, f)
        TE.backward()
        grad_norm = cloud_values.grad.norm().item()
        if grad_norm < eps:
            break

        with torch.no_grad():
            cloud_values -= step(i) * cloud_values.grad
            cloud_values.grad.zero_()

    return torch.column_stack([cloud_coords, cloud_values])
