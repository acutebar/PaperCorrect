import numpy as np
import math
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    total_energy,
    quadratic_fit
)
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_flat_cloud(x_start, x_end, y_start, y_end, mult=100, depth=1.0):
    num_points = int(mult) + 1
    
    xs = torch.linspace(float(x_start), float(x_end), steps=num_points, dtype=torch.float64)
    ys = torch.linspace(float(y_start), float(y_end), steps=num_points, dtype=torch.float64)
    
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    
    u = grid_x.reshape(-1)
    v = grid_y.reshape(-1)
    
    # Pure flat plane at constant projective depth
    w = torch.full_like(u, depth)
    
    return torch.column_stack([u, v, w])

def run_gradient_descent(t, curves, f=50.0, num_points=256, learning_rate=0.001, steps=500, eps=1e-4, initial_cloud=None):
    return run_adam_descent(t, curves, f, num_points, learning_rate, steps, eps, initial_cloud)

def run_vanilla_descent(t, curves, f = 50.0, num_points=256, learning_rate= 0.001, steps=500, eps=1e-5, initial_cloud=None):
    deformation_cloud = initial_cloud 
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)

    pinvX, (cu, cv) = quadratic_fit(cloud_coords)

    for i in range(steps):
        #optimizer.zero_grad()
        TE = total_energy(t, curves, cloud_values, pinvX, cu, cv, f)
        TE.backward()
        
        grad_norm = cloud_values.grad.norm().item()
        print(f"VANILLVANILLAA  [GD Step {i+1:3d}/{steps}] Energy: {TE.item():.4f} | Grad Norm: {grad_norm:.6f}")

        if grad_norm < eps:
            print(f"  [GD] Converged at step {i+1} with gradient norm {grad_norm:.6e}")
            break

        with torch.no_grad():
            cloud_values -= learning_rate * cloud_values.grad
            cloud_values.grad.zero_()

    return torch.column_stack([cloud_coords, cloud_values])

def run_adam_descent(t, curves, f=1.0, num_points=256, learning_rate=0.001, steps=500, eps=1e-5, initial_cloud=None, deg=2, bin_size=0.5):
    deformation_cloud = initial_cloud
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([cloud_values], lr=learning_rate)
    
    pinvX, (cu, cv) = quadratic_fit(cloud_coords)

    for i in range(steps):
        optimizer.zero_grad()
        TE = total_energy(t, curves, cloud_values, pinvX, cu, cv, f)
        TE.backward()
        
        grad_norm = cloud_values.grad.norm().item()
        print(f"  [GD Step {i+1:3d}/{steps}] Energy: {TE.item():.4f} | Grad Norm: {grad_norm:.6f}")


        if grad_norm < eps:
            print(f"  [GD] Converged at step {i+1} with gradient norm {grad_norm:.6e}")
            break
            
        optimizer.step()


    return torch.column_stack([cloud_coords, cloud_values])
