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

def generate_random_smooth_cloud(x_start, x_end, y_start, y_end, mult=100, depth_mean=1.0, depth_var=0.3, sigma=4.0):
    num_points = int(mult) + 1
    
    xs = np.linspace(float(x_start), float(x_end), num_points)
    ys = np.linspace(float(y_start), float(y_end), num_points)
    
    # 1. Generate pure uncorrelated Gaussian noise
    raw_noise = np.random.normal(loc=0.0, scale=depth_var, size=(num_points, num_points))
    
    # 2. Apply low-pass Gaussian filter to create a smooth surface
    smoothed_noise = gaussian_filter(raw_noise, sigma=sigma)
    
    # Normalize the smoothed noise back to the desired variance scale
    if np.std(smoothed_noise) > 1e-8:
        smoothed_noise = (smoothed_noise / np.std(smoothed_noise)) * depth_var
        
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    
    R_grid = np.sqrt(grid_x**2 + grid_y**2 + 1.0)
    rho_grid = (depth_mean + smoothed_noise) * R_grid
    
    u = torch.tensor(grid_x.reshape(-1), dtype=torch.float64)
    v = torch.tensor(grid_y.reshape(-1), dtype=torch.float64)
    rho = torch.tensor(rho_grid.reshape(-1), dtype=torch.float64)
    
    return torch.column_stack([u, v, rho])

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

def run_multi_start_optimization(T, active_curves, f, num_points, learning_rate, steps, eps, x_start, x_end, y_start, y_end, mult=100, deg=2, bin_size=0.5):
    span = torch.pi / 3
    flat_cloud = generate_flat_cloud(x_start, x_end, y_start, y_end, mult)
    
    topologies = [
        ("Flat", flat_cloud)
    ]
    results = []
    
    for name, init_cloud in topologies:
        print(f"  -> Testing {name} topology...")
        
        opt_cloud = run_gradient_descent(
            T, active_curves, f=f, 
            num_points=num_points, 
            learning_rate=learning_rate, 
            steps=steps, 
            eps=eps,
            initial_cloud=init_cloud,
        )
        
        with torch.no_grad():
            coords, values = opt_cloud[:, :2], opt_cloud[:, 2]
            pinvX, (cu, cv) = quadratic_fit(coords)
            energy = total_energy(T, active_curves, values, pinvX, cu, cv, f)
            
        results.append({
            'name': name,
            'cloud': opt_cloud,
            'energy': energy,
        })
        print(f"     Energy: {energy:.4f}")

    best_score = float('inf')
    winner = None
    
    for res in results:
        score =  res['energy']
        res['score'] = score
        
        if score < best_score:
            best_score = score
            winner = res
            
    print(f"\nWinner: {winner['name']} (E: {winner['energy']:.4f})")
        
    return winner['cloud'], winner['energy']
