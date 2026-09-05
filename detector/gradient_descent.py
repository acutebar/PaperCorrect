import numpy as np
import math
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    projective_kinematics,
    compute_projective_bending_energy,
    total_energy,
    evaluate_complexity
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

def generate_flat_rho_cloud(num_points, max_angle=torch.pi/3):
    # Call your existing function to preserve your exact (u, v) point distribution
    cloud = generate_uniform_rho_cloud(num_points, max_angle)
    u = cloud[:, 0]
    v = cloud[:, 1]
    
    # Override rho to map to a flat plane at Z=1 instead of a unit sphere
    flat_rho = (1.0 + u**2 + v**2) / (1.0 - u**2 - v**2)
    
    return torch.column_stack([u, v, flat_rho])

def run_gradient_descent(t, curves, f=50.0, num_points=256, learning_rate=0.01, steps=500, eps=1e-5, initial_cloud=None):
    deformation_cloud = initial_cloud
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([cloud_values], lr=learning_rate)

    for i in range(steps):
        optimizer.zero_grad()
        TE = total_energy(t, curves, cloud_coords, cloud_values, f)
        TE.backward()
        
        grad_norm = cloud_values.grad.norm().item()
        if (i + 1) % max(1, steps // 10) == 0 or i == 0 or i == steps - 1:
            print(f"  [GD Step {i+1:3d}/{steps}] Energy: {TE.item():.4f} | Grad Norm: {grad_norm:.6f}")
            
        if grad_norm < eps:
            print(f"  [GD] Converged at step {i+1} with gradient norm {grad_norm:.6e}")
            break
            
        optimizer.step()

    return torch.column_stack([cloud_coords, cloud_values])

def generate_spherical_rho_cloud(num_points, span):
    grid_1d = torch.linspace(-span, span, int(np.sqrt(num_points)))
    U, V = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    rho_sphere = torch.ones_like(U)
    return torch.stack([U.flatten(), V.flatten(), rho_sphere.flatten()], dim=1)

def generate_hyperbolic_rho_cloud(num_points, span):
    grid_1d = torch.linspace(-span, span, int(np.sqrt(num_points)))
    U, V = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    rho_saddle = 1.0 + (U**2 - V**2)
    return torch.stack([U.flatten(), V.flatten(), rho_saddle.flatten()], dim=1)

def generate_random_rho_cloud(num_points, span, base_cloud):
    noise = torch.randn(num_points) * 0.1
    rand_cloud = base_cloud.clone()
    rand_cloud[:, 2] += noise
    return rand_cloud

def run_multi_start_optimization(T, active_curves, f, num_points, learning_rate, steps, eps):
    span = torch.pi / 3
    flat_cloud = generate_flat_rho_cloud(num_points, span)
    
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
            initial_cloud=init_cloud
        )
        
        with torch.no_grad():
            coords = opt_cloud[:, :2]
            values = opt_cloud[:, 2]
            
            energy = total_energy(T, active_curves, coords, values, f).item()
            complexity = evaluate_complexity(T, active_curves, coords, values, f)
            
        results.append({
            'name': name,
            'cloud': opt_cloud,
            'energy': energy,
            'complexity': complexity
        })
        print(f"     Energy: {energy:.4f} | Complexity: {complexity:.4f}")

    # Hardcoded weights (adjust these based on the relative magnitudes of your E and H^2)
    W_ENERGY = 1.0
    W_COMPLEXITY = 0.1 
    
    best_score = float('inf')
    winner = None
    
    for res in results:
        score = (W_ENERGY * res['energy']) + (W_COMPLEXITY * res['complexity'])
        res['score'] = score
        
        if score < best_score:
            best_score = score
            winner = res
            
    print(f"\nWinner: {winner['name']} (Score: {best_score:.4f} | E: {winner['energy']:.4f} | H^2: {winner['complexity']:.4f})")
        
    return winner['cloud'], winner['energy']

    #min_energy_idx = min(range(len(results)), key=lambda i: results[i]['energy'])
    #min_energy = results[min_energy_idx]['energy']
    #
    #tolerance_threshold = min_energy * 1.20 + 0.1
    #
    #comparable_candidates = [res for res in results if res['energy'] <= tolerance_threshold]
    #
    #if len(comparable_candidates) == 1:
    #    winner = comparable_candidates[0]
    #    print(f"\nWinner: {winner['name']} (Selected via Dominant Energy)")
    #else:
    #    winner = min(comparable_candidates, key=lambda x: x['complexity'])
    #    print(f"\nWinner: {winner['name']} (Selected via Minimal Complexity)")
    #    
    #return winner['cloud'], winner['energy']
