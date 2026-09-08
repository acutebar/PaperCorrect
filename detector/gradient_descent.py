import numpy as np
import math
from scipy.optimize import minimize
from .energy import (
    surface_fit,
    total_energy,
    evaluate_complexity
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
    
    # \rho / R = w (where w is the depth field). To initialize flat paper at depth w:
    # \rho = w * R
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

def run_gradient_descent(t, curves, f=50.0, num_points=256, learning_rate=0.001, steps=500, eps=1e-4, initial_cloud=None, deg=2, bin_size=0.5):
    return run_adam_descent(t, curves, f, num_points, learning_rate, steps, eps, initial_cloud, deg, bin_size)


def run_vanilla_descent(t, curves, f = 50.0, num_points=256, learning_rate= 0.001, steps=500, eps=1e-5, initial_cloud=None, deg=2, bin_size=0.5):
    deformation_cloud = initial_cloud 
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)

    # -------------------------------------------------------------
    # Live 3D Plot Setup
    # -------------------------------------------------------------
    plt.ion()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Infer grid resolution along each axis
    grid_dim = int(round(math.sqrt(cloud_coords.shape[0])))
    x_flat = cloud_coords[:, 0].detach().cpu().numpy()
    y_flat = cloud_coords[:, 1].detach().cpu().numpy()
    
    surf_plot = None

    for i in range(steps):
        #optimizer.zero_grad()
        TE = total_energy(t, curves, cloud_coords, cloud_values, f, deg, bin_size)
        TE.backward()
        
        grad_norm = cloud_values.grad.norm().item()
        print(f"VANILLVANILLAA  [GD Step {i+1:3d}/{steps}] Energy: {TE.item():.4f} | Grad Norm: {grad_norm:.6f}")

        # ---------------------------------------------------------
        # Plot check: first 5 steps (i < 5), every n/10th step, or last step
        # ---------------------------------------------------------
        all_mins = [ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]
        all_maxs = [ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]]
        common_lim = (min(all_mins), max(all_maxs))
        
        ax.set_xlim(common_lim)
        ax.set_ylim(common_lim)
        ax.set_zlim(common_lim)
        ax.set_box_aspect([1, 1, 1])
        interval = max(1, steps // 10)
        if i < 5 or (i + 1) % interval == 0 or i == steps - 1:
            with torch.no_grad():
                w_flat = cloud_values.detach().cpu().numpy()
                
                # Physical 3D projection: (w * x, w * y, w * f)
                px = (w_flat * x_flat).reshape(grid_dim, grid_dim)
                py = (w_flat * y_flat).reshape(grid_dim, grid_dim)
                pz = (-w_flat * f).reshape(grid_dim, grid_dim)
                
                ax.clear()
                ax.plot_surface(px, py, pz, cmap='viridis', edgecolor='none', alpha=0.9)
                ax.set_title(f"Step {i+1}/{steps} | Energy: {TE.item():.4f}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

        if grad_norm < eps:
            print(f"  [GD] Converged at step {i+1} with gradient norm {grad_norm:.6e}")
            break

        with torch.no_grad():
            cloud_values -= learning_rate * cloud_values.grad
            cloud_values.grad.zero_()
    plt.ioff()
    plt.close(fig)

    return torch.column_stack([cloud_coords, cloud_values])

def run_adam_descent(t, curves, f=1.0, num_points=256, learning_rate=0.001, steps=500, eps=1e-5, initial_cloud=None, deg=2, bin_size=0.5):
    deformation_cloud = initial_cloud
    cloud_coords = deformation_cloud[:, :2].detach()
    cloud_values = deformation_cloud[:, 2].detach().clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([cloud_values], lr=learning_rate)

    # -------------------------------------------------------------
    # Live 3D Plot Setup
    # -------------------------------------------------------------
    plt.ion()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Infer grid resolution along each axis
    grid_dim = int(round(math.sqrt(cloud_coords.shape[0])))
    x_flat = cloud_coords[:, 0].detach().cpu().numpy()
    y_flat = cloud_coords[:, 1].detach().cpu().numpy()
    
    surf_plot = None

    for i in range(steps):
        optimizer.zero_grad()
        TE = total_energy(t, curves, cloud_coords, cloud_values, f, deg, bin_size)
        TE.backward()
        
        grad_norm = cloud_values.grad.norm().item()
        print(f"  [GD Step {i+1:3d}/{steps}] Energy: {TE.item():.4f} | Grad Norm: {grad_norm:.6f}")

        # ---------------------------------------------------------
        # Plot check: first 5 steps (i < 5), every n/10th step, or last step
        # ---------------------------------------------------------

        interval = max(1, steps // 10)
        if i < 5 or (i + 1) % interval == 0 or i == steps - 1:
            with torch.no_grad():
                w_flat = cloud_values.detach().cpu().numpy()
                w_fit, _, _, _, _, _ = surface_fit(cloud_coords[:, 0], cloud_coords[:, 1], torch.column_stack([cloud_coords, cloud_values]), deg=deg, bin_size=bin_size)
                
                # Physical 3D projection: (w * x, w * y, w * f)
                px = (w_fit * x_flat).reshape(grid_dim, grid_dim)
                py = (w_fit * y_flat).reshape(grid_dim, grid_dim)
                pz = (-w_fit * f).reshape(grid_dim, grid_dim)

                qx = (w_flat * x_flat).reshape(grid_dim, grid_dim)
                qy = (w_flat * y_flat).reshape(grid_dim, grid_dim)
                qz = (-w_flat * f).reshape(grid_dim, grid_dim)


                #if surf_plot is not None:
                #    surf_plot.remove()

                ax.clear()
                
                #ax.plot_surface(px, py, pz, cmap='viridis', edgecolor='none', alpha=0.9)
                ax.plot_surface(qx, qy, qz, cmap='gray', edgecolor='none', alpha=0.9)


                #all_mins = [ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]
                #all_maxs = [ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]]
                #common_lim = (min(all_mins), max(all_maxs))
                #
                #ax.set_xlim(common_lim)
                #ax.set_ylim(common_lim)
                #ax.set_zlim(common_lim)
                #ax.set_box_aspect([1, 1, 1])

                ax.set_title(f"Step {i+1}/{steps} | Energy: {TE.item():.4f}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(1.0)

        if grad_norm < eps:
            print(f"  [GD] Converged at step {i+1} with gradient norm {grad_norm:.6e}")
            break
            
        optimizer.step()

    plt.ioff()
    plt.close(fig)

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
            deg=deg,
            bin_size=bin_size
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
    W_COMPLEXITY = 0.5 
    
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
