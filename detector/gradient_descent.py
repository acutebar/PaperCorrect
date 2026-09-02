import numpy as np
from scipy.optimize import minimize
from .energy import total_energy

def generate_uniform_rho_cloud(num_points=64):
    """Uniformly samples S^2_+ and maps to stereographic (u,v) with rho=1."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - indices/num_points) 
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    u = x / (1 + z)
    v = y / (1 + z)
    rho = np.ones_like(u)
    
    return np.column_stack([u, v, rho])

def run_gradient_descent(T_domain, curves, f_pixels):
    """Delegates the gradient descent to SciPy's optimized L-BFGS-B solver."""
    
    cloud = generate_uniform_rho_cloud(num_points=64)
    initial_rho = cloud[:, 2].copy()
    
    # Downsample the time domain strictly for the optimization loop to speed up evaluations
    T_fast = np.linspace(0, 1, 20) 
    
    def objective(rho_params):
        cloud[:, 2] = rho_params
        return total_energy(T_fast, curves, cloud, f=f_pixels)
        
    print("Delegating optimization to SciPy (L-BFGS-B)...")
    
    result = minimize(
        fun=objective, 
        x0=initial_rho, 
        method='L-BFGS-B',
        jac='2-point',
        options={
            'disp': True,
            'maxiter': 50,
            'ftol': 1e-5
        }
    )
    
    cloud[:, 2] = result.x
    print(f"Optimization finished. Final Energy: {result.fun:.5f}")
    
    return cloud
