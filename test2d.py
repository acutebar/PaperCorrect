# Updated test2d.py
import numpy as np
import matplotlib.pyplot as plt
import detector

def stereographic_projection(x, y, z):
    u = x / (1 + z)
    v = y / (1 + z)
    return u, v

def generate_random_rho_function():
    # Randomize harmonics for the scaling function
    c1 = np.random.uniform(-0.5, 0.5)
    c2 = np.random.uniform(-0.5, 0.5)
    c3 = np.random.uniform(-0.5, 0.5)
    
    def rho(x, y, z):
        return 1.0 + c1 * (x**2 - y**2) + c2 * z + c3 * x * y
    return rho

def run_test():
    rho_func = generate_random_rho_function()
    
    # 1. Generate the true spherical surface grid
    theta = np.linspace(0, 2*np.pi, 2000)
    phi = np.linspace(0, np.pi/2, 1000)
    Theta, Phi = np.meshgrid(theta, phi)
    
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi)
    
    # 2. Extract a uniformly sampled point cloud 
    c_theta = np.linspace(0, 2*np.pi, 30)
    c_phi = np.linspace(0, np.pi/2, 15)
    C_Theta, C_Phi = np.meshgrid(c_theta, c_phi)
    
    cloud_x = (np.sin(C_Phi) * np.cos(C_Theta)).flatten()
    cloud_y = (np.sin(C_Phi) * np.sin(C_Theta)).flatten()
    cloud_z = (np.cos(C_Phi)).flatten()
    
    cloud_rho = rho_func(cloud_x, cloud_y, cloud_z) + np.random.normal(0, 0.02, cloud_x.size)
    
    # 3. Project cloud to stereographic domain
    cloud_u, cloud_v = stereographic_projection(cloud_x, cloud_y, cloud_z)
    cloud_dataset = np.column_stack([cloud_u, cloud_v, cloud_rho])
    
    # 4. Evaluate true scaling over the grid
    true_Rho = rho_func(X, Y, Z)
    X_true = true_Rho * X
    Y_true = true_Rho * Y
    Z_true = true_Rho * Z
    
    # 5. Fit surface over the evaluation grid
    U_grid, V_grid = stereographic_projection(X, Y, Z)
    fitted_Rho, _, _, _, _, _ = detector.surface_fit(U_grid, V_grid, cloud_dataset, deg=2, bin_size=0.4)
    
    X_fit = fitted_Rho * X
    Y_fit = fitted_Rho * Y
    Z_fit = fitted_Rho * Z
    
    # 6. Plotting
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_true, Y_true, Z_true, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_title("True Scaled Surface $\\rho_{true} \\cdot S^2_+$")
    ax1.set_zlim(0, 1.5)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_fit, Y_fit, Z_fit, cmap='plasma', alpha=0.8, edgecolor='none')
    ax2.scatter(cloud_rho * cloud_x, cloud_rho * cloud_y, cloud_rho * cloud_z, color='k', s=10, alpha=1.0)
    ax2.set_title("B-Spline Fit $\\rho_{fit} \\cdot S^2_+$ from Uniform Cloud")
    ax2.set_zlim(0, 1.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_test()
