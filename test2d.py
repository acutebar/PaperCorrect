import numpy as np
import matplotlib.pyplot as plt
import detector

def stereographic_projection(x, y, z):
    u = x / (1 + z)
    v = y / (1 + z)
    return u, v

def run_test():
    # 1. Generate a random cloud of points on the upper half sphere
    N = 1000
    rand_theta = np.random.uniform(0, 2*np.pi, N)
    # Uniform sampling over the upper hemisphere
    rand_phi = np.arccos(np.random.uniform(0, 1, N)) 
    
    cloud_x = np.sin(rand_phi) * np.cos(rand_theta)
    cloud_y = np.sin(rand_phi) * np.sin(rand_theta)
    cloud_z = np.cos(rand_phi)
    
    # Generate purely random noise for rho
    raw_rho = np.random.uniform(0.5, 1.5, N)
    
    # 2. Apply a low-pass Gaussian filter directly to the point cloud (Kernel Smoothing)
    points = np.column_stack([cloud_x, cloud_y, cloud_z])
    # Compute pairwise squared distances using NumPy broadcasting
    dists_sq = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=-1)
    
    sigma = 0.25  # Gaussian bandwidth (smoothness parameter)
    weights = np.exp(-dists_sq / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    
    # The filtered cloud is a local weighted average of the random noise
    smooth_rho = weights @ raw_rho
    
    # 3. Project cloud to stereographic domain
    cloud_u, cloud_v = stereographic_projection(cloud_x, cloud_y, cloud_z)
    cloud_dataset = np.column_stack([cloud_u, cloud_v, smooth_rho])
    
    # 4. Generate a regular evaluation grid for plotting the surface
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi/2, 100)
    Theta, Phi = np.meshgrid(theta, phi)
    
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi)
    
    U_grid, V_grid = stereographic_projection(X, Y, Z)
    
    # 5. Fit the surface using the energy functional script
    fitted_Rho, _, _, _, _, _ = detector.surface_fit(U_grid, V_grid, cloud_dataset, deg=2, bin_size=0.4)
    
    # Scale the evaluation grid coordinates by the fitted Rho
    X_fit = fitted_Rho * X
    Y_fit = fitted_Rho * Y
    Z_fit = fitted_Rho * Z
    
    # 6. Plotting (Single Combined Plot)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the fitted surface
    ax.plot_surface(X_fit, Y_fit, Z_fit, cmap='plasma', alpha=0.7, edgecolor='none')
    
    # Overlay the smoothed point cloud
    ax.scatter(smooth_rho * cloud_x, smooth_rho * cloud_y, smooth_rho * cloud_z, 
               color='k', s=15, alpha=1.0, label="Smoothed Cloud")
               
    ax.set_title("B-Spline Fit $\\rho_{fit} \\cdot S^2_+$ from Smoothed Random Cloud")
    
    # Dynamic Z-axis scaling to fit the data
    max_z = max(np.max(Z_fit), np.max(smooth_rho * cloud_z))
    ax.set_zlim(0, max_z + 0.1)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_test()
