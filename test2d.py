import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import from your detector module W
from detector.energy import surface_fit
from detector.gradient_descent import generate_random_smooth_cloud, generate_flat_cloud

def test_surface_fit():
    print("Generating random smooth cloud...")
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    cloud = generate_random_smooth_cloud(x_min, x_max, y_min, y_max, mult=20, depth_mean=1.5, depth_var=0.5, sigma=2.0)
    #cloud = generate_flat_cloud(x_min, x_max, y_min, y_max, mult=20, depth=1.5)
    
    print("Fitting surface...")
    # Create evaluation grid
    eval_pts = 50
    x_val = torch.linspace(x_min, x_max, eval_pts, dtype=torch.float64)
    y_val = torch.linspace(y_min, y_max, eval_pts, dtype=torch.float64)
    X, Y = torch.meshgrid(x_val, y_val, indexing='xy')
    
    u_flat = X.reshape(-1)
    v_flat = Y.reshape(-1)
    
    # Fit the surface (w is the depth field)
    w_flat, w_x, w_y, w_xx, w_yy, w_xy = surface_fit(u_flat, v_flat, cloud, deg=2, bin_size=0.2)
    W = w_flat.reshape(eval_pts, eval_pts).detach().numpy()
    
    X_np = X.detach().numpy()
    Y_np = Y.detach().numpy()
    
    # Physical 3D mapping: P = w * (x, y, 1.0)
    f = 1.0
    R = np.sqrt(X_np**2 + Y_np**2 + f**2)
    P_X = W/R * X_np
    P_Y = W/R * Y_np
    P_Z = W/R * f
    
    # Plotting
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Raw depth function w(x, y)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_np, Y_np, W, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.scatter(cloud[:, 0].numpy(), cloud[:, 1].numpy(), cloud[:, 2].numpy(), c='red', s=5, label='Control Points')
    ax1.set_title('Raw Parameter Space: Depth $w(x, y)$')
    ax1.set_xlabel('Image x')
    ax1.set_ylabel('Image y')
    ax1.set_zlabel('Depth w')
    ax1.legend()

    # Plot 2: Physical 3D Surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(P_X, P_Y, P_Z, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.9)
    ax2.set_title('Physical 3D Surface: $\mathbf{P} = w(x, y) \cdot (x, y, f)$')
    ax2.set_xlabel('3D X')
    ax2.set_ylabel('3D Y')
    ax2.set_zlabel('3D Z')
    
    # Ensure equal aspect ratio for realistic paper representation
    max_range = np.array([P_X.max()-P_X.min(), P_Y.max()-P_Y.min(), P_Z.max()-P_Z.min()]).max() / 2.0
    mid_x = (P_X.max()+P_X.min()) * 0.5
    mid_y = (P_Y.max()+P_Y.min()) * 0.5
    mid_z = (P_Z.max()+P_Z.min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_surface_fit()
