import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA

# Check and import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    # Set plot style
    plt.style.use('default')  # Use a more compatible default style
    sns.set_palette("husl")
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries loaded successfully")
except ImportError as e:
    print(f"⚠ Error importing visualization libraries: {e}")
    print("Install with: pip install matplotlib seaborn")
    VISUALIZATION_AVAILABLE = False

# Step 1: Generate 1000 pairs of random points in [-6, 6]
data = np.random.uniform(-6, 6, size=(1000, 2))

# Step 2: Apply 20 transformation functions
def apply_functions(x):
    return np.array([
        np.sin(x[:,0]),
        np.cos(x[:,1]),
        np.tanh(x[:,0]),
        np.exp(-x[:,1]**2),
        x[:,0]**2 + x[:,1]**2,
        np.abs(x[:,0] - x[:,1]),
        x[:,0] * x[:,1],
        np.log1p(np.abs(x[:,0])),
        np.log1p(np.abs(x[:,1])),
        np.sin(x[:,0]*x[:,1]),
        np.cos(x[:,0]+x[:,1]),
        x[:,0]**3,
        x[:,1]**3,
        np.maximum(x[:,0], x[:,1]),
        np.minimum(x[:,0], x[:,1]),
        np.arctan2(x[:,0], x[:,1]),
        x[:,0] / (1 + np.abs(x[:,1])),
        x[:,1] / (1 + np.abs(x[:,0])),
        np.exp(x[:,0]*0.1),
        np.exp(x[:,1]*0.1)
    ]).T

X = apply_functions(data)

# Step 3: Define the autoencoder (20D -> 2D -> 20D)
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 20)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# Step 4: Train the autoencoder
print("Training autoencoder...")
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.tensor(X, dtype=torch.float32)

for epoch in range(1000):
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/1000, Loss: {loss.item():.6f}")

# Step 5: Prepare data for PCA
print("\nPreparing PCA...")
# Get the outputs of the autoencoder decoder
with torch.no_grad():
    decoder_outputs = model.decode(model.encode(X_tensor))
    decoder_outputs_np = decoder_outputs.numpy()

# Apply PCA to reduce from 20D to 2D
pca = PCA(n_components=2)
pca.fit(decoder_outputs_np)

print(f"Explained variance by PCA: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Step 6: Objective function using PCA
def rosenbrock_pca_objective(x, y):
    # Point in the autoencoder's latent space
    z = torch.tensor([[x, y]], dtype=torch.float32)

    # Decode with the autoencoder
    with torch.no_grad():
        decoder_output = model.decode(z)
        decoder_output_np = decoder_output.numpy()

        # Project with PCA
        projection = pca.transform(decoder_output_np)
        proj = projection[0]

    # Rosenbrock function (negative because we want to maximize)
    return -((1 - proj[0])**2 + 100*(proj[1] - proj[0]**2)**2)

# Step 7: Set up BO progress tracking
# List to store progress
bo_progress = []

# Step 8: Bayesian optimization
print("\nStarting Bayesian optimization with autoencoder + PCA...")
opt = BayesianOptimization(
    f=rosenbrock_pca_objective,
    pbounds={'x': (-3, 3), 'y': (-3, 3)},
    verbose=2,
    random_state=1
)

# Run optimization and capture progress manually
print("Performing initial optimization steps...")
opt.maximize(init_points=5, n_iter=0)  # Only initial points

# Capture progress after initial points
points_x = [p['params']['x'] for p in opt.res]
points_y = [p['params']['y'] for p in opt.res]
values = [p['target'] for p in opt.res]

best_value = max(values)
best_index = values.index(best_value)
best_x = points_x[best_index]
best_y = points_y[best_index]

bo_progress.append({
    'iteration': len(values),
    'best_value': best_value,
    'best_x': best_x,
    'best_y': best_y,
    'all_x': points_x.copy(),
    'all_y': points_y.copy(),
    'all_values': values.copy()
})

# Continue with remaining iterations, capturing progress every 5 iterations
remaining_iterations = 25
print(f"Continuing with {remaining_iterations} iterations...")

for i in range(0, remaining_iterations, 5):
    current_iterations = min(5, remaining_iterations - i)
    opt.maximize(init_points=0, n_iter=current_iterations)

    # Capture progress
    points_x = [p['params']['x'] for p in opt.res]
    points_y = [p['params']['y'] for p in opt.res]
    values = [p['target'] for p in opt.res]

    best_value = max(values)
    best_index = values.index(best_value)
    current_best_x = points_x[best_index]
    current_best_y = points_y[best_index]

    bo_progress.append({
        'iteration': len(values),
        'best_value': best_value,
        'best_x': current_best_x,
        'best_y': current_best_y,
        'all_x': points_x.copy(),
        'all_y': points_y.copy(),
        'all_values': values.copy()
    })

    print(f"  Progress: {len(values)}/30 evaluations, Best value: {best_value:.6f}")

print("Optimization completed!")

# Step 9: Analyze the results
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"Best parameters found: {opt.max['params']}")
print(f"Best objective value: {opt.max['target']}")

# Get the final projection of the best point
best_x = opt.max['params']['x']
best_y = opt.max['params']['y']

optimal_z = torch.tensor([[best_x, best_y]], dtype=torch.float32)
with torch.no_grad():
    optimal_decoder_output = model.decode(optimal_z)
    final_projection = pca.transform(optimal_decoder_output.numpy())

print(f"\nOptimal point in autoencoder latent space: ({best_x:.4f}, {best_y:.4f})")
print(f"Final projection with PCA: ({final_projection[0,0]:.4f}, {final_projection[0,1]:.4f})")

# Verify the Rosenbrock function at the projected point
final_proj = final_projection[0]
rosenbrock_value = (1 - final_proj[0])**2 + 100*(final_proj[1] - final_proj[0]**2)**2
print(f"Rosenbrock function value: {rosenbrock_value:.6f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

if not VISUALIZATION_AVAILABLE:
    print(f"\n{'='*50}")
    print("VISUALIZATIONS NOT AVAILABLE")
    print(f"{'='*50}")
    print("To see the plots, install:")
    print("pip install matplotlib seaborn")
    print("\nNumerical results are available above.")
else:
    print(f"\n{'='*50}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*50}")

    # Function to evaluate Rosenbrock on a grid
    def evaluate_rosenbrock_surface(x_min=-3, x_max=3, y_min=-3, y_max=3, resolution=50):
        """Evaluates the objective function on a grid for visualization"""
        x_vals = np.linspace(x_min, x_max, resolution)
        y_vals = np.linspace(y_min, y_max, resolution)
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

        Z = np.zeros_like(X_mesh)

        # Evaluate the function at each point in the grid
        print(f"Evaluating surface ({resolution}x{resolution} points)...")
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = rosenbrock_pca_objective(X_mesh[i, j], Y_mesh[i, j])
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{resolution} rows completed")

        return X_mesh, Y_mesh, Z

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Objective function surface with BO trajectory
    print("Generating objective function surface...")
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X_mesh, Y_mesh, Z = evaluate_rosenbrock_surface(resolution=25)  # Reduced for speed

    # Plot the surface
    surf = ax1.plot_surface(X_mesh, Y_mesh, Z, cmap='viridis', alpha=0.7)

    # Plot BO trajectory
    if bo_progress:
        all_x = bo_progress[-1]['all_x']
        all_y = bo_progress[-1]['all_y']
        all_z = bo_progress[-1]['all_values']

        ax1.scatter(all_x, all_y, all_z, c='red', s=50, alpha=0.8, label='Evaluated points')
        ax1.scatter([best_x], [best_y], [opt.max['target']], c='black', s=200,
                   marker='*', label='Best point')

    ax1.set_xlabel('X (Latent Space)')
    ax1.set_ylabel('Y (Latent Space)')
    ax1.set_zlabel('Objective Value')
    ax1.set_title('Objective Function Surface\nwith BO Trajectory')
    ax1.legend()

    # 2. 2D Heatmap with trajectory
    ax2 = fig.add_subplot(2, 3, 2)
    print("Generating 2D heatmap...")
    X_mesh_2d, Y_mesh_2d, Z_2d = evaluate_rosenbrock_surface(resolution=40)

    # Create the heatmap
    contour = ax2.contourf(X_mesh_2d, Y_mesh_2d, Z_2d, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)

    # Plot BO trajectory
    if bo_progress:
        all_x = bo_progress[-1]['all_x']
        all_y = bo_progress[-1]['all_y']

        # Connect points in evaluation order
        ax2.plot(all_x, all_y, 'r-', alpha=0.6, linewidth=2, label='BO Trajectory')
        ax2.scatter(all_x, all_y, c='red', s=30, alpha=0.8, zorder=5)

        # Number the first and last points for clarity
        for i, (x, y) in enumerate(zip(all_x[:5], all_y[:5])):  # First 5
            ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold')
        for i, (x, y) in enumerate(zip(all_x[-3:], all_y[-3:]), len(all_x)-3):  # Last 3
            ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold')


        # Highlight the best point
        ax2.scatter([best_x], [best_y], c='black', s=200, marker='*',
                   label=f'Best point', zorder=10)

    ax2.set_xlabel('X (Latent Space)')
    ax2.set_ylabel('Y (Latent Space)')
    ax2.set_title('Heatmap with BO Trajectory')
    ax2.legend()

    # 3. Best value convergence
    ax3 = fig.add_subplot(2, 3, 3)
    if bo_progress:
        iterations = [p['iteration'] for p in bo_progress]
        best_values = [p['best_value'] for p in bo_progress]

        ax3.plot(iterations, best_values, 'b-o', linewidth=2, markersize=6)
        ax3.axhline(y=opt.max['target'], color='r', linestyle='--',
                   label=f'Best value: {opt.max["target"]:.4f}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Best Objective Value')
        ax3.set_title('Bayesian Optimization Convergence')
        ax3.legend()

    # 4. PCA Projected Space
    ax4 = fig.add_subplot(2, 3, 4)

    # Create a grid in the autoencoder latent space and see their PCA projections
    print("Generating PCA projected space...")
    x_vals = np.linspace(-3, 3, 15)  # Reduced for speed
    y_vals = np.linspace(-3, 3, 15)
    projections_x = []
    projections_y = []

    for x in x_vals:
        for y in y_vals:
            z = torch.tensor([[x, y]], dtype=torch.float32)
            with torch.no_grad():
                decoder_output = model.decode(z)
                projection = pca.transform(decoder_output.numpy())
                projections_x.append(projection[0, 0])
                projections_y.append(projection[0, 1])

    ax4.scatter(projections_x, projections_y, alpha=0.6, s=20, c='lightblue', label='PCA projected space')

    # Projection of the optimal point
    proj_x = final_projection[0, 0]
    proj_y = final_projection[0, 1]
    ax4.scatter([proj_x], [proj_y], c='red', s=200, marker='*', label='Optimal projected point')

    # Theoretical optimal point of Rosenbrock
    ax4.scatter([1], [1], c='green', s=100, marker='x', linewidth=3,
               label='Theoretical RB optimum (1,1)')

    ax4.set_xlabel('First PCA Component')
    ax4.set_ylabel('Second PCA Component')
    ax4.set_title('PCA Projected Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Classical Rosenbrock function for comparison
    ax5 = fig.add_subplot(2, 3, 5)
    x_rb = np.linspace(-2, 2, 50)
    y_rb = np.linspace(-1, 3, 50)
    X_rb, Y_rb = np.meshgrid(x_rb, y_rb)
    Z_rb = (1 - X_rb)**2 + 100*(Y_rb - X_rb**2)**2

    # Use logarithmic scale for better visualization
    try:
        contour_rb = ax5.contour(X_rb, Y_rb, Z_rb, levels=np.logspace(0, 3, 20), norm=LogNorm())
        ax5.clabel(contour_rb, inline=True, fontsize=8)
    except:
        # If LogNorm fails, use normal contour
        contour_rb = ax5.contour(X_rb, Y_rb, Z_rb, levels=20)

    # Theoretical optimal point
    ax5.scatter([1], [1], c='red', s=200, marker='*', label='Optimum (1,1)')

    # Our projected point
    ax5.scatter([proj_x], [proj_y], c='blue', s=100, marker='o',
               label=f'Our result ({proj_x:.3f}, {proj_y:.3f})')

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('Classical Rosenbrock Function\n(for comparison)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Distribution of evaluated points + PCA Info
    ax6 = fig.add_subplot(2, 3, 6)

    # Create subplots within the subplot to show histogram and PCA components
    ax6_hist = ax6
    if bo_progress:
        all_values = bo_progress[-1]['all_values']

        # Smaller histogram
        n, bins, patches = ax6_hist.hist(all_values, bins=12, alpha=0.7,
                                        color='skyblue', edgecolor='black')
        ax6_hist.axvline(x=opt.max['target'], color='red', linestyle='--', linewidth=2,
                        label=f'Best value: {opt.max["target"]:.4f}')
        ax6_hist.set_xlabel('Objective Value')
        ax6_hist.set_ylabel('Frequency')
        ax6_hist.set_title('Distribution of Values + PCA Info')
        ax6_hist.legend()
        ax6_hist.grid(True, alpha=0.3)

        # Add text with PCA information
        pca_info = f"PCA Info:\nComp. 1: {pca.explained_variance_ratio_[0]:.3f}\nComp. 2: {pca.explained_variance_ratio_[1]:.3f}\nTotal: {pca.explained_variance_ratio_.sum():.3f}"
        ax6_hist.text(0.02, 0.95, pca_info, transform=ax6_hist.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print("✓ Visualizations generated successfully!")

# Final numerical summary
print(f"\n{'='*60}")
print("COMPARATIVE SUMMARY")
print(f"{'='*60}")
print(f"Theoretical Rosenbrock optimum point: (1.0, 1.0)")
print(f"Theoretical Rosenbrock optimum value: 0.0")
print(f"")
print(f"Our point in latent space: ({best_x:.4f}, {best_y:.4f})")
print(f"Projection with PCA: ({proj_x:.4f}, {proj_y:.4f})")
print(f"Rosenbrock value obtained: {rosenbrock_value:.6f}")
print(f"")
print(f"PCA Information:")
print(f"  Explained variance comp. 1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  Explained variance comp. 2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"")
print(f"Distance to theoretical optimum: {np.sqrt((proj_x-1)**2 + (proj_y-1)**2):.4f}")
print(f"Achieved improvement: {(1000 - rosenbrock_value)/1000*100:.2f}% (compared to a typical value)")
print(f"Total evaluations: {len(opt.res)}")