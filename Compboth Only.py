import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import time

# Check and import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette("husl")
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries loaded successfully")
except ImportError as e:
    print(f"⚠ Error importing visualization libraries: {e}")
    print("Install with: pip install matplotlib seaborn")
    VISUALIZATION_AVAILABLE = False

# ============================================================================
# CONFIGURATION AND DATA
# ============================================================================

# Generate synthetic data
np.random.seed(42)
torch.manual_seed(42)

data = np.random.uniform(-6, 6, size=(1000, 2))

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
X_tensor = torch.tensor(X, dtype=torch.float32)

print(f"Generated data: {X.shape}")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class FirstAutoencoder(nn.Module):
    """First Autoencoder: 20D -> 2D -> 20D"""
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

class SecondAutoencoder(nn.Module):
    """Second Autoencoder for dimensionality reduction"""
    def __init__(self, input_dim=20, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 15),
            nn.ReLU(),
            nn.Linear(15, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# ============================================================================
# TRAINING THE FIRST AUTOENCODER (COMMON FOR BOTH APPROACHES)
# ============================================================================

print("\n" + "="*60)
print("TRAINING FIRST AUTOENCODER")
print("="*60)

model_1 = FirstAutoencoder()
criterion = nn.MSELoss()
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.01)

start_training_time = time.time()

for epoch in range(1000):
    output = model_1(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer_1.zero_grad()
    loss.backward()
    optimizer_1.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/1000, Loss: {loss.item():.6f}")

time_first_ae = time.time() - start_training_time
print(f"First autoencoder trained in {time_first_ae:.2f} seconds")

# Get decoder outputs from the first autoencoder
with torch.no_grad():
    decoder_outputs_1 = model_1.decode(model_1.encode(X_tensor))
    decoder_outputs_1_np = decoder_outputs_1.numpy()

print(f"Shape of decoder outputs: {decoder_outputs_1.shape}")

# ============================================================================
# APPROACH 1: USE PCA FOR DIMENSIONALITY REDUCTION
# ============================================================================

print("\n" + "="*60)
print("APPROACH 1: REDUCTION WITH PCA")
print("="*60)

start_pca_time = time.time()

# Fit PCA
pca = PCA(n_components=2)
pca.fit(decoder_outputs_1_np)

# Get principal components
pca_projections = pca.transform(decoder_outputs_1_np)

# Calculate explained variance
explained_variance_pca = pca.explained_variance_ratio_
print(f"Explained variance by PCA: {explained_variance_pca}")
print(f"Total explained variance: {np.sum(explained_variance_pca):.4f}")

time_pca = time.time() - start_pca_time

# Objective function using PCA
def rosenbrock_objective_pca(x, y):
    # Point in the latent space of the first autoencoder
    z = torch.tensor([[x, y]], dtype=torch.float32)

    # Decode with the first autoencoder
    with torch.no_grad():
        decoder_output_1 = model_1.decode(z)

        # Project with PCA
        projection = pca.transform(decoder_output_1.numpy())
        proj = projection[0]

    # Rosenbrock function (negative because we want to maximize)
    return -((1 - proj[0])**2 + 100*(proj[1] - proj[0]**2)**2)

# ============================================================================
# APPROACH 2: USE SECOND AUTOENCODER FOR REDUCTION
# ============================================================================

print("\n" + "="*60)
print("APPROACH 2: REDUCTION WITH AUTOENCODER")
print("="*60)

start_ae_time = time.time()

# Train second autoencoder
model_2 = SecondAutoencoder()
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.01)

for epoch in range(800):
    output = model_2(decoder_outputs_1)
    loss = criterion(output, decoder_outputs_1)
    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/800, Loss: {loss.item():.6f}")

time_ae = time.time() - start_ae_time

# Evaluate reconstruction quality of the second autoencoder
with torch.no_grad():
    ae_reconstructions = model_2(decoder_outputs_1)
    mse_ae = nn.MSELoss()(ae_reconstructions, decoder_outputs_1)
    print(f"MSE of second AE reconstruction: {mse_ae.item():.6f}")

# Objective function using second autoencoder
def rosenbrock_objective_ae(x, y):
    # Point in the latent space of the first autoencoder
    z = torch.tensor([[x, y]], dtype=torch.float32)

    # Decode with the first autoencoder
    with torch.no_grad():
        decoder_output_1 = model_1.decode(z)

        # Encode with the second autoencoder to get the 2D projection
        projection = model_2.encode(decoder_output_1)
        proj = projection.numpy()[0]

    # Rosenbrock function (negative because we want to maximize)
    return -((1 - proj[0])**2 + 100*(proj[1] - proj[0]**2)**2)

# ============================================================================
# BAYESIAN OPTIMIZATION - PCA APPROACH
# ============================================================================

print("\n" + "="*60)
print("BAYESIAN OPTIMIZATION - PCA APPROACH")
print("="*60)

start_opt_pca_time = time.time()

opt_pca = BayesianOptimization(
    f=rosenbrock_objective_pca,
    pbounds={'x': (-3, 3), 'y': (-3, 3)},
    verbose=1,
    random_state=42
)

opt_pca.maximize(init_points=5, n_iter=25)

time_opt_pca = time.time() - start_opt_pca_time

print(f"PCA optimization completed in {time_opt_pca:.2f} seconds")
print(f"Best PCA parameters: {opt_pca.max['params']}")
print(f"Best PCA value: {opt_pca.max['target']}")

# Get final PCA projection
best_x_pca = opt_pca.max['params']['x']
best_y_pca = opt_pca.max['params']['y']
optimal_z_pca = torch.tensor([[best_x_pca, best_y_pca]], dtype=torch.float32)

with torch.no_grad():
    decoder_output_1_pca = model_1.decode(optimal_z_pca)
    final_projection_pca = pca.transform(decoder_output_1_pca.numpy())[0]

rosenbrock_value_pca = (1 - final_projection_pca[0])**2 + 100*(final_projection_pca[1] - final_projection_pca[0]**2)**2

# ============================================================================
# BAYESIAN OPTIMIZATION - AUTOENCODER APPROACH
# ============================================================================

print("\n" + "="*60)
print("BAYESIAN OPTIMIZATION - AUTOENCODER APPROACH")
print("="*60)

start_opt_ae_time = time.time()

opt_ae = BayesianOptimization(
    f=rosenbrock_objective_ae,
    pbounds={'x': (-3, 3), 'y': (-3, 3)},
    verbose=1,
    random_state=42
)

opt_ae.maximize(init_points=5, n_iter=25)

time_opt_ae = time.time() - start_opt_ae_time

print(f"AE optimization completed in {time_opt_ae:.2f} seconds")
print(f"Best AE parameters: {opt_ae.max['params']}")
print(f"Best AE value: {opt_ae.max['target']}")

# Get final AE projection
best_x_ae = opt_ae.max['params']['x']
best_y_ae = opt_ae.max['params']['y']
optimal_z_ae = torch.tensor([[best_x_ae, best_y_ae]], dtype=torch.float32)

with torch.no_grad():
    decoder_output_1_ae = model_1.decode(optimal_z_ae)
    final_projection_ae = model_2.encode(decoder_output_1_ae).numpy()[0]

rosenbrock_value_ae = (1 - final_projection_ae[0])**2 + 100*(final_projection_ae[1] - final_projection_ae[0]**2)**2

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DETAILED COMPARATIVE ANALYSIS")
print("="*80)

# Time metrics
print("EXECUTION TIMES:")
print(f"  First AE training: {time_first_ae:.2f} seconds")
print(f"  PCA fitting: {time_pca:.2f} seconds")
print(f"  Second AE training: {time_ae:.2f} seconds")
print(f"  PCA optimization: {time_opt_pca:.2f} seconds")
print(f"  AE optimization: {time_opt_ae:.2f} seconds")
print(f"  Total PCA time: {time_first_ae + time_pca + time_opt_pca:.2f} seconds")
print(f"  Total AE time: {time_first_ae + time_ae + time_opt_ae:.2f} seconds")

print("\nOPTIMIZATION PERFORMANCE:")
print(f"  Best PCA objective value: {opt_pca.max['target']:.6f}")
print(f"  Best AE objective value:  {opt_ae.max['target']:.6f}")
print(f"  Difference: {opt_ae.max['target'] - opt_pca.max['target']:.6f}")

print("\nFINAL PROJECTIONS:")
print(f"  PCA Projection: ({final_projection_pca[0]:.4f}, {final_projection_pca[1]:.4f})")
print(f"  AE Projection:  ({final_projection_ae[0]:.4f}, {final_projection_ae[1]:.4f})")
print(f"  Theoretical optimum: (1.0000, 1.0000)")

print("\nROSENBROCK FUNCTION VALUE:")
print(f"  Value with PCA: {rosenbrock_value_pca:.6f}")
print(f"  Value with AE: {rosenbrock_value_ae:.6f}")
print(f"  Optimal value:  0.000000")

# Distances to optimum
dist_pca = np.sqrt((final_projection_pca[0] - 1)**2 + (final_projection_pca[1] - 1)**2)
dist_ae = np.sqrt((final_projection_ae[0] - 1)**2 + (final_projection_ae[1] - 1)**2)

print("\nDISTANCE TO THE THEORETICAL OPTIMUM:")
print(f"  PCA Distance: {dist_pca:.4f}")
print(f"  AE Distance: {dist_ae:.4f}")

# Determine winner
if opt_ae.max['target'] > opt_pca.max['target']:
    winner = "AUTOENCODER"
    advantage = opt_ae.max['target'] - opt_pca.max['target']
else:
    winner = "PCA"
    advantage = opt_pca.max['target'] - opt_ae.max['target']

print(f"\nWINNER: {winner}")
print(f"Advantage: {advantage:.6f} in objective value")

# Efficiency analysis
efficiency_pca = opt_pca.max['target'] / (time_first_ae + time_pca + time_opt_pca)
efficiency_ae = opt_ae.max['target'] / (time_first_ae + time_ae + time_opt_ae)

print(f"\nEFFICIENCY (objective value / total time):")
print(f"  PCA Efficiency: {efficiency_pca:.6f}")
print(f"  AE Efficiency:  {efficiency_ae:.6f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

if VISUALIZATION_AVAILABLE:
    print("\n" + "="*60)
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print("="*60)

    fig = plt.figure(figsize=(20, 16))

    # 1. Convergence Comparison
    ax1 = fig.add_subplot(2, 4, 1)

    # Extract convergence values
    pca_values = [res['target'] for res in opt_pca.res]
    ae_values = [res['target'] for res in opt_ae.res]

    best_pca = [max(pca_values[:i+1]) for i in range(len(pca_values))]
    best_ae = [max(ae_values[:i+1]) for i in range(len(ae_values))]

    ax1.plot(range(1, len(best_pca)+1), best_pca, 'b-o', label='PCA', linewidth=2, markersize=4)
    ax1.plot(range(1, len(best_ae)+1), best_ae, 'r-s', label='Autoencoder', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Objective Value')
    ax1.set_title('Convergence: PCA vs Autoencoder')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of evaluated points
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.hist(pca_values, bins=15, alpha=0.6, label='PCA', color='blue', density=True)
    ax2.hist(ae_values, bins=15, alpha=0.6, label='Autoencoder', color='red', density=True)
    ax2.axvline(opt_pca.max['target'], color='blue', linestyle='--', label=f'Best PCA: {opt_pca.max["target"]:.3f}')
    ax2.axvline(opt_ae.max['target'], color='red', linestyle='--', label=f'Best AE: {opt_ae.max["target"]:.3f}')
    ax2.set_xlabel('Objective Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Evaluated Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Projections in 2D space - PCA
    ax3 = fig.add_subplot(2, 4, 3)

    # Generate mesh of PCA projections
    x_vals = np.linspace(-3, 3, 20)
    y_vals = np.linspace(-3, 3, 20)
    pca_projections_x, pca_projections_y = [], []

    for x in x_vals:
        for y in y_vals:
            z = torch.tensor([[x, y]], dtype=torch.float32)
            with torch.no_grad():
                output = model_1.decode(z)
                proj = pca.transform(output.numpy())[0]
                pca_projections_x.append(proj[0])
                pca_projections_y.append(proj[1])

    ax3.scatter(pca_projections_x, pca_projections_y, alpha=0.6, s=15, c='lightblue')
    ax3.scatter([final_projection_pca[0]], [final_projection_pca[1]], c='blue', s=200, marker='*', label='Optimal PCA')
    ax3.scatter([1], [1], c='green', s=100, marker='x', linewidth=3, label='Theoretical optimum')
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    ax3.set_title('Projected Space - PCA')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Projections in 2D space - Autoencoder
    ax4 = fig.add_subplot(2, 4, 4)

    # Generate mesh of AE projections
    ae_projections_x, ae_projections_y = [], []

    for x in x_vals:
        for y in y_vals:
            z = torch.tensor([[x, y]], dtype=torch.float32)
            with torch.no_grad():
                output = model_1.decode(z)
                proj = model_2.encode(output).numpy()[0]
                ae_projections_x.append(proj[0])
                ae_projections_y.append(proj[1])

    ax4.scatter(ae_projections_x, ae_projections_y, alpha=0.6, s=15, c='lightcoral')
    ax4.scatter([final_projection_ae[0]], [final_projection_ae[1]], c='red', s=200, marker='*', label='Optimal AE')
    ax4.scatter([1], [1], c='green', s=100, marker='x', linewidth=3, label='Theoretical optimum')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.set_title('Projected Space - Autoencoder')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Rosenbrock function with optimal points
    ax5 = fig.add_subplot(2, 4, 5)
    x_rb = np.linspace(-2, 2.5, 100)
    y_rb = np.linspace(-1, 3, 100)
    X_rb, Y_rb = np.meshgrid(x_rb, y_rb)
    Z_rb = (1 - X_rb)**2 + 100*(Y_rb - X_rb**2)**2

    contour = ax5.contour(X_rb, Y_rb, Z_rb, levels=np.logspace(0, 3, 15), norm=LogNorm(), colors='gray', alpha=0.6)
    ax5.scatter([1], [1], c='green', s=200, marker='x', linewidth=4, label='Theoretical optimum', zorder=5)
    ax5.scatter([final_projection_pca[0]], [final_projection_pca[1]], c='blue', s=150, marker='*',
                 label=f'PCA ({final_projection_pca[0]:.3f}, {final_projection_pca[1]:.3f})', zorder=5)
    ax5.scatter([final_projection_ae[0]], [final_projection_ae[1]], c='red', s=150, marker='s',
                 label=f'AE ({final_projection_ae[0]:.3f}, {final_projection_ae[1]:.3f})', zorder=5)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('Rosenbrock Function - Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Performance metrics
    ax6 = fig.add_subplot(2, 4, 6)
    metrics = ['Objective Value', 'Time (s)', 'Efficiency', 'Dist. to Optimum']
    pca_values_norm = [
        opt_pca.max['target'],
        time_first_ae + time_pca + time_opt_pca,
        efficiency_pca * 1000,  # Scaled for visualization
        dist_pca
    ]
    ae_values_norm = [
        opt_ae.max['target'],
        time_first_ae + time_ae + time_opt_ae,
        efficiency_ae * 1000,   # Scaled for visualization
        dist_ae
    ]

    x_pos = np.arange(len(metrics))
    width = 0.35

    ax6.bar(x_pos - width/2, pca_values_norm, width, label='PCA', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, ae_values_norm, width, label='Autoencoder', color='red', alpha=0.7)
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Values (normalized)')
    ax6.set_title('Comparison of Metrics')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Optimization trajectories in latent space
    ax7 = fig.add_subplot(2, 4, 7)

    # Points evaluated by PCA
    pca_points_x = [res['params']['x'] for res in opt_pca.res]
    pca_points_y = [res['params']['y'] for res in opt_pca.res]

    # Points evaluated by AE
    ae_points_x = [res['params']['x'] for res in opt_ae.res]
    ae_points_y = [res['params']['y'] for res in opt_ae.res]

    ax7.plot(pca_points_x, pca_points_y, 'b-o', alpha=0.6, markersize=4, label='PCA Trajectory')
    ax7.plot(ae_points_x, ae_points_y, 'r-s', alpha=0.6, markersize=4, label='Autoencoder Trajectory')
    ax7.scatter([best_x_pca], [best_y_pca], c='blue', s=200, marker='*', label='Best PCA', zorder=5)
    ax7.scatter([best_x_ae], [best_y_ae], c='red', s=200, marker='*', label='Best AE', zorder=5)
    ax7.set_xlabel('X (Latent Space)')
    ax7.set_ylabel('Y (Latent Space)')
    ax7.set_title('Trajectories in Latent Space')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Text summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    summary_text = f"""
COMPARATIVE SUMMARY

WINNER: {winner}
Advantage: {advantage:.6f}

OBJECTIVE VALUES:
• PCA: {opt_pca.max['target']:.6f}
• AE: {opt_ae.max['target']:.6f}

TOTAL TIME:
• PCA: {time_first_ae + time_pca + time_opt_pca:.1f}s
• AE: {time_first_ae + time_ae + time_opt_ae:.1f}s

DISTANCE TO OPTIMUM:
• PCA: {dist_pca:.4f}
• AE: {dist_ae:.4f}

EFFICIENCY:
• PCA: {efficiency_pca:.6f}
• AE: {efficiency_ae:.6f}

PCA EXPLAINED VARIANCE:
{np.sum(explained_variance_pca):.1%}
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print("✓ Comparative visualizations generated successfully!")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

print("1. PERFORMANCE:")
if winner == "AUTOENCODER":
    print("   • The autoencoder outperformed PCA in optimization")
    print(f"   • Advantage of {advantage:.6f} in objective value")
else:
    print("   • PCA outperformed the autoencoder in optimization")
    print(f"   • Advantage of {advantage:.6f} in objective value")

print(f"\n2. TEMPORAL EFFICIENCY:")
if efficiency_ae > efficiency_pca:
    print("   • The autoencoder is more efficient considering time vs performance")
else:
    print("   • PCA is more efficient considering time vs performance")

print(f"\n3. EXPLAINED VARIANCE:")
print(f"   • PCA explains {np.sum(explained_variance_pca):.1%} of the variance")
print("   • The autoencoder does not have a direct equivalent metric")

print(f"\n4. RECONSTRUCTION QUALITY:")
# Compare reconstructions
with torch.no_grad():
    # PCA Reconstruction
    all_pca_projections = pca.transform(decoder_outputs_1_np)
    pca_reconstructions = pca.inverse_transform(all_pca_projections)
    mse_pca = mean_squared_error(decoder_outputs_1_np, pca_reconstructions)

    # AE Reconstruction
    all_ae_reconstructions = model_2(decoder_outputs_1)
    mse_ae_recon = nn.MSELoss()(all_ae_reconstructions, decoder_outputs_1).item()

print(f"   • PCA reconstruction error: {mse_pca:.6f}")
print(f"   • AE reconstruction error: {mse_ae_recon:.6f}")

if mse_pca < mse_ae_recon:
    print("   • PCA has better reconstruction")
else:
    print("   • The autoencoder has better reconstruction")

print(f"\n5. RECOMMENDATIONS:")
if winner == "AUTOENCODER":
    print("   • Use autoencoder when:")
    print("     - Sufficient time is available for training")
    print("     - Maximum performance in optimization is required")
    print("     - Data has complex non-linear patterns")
    print("   • Use PCA when:")
    print("     - Quick implementation is needed")
    print("     - Computational resources are limited")
    print("     - Interpretability of components is required")
else:
    print("   • Use PCA when:")
    print("     - A fast and efficient solution is needed")
    print("     - Data has a primarily linear structure")
    print("     - Interpretability is required")
    print("   • Use autoencoder when:")
    print("     - Data has complex non-linear patterns")
    print("     - More time can be invested in training")

print(f"\n6. TECHNICAL CHARACTERISTICS:")
print(f"   • PCA principal components: {pca.components_.shape}")
print(f"   • Second AE parameters: {sum(p.numel() for p in model_2.parameters())}")
print(f"   • PCA fitting time: {time_pca:.2f}s")
print(f"   • AE training time: {time_ae:.2f}s")

print(f"\n7. STABILITY ANALYSIS:")
# Check how consistent the results are
print(f"   • Standard deviation of PCA values: {np.std(pca_values):.6f}")
print(f"   • Standard deviation of AE values: {np.std(ae_values):.6f}")

if np.std(pca_values) < np.std(ae_values):
    print("   • PCA shows greater stability in exploration")
else:
    print("   • The autoencoder shows greater stability in exploration")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED")
print("="*80)

# ============================================================================
# ADDITIONAL ANALYSIS: GENERALIZATION CAPACITY
# ============================================================================

print("\n" + "="*60)
print("GENERALIZATION ANALYSIS")
print("="*60)

# Generate new test data
test_data = np.random.uniform(-6, 6, size=(200, 2))
X_test = apply_functions(test_data)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Get decoder outputs for test data
with torch.no_grad():
    decoder_test_outputs = model_1.decode(model_1.encode(X_test_tensor))
    decoder_test_outputs_np = decoder_test_outputs.numpy()

# Evaluate reconstructions on test data
# PCA
test_pca_projections = pca.transform(decoder_test_outputs_np)
test_pca_reconstructions = pca.inverse_transform(test_pca_projections)
mse_test_pca = mean_squared_error(decoder_test_outputs_np, test_pca_reconstructions)

# Autoencoder
with torch.no_grad():
    test_ae_reconstructions = model_2(decoder_test_outputs)
    mse_test_ae = nn.MSELoss()(test_ae_reconstructions, decoder_test_outputs).item()

print(f"GENERALIZATION CAPACITY:")
print(f"   • PCA reconstruction error (new data): {mse_test_pca:.6f}")
print(f"   • AE reconstruction error (new data): {mse_test_ae:.6f}")
print(f"   • PCA Ratio (test/train): {mse_test_pca/mse_pca:.3f}")
print(f"   • AE Ratio (test/train): {mse_test_ae/mse_ae_recon:.3f}")

if mse_test_pca/mse_pca < mse_test_ae/mse_ae_recon:
    print("   • PCA generalizes better (less degradation on new data)")
else:
    print("   • The autoencoder generalizes better")

# ============================================================================
# ROBUSTNESS TEST WITH NOISE
# ============================================================================

print(f"\nROBUSTNESS TEST:")

# Add noise to data
noise = np.random.normal(0, 0.1, decoder_outputs_1_np.shape)
noisy_data = decoder_outputs_1_np + noise
noisy_data_tensor = torch.tensor(noisy_data, dtype=torch.float32)

# Evaluate with noise
# PCA
try:
    noisy_pca_projections = pca.transform(noisy_data)
    noisy_pca_reconstructions = pca.inverse_transform(noisy_pca_projections)
    mse_noisy_pca = mean_squared_error(decoder_outputs_1_np, noisy_pca_reconstructions)
except:
    mse_noisy_pca = float('inf')

# Autoencoder
with torch.no_grad():
    try:
        noisy_ae_reconstructions = model_2(noisy_data_tensor)
        mse_noisy_ae = nn.MSELoss()(noisy_ae_reconstructions, decoder_outputs_1).item()
    except:
        mse_noisy_ae = float('inf')

print(f"   • PCA error with noise: {mse_noisy_pca:.6f}")
print(f"   • AE error with noise: {mse_noisy_ae:.6f}")

if mse_noisy_pca < mse_noisy_ae:
    print("   • PCA is more robust to noise")
else:
    print("   • The autoencoder is more robust to noise")

# ============================================================================
# FINAL EXECUTIVE SUMMARY
# ============================================================================

print("\n" + "#"*80)
print("EXECUTIVE SUMMARY")
print("#"*80)

print(f"\nWINNING METHOD: {winner}")
print(f"OPTIMIZATION ADVANTAGE: {advantage:.6f}")

print(f"\nOVERALL SCORE:")
pca_score = 0
ae_score = 0

# Criterion 1: Objective value
if opt_ae.max['target'] > opt_pca.max['target']:
    ae_score += 2
    print(f"   • Objective value: AUTOENCODER (+2)")
else:
    pca_score += 2
    print(f"   • Objective value: PCA (+2)")

# Criterion 2: Execution time
total_time_pca = time_first_ae + time_pca + time_opt_pca
total_time_ae = time_first_ae + time_ae + time_opt_ae

if total_time_pca < total_time_ae:
    pca_score += 1
    print(f"   • Temporal efficiency: PCA (+1)")
else:
    ae_score += 1
    print(f"   • Temporal efficiency: AUTOENCODER (+1)")

# Criterion 3: Reconstruction
if mse_pca < mse_ae_recon:
    pca_score += 1
    print(f"   • Reconstruction quality: PCA (+1)")
else:
    ae_score += 1
    print(f"   • Reconstruction quality: AUTOENCODER (+1)")

# Criterion 4: Generalization
if mse_test_pca/mse_pca < mse_test_ae/mse_ae_recon:
    pca_score += 1
    print(f"   • Generalization: PCA (+1)")
else:
    ae_score += 1
    print(f"   • Generalization: AUTOENCODER (+1)")

# Criterion 5: Robustness
if mse_noisy_pca < mse_noisy_ae:
    pca_score += 1
    print(f"   • Robustness to noise: PCA (+1)")
else:
    ae_score += 1
    print(f"   • Robustness to noise: AUTOENCODER (+1)")

print(f"\nFINAL SCORE:")
print(f"   • PCA: {pca_score}/6 points")
print(f"   • AUTOENCODER: {ae_score}/6 points")

if ae_score > pca_score:
    recommended_method = "AUTOENCODER"
elif pca_score > ae_score:
    recommended_method = "PCA"
else:
    recommended_method = "TIE - Depends on the use case"

print(f"\nFINAL RECOMMENDATION: {recommended_method}")

print(f"\nCONSIDERATIONS FOR DECISION:")
print("• If time is critical: PCA")
print("• If performance is critical: evaluate case by case")
print("• If interpretability is needed: PCA")
print("• If data is highly non-linear: Autoencoder")
print("• If resources are limited: PCA")
print("• If maximum precision is required: try both methods")

print("\n" + "#"*80)
print("COMPARATIVE EXPERIMENT COMPLETED")
print("#"*80)
