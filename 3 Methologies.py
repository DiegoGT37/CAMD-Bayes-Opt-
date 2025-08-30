import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from bayes_opt import BayesianOptimization
# Attempting to import UpperConfidenceBound from a common location, or will define it if not found
try:
    from bayes_opt.util import UpperConfidenceBound
except ImportError:
    # If direct import fails, define a basic UCB function
    def UpperConfidenceBound(gp, X, kappa):
        mean, std = gp.predict(X, return_std=True)
        return mean + kappa * std

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import json
import random
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Check and import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette("husl")
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries loaded correctly")
except ImportError as e:
    print(f"⚠ Error importing visualization libraries: {e}")
    print("Install with: pip install matplotlib seaborn")
    VISUALIZATION_AVAILABLE = False

# ============================================================================
# AUTOENCODER CLASSES 
# ============================================================================

class FirstAutoencoder(nn.Module):
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

# Function to apply transformations to the data (from original code)
def apply_functions(x):
    return np.array([
        np.sin(x[:,0]), np.cos(x[:,1]), np.tanh(x[:,0]), np.exp(-x[:,1]**2),
        x[:,0]**2 + x[:,1]**2, np.abs(x[:,0] - x[:,1]), x[:,0] * x[:,1],
        np.log1p(np.abs(x[:,0])), np.log1p(np.abs(x[:,1])), np.sin(x[:,0]*x[:,1]),
        np.cos(x[:,0]+x[:,1]), x[:,0]**3, x[:,1]**3, np.maximum(x[:,0], x[:,1]),
        np.minimum(x[:,0], x[:,1]), np.arctan2(x[:,0], x[:,1]),
        x[:,0] / (1 + np.abs(x[:,1])), x[:,1] / (1 + np.abs(x[:,0])),
        np.exp(x[:,0]*0.1), np.exp(x[:,1]*0.1)
    ]).T

# ============================================================================
# LLM FOR OPTUNA 
# ============================================================================

class LLMOptimizationGuide:
    """
    LLM Simulator to guide optimization.
    """

    def __init__(self):
        self.optimization_history = []
        self.best_params = None
        self.best_value = float('-inf')

    def analyze_optimization_state(self, trial_history: List[Dict]) -> Dict:
        """Analyzes the current optimization state using simulated LLM logic"""

        if not trial_history:
            return {
                'strategy': 'exploration',
                'reasoning': 'Start of optimization, explore broadly',
                'suggested_region': {'x': (-2, 2), 'y': (-2, 2)},
                'confidence': 0.5
            }

        # Extract relevant information
        values = [trial['value'] for trial in trial_history]
        params = [trial['params'] for trial in trial_history]

        best_idx = np.argmax(values)
        best_trial = trial_history[best_idx]

        # Convergence analysis
        recent_trials = trial_history[-5:] if len(trial_history) >= 5 else trial_history
        recent_values = [t['value'] for t in recent_trials]

        # Pattern detection (simulating LLM reasoning)
        if len(trial_history) < 10:
            strategy = 'exploration'
            reasoning = 'Initial phase: explore different regions of the space'
        elif np.std(recent_values) < 0.1:
            strategy = 'local_search'
            reasoning = 'Convergence detected: focus local search'
        else:
            strategy = 'balanced'
            reasoning = 'Balanced search between exploration and exploitation'

        # Suggested region based on best results
        if len(trial_history) >= 3:
            top_3_trials = sorted(trial_history, key=lambda x: x['value'], reverse=True)[:3]
            x_coords = [t['params']['x'] for t in top_3_trials]
            y_coords = [t['params']['y'] for t in top_3_trials]

            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            radius = max(np.std(x_coords), np.std(y_coords), 0.5)

            suggested_region = {
                'x': (center_x - radius, center_x + radius),
                'y': (center_y - radius, center_y + radius)
            }
        else:
            suggested_region = {'x': (-3, 3), 'y': (-3, 3)}

        return {
            'strategy': strategy,
            'reasoning': reasoning,
            'suggested_region': suggested_region,
            'confidence': min(len(trial_history) / 20, 1.0),
            'best_params': best_trial['params'],
            'best_value': best_trial['value']
        }

    def suggest_next_trial(self, analysis: Dict, trial_number: int) -> Optional[Dict]:
        """Suggests parameters for the next trial based on the analysis"""

        if analysis['strategy'] == 'exploration':
            # Broad exploration biased towards promising regions
            region = analysis['suggested_region']
            return {
                'x': np.random.uniform(region['x'][0], region['x'][1]),
                'y': np.random.uniform(region['y'][0], region['y'][1]),
                'source': 'llm_exploration'
            }

        elif analysis['strategy'] == 'local_search':
            # Local search around the best point
            best_params = analysis['best_params']
            noise_scale = 0.3 * (1 - analysis['confidence'])

            return {
                'x': best_params['x'] + np.random.normal(0, noise_scale),
                'y': best_params['y'] + np.random.normal(0, noise_scale),
                'source': 'llm_local_search'
            }

        else:  # balanced
            # Mix of exploration and exploitation
            if np.random.random() < 0.6:
                # Local search
                best_params = analysis['best_params']
                noise_scale = 0.5
                return {
                    'x': best_params['x'] + np.random.normal(0, noise_scale),
                    'y': best_params['y'] + np.random.normal(0, noise_scale),
                    'source': 'llm_balanced_local'
                }
            else:
                # Exploration
                region = analysis['suggested_region']
                return {
                    'x': np.random.uniform(region['x'][0], region['x'][1]),
                    'y': np.random.uniform(region['y'][0], region['y'][1]),
                    'source': 'llm_balanced_exploration'
                }

# LLM Guided Optuna Sampler 
class LLMGuidedSampler(TPESampler):
    """Sampler that combines TPE with LLM suggestions"""

    def __init__(self, llm_guide: LLMOptimizationGuide, llm_influence: float = 0.3):
        super().__init__()
        self.llm_guide = llm_guide
        self.llm_influence = llm_influence
        self.trial_history = []

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Decide whether to use LLM suggestion or TPE
        if (len(self.trial_history) > 0 and
            np.random.random() < self.llm_influence):

            # Use LLM suggestion
            analysis = self.llm_guide.analyze_optimization_state(self.trial_history)
            suggestion = self.llm_guide.suggest_next_trial(analysis, len(self.trial_history))

            if suggestion and param_name in suggestion:
                # Ensure value is within bounds
                value = suggestion[param_name]
                if hasattr(param_distribution, 'low') and hasattr(param_distribution, 'high'):
                    value = np.clip(value, param_distribution.low, param_distribution.high)
                return value

        # Use TPE by default
        return super().sample_independent(study, trial, param_name, param_distribution)

# ============================================================================
# IMPROVED LLM-BAYESOPT CLASS 
# ============================================================================

class LLMBayesOptimizer:
    """
    Improved Bayesian Optimizer with LLM capabilities to suggest
    smarter exploration points.
    """

    def __init__(self, objective_function, bounds: Dict[str, Tuple[float, float]],
                 max_iterations: int = 30, verbose: bool = True):
        self.objective_function = objective_function
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Evaluation history
        self.history = []
        self.X_observed = []
        self.y_observed = []

        # Gaussian Process configuration
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6,
                                           normalize_y=True, n_restarts_optimizer=10)

        # Acquisition configuration
        # Using the UCB function defined or imported
        self.acquisition = lambda gp, X: UpperConfidenceBound(gp, X, kappa=2.5)


        # LLM-BayesOpt parameters
        self.llm_suggestions_weight = 0.3  # Weight of LLM suggestions
        self.exploration_factor = 0.2      # Exploration factor
        self.pattern_memory = []           # Memory of successful patterns

    def _generate_llm_suggestions(self, iteration: int) -> List[Dict[str, float]]:
        """
        Generates intelligent suggestions based on problem analysis
        and evaluation history.
        """
        suggestions = []

        if iteration < 5:
            # Early iterations: explore near the center and corners
            suggestions.extend([
                {'x': 0.0, 'y': 0.0},      # Center
                {'x': 1.0, 'y': 1.0},      # Near theoretical optimum
                {'x': -1.0, 'y': -1.0},    # Opposite corner
                {'x': 0.5, 'y': 0.5},      # Intermediate
            ])

        elif iteration < 15:
            # Mid iterations: explore based on best results
            if self.history:
                best_points = sorted(self.history, key=lambda x: x['target'], reverse=True)[:3]

                for point in best_points:
                    # Generate variations around the best points
                    for _ in range(2):
                        noise_x = np.random.normal(0, 0.3)
                        noise_y = np.random.normal(0, 0.3)

                        new_x = np.clip(point['x'] + noise_x, self.bounds['x'][0], self.bounds['x'][1])
                        new_y = np.clip(point['y'] + noise_y, self.bounds['y'][0], self.bounds['y'][1])

                        suggestions.append({'x': new_x, 'y': new_y})

        else:
            # Final iterations: intensive local refinement
            if self.history:
                best_point = max(self.history, key=lambda x: x['target'])

                # Refined local search
                for radius in [0.1, 0.05, 0.02]:
                    for angle in np.linspace(0, 2*np.pi, 8):
                        new_x = best_point['x'] + radius * np.cos(angle)
                        new_y = best_point['y'] + radius * np.sin(angle)

                        new_x = np.clip(new_x, self.bounds['x'][0], self.bounds['x'][1])
                        new_y = np.clip(new_y, self.bounds['y'][0], self.bounds['y'][1])

                        suggestions.append({'x': new_x, 'y': new_y})

        return suggestions[:10]  # Limit to 10 suggestions

    def _analyze_convergence_patterns(self) -> Dict[str, float]:
        """
        Analyzes convergence patterns to adjust strategies.
        """
        if len(self.history) < 5:
            return {'trend': 0.0, 'volatility': 1.0, 'stagnation': 0.0}

        recent_values = [h['target'] for h in self.history[-5:]]

        # Calculate trend
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        # Calculate volatility
        volatility = np.std(recent_values)

        # Detect stagnation
        stagnation = 1.0 if abs(trend) < 0.001 else 0.0

        return {
            'trend': trend,
            'volatility': volatility,
            'stagnation': stagnation
        }

    def _adaptive_acquisition_strategy(self, iteration: int) -> str:
        """
        Selects adaptive acquisition strategy.
        """
        patterns = self._analyze_convergence_patterns()

        if patterns['stagnation'] > 0.5:
            return 'exploration'  # Explore if we are stagnant
        elif patterns['trend'] > 0:
            return 'exploitation'  # Exploit if we are improving
        else:
            return 'balanced'  # Balanced strategy

    def _select_next_point(self, iteration: int) -> Dict[str, float]:
        """
        Selects the next point combining traditional BO with LLM suggestions.
        """
        # Get LLM suggestions
        llm_suggestions = self._generate_llm_suggestions(iteration)

        # Adaptive strategy
        strategy = self._adaptive_acquisition_strategy(iteration)

        if strategy == 'exploration':
            # Prioritize exploration
            kappa = 3.0
        elif strategy == 'exploitation':
            # Prioritize exploitation
            kappa = 1.0
        else:
            # Balanced
            kappa = 2.0

        # Evaluate LLM suggestions if we have enough data
        if len(self.X_observed) >= 3:
            self.gp.fit(self.X_observed, self.y_observed)

            best_llm_suggestion = None
            best_llm_value = float('-inf')

            for suggestion in llm_suggestions:
                point = np.array([[suggestion['x'], suggestion['y']]])

                # Predict value with GP
                mean, std = self.gp.predict(point, return_std=True)

                # Acquisition function UCB
                # Using the acquisition function defined or imported
                ucb_value = self.acquisition(self.gp, point)


                if ucb_value > best_llm_value:
                    best_llm_value = ucb_value
                    best_llm_suggestion = suggestion

            # Decide between LLM suggestion and intelligent random exploration
            if (best_llm_suggestion and
                random.random() < self.llm_suggestions_weight):
                return best_llm_suggestion

        # Fallback: intelligent random selection
        return self._intelligent_random_selection()

    def _intelligent_random_selection(self) -> Dict[str, float]:
        """
        Intelligent random selection based on history.
        """
        if not self.history:
            # If no history, completely random selection
            return {
                'x': np.random.uniform(self.bounds['x'][0], self.bounds['x'][1]),
                'y': np.random.uniform(self.bounds['y'][0], self.bounds['y'][1])
            }

        # Selection based on promising regions
        best_points = sorted(self.history, key=lambda x: x['target'], reverse=True)[:5]

        # Select region randomly
        base_point = random.choice(best_points)

        # Add adaptive noise
        noise_scale = 0.5 if len(self.history) < 10 else 0.2

        new_x = base_point['x'] + np.random.normal(0, noise_scale)
        new_y = base_point['y'] + np.random.normal(0, noise_scale)

        # Ensure it is within bounds
        new_x = np.clip(new_x, self.bounds['x'][0], self.bounds['x'][1])
        new_y = np.clip(new_y, self.bounds['y'][0], self.bounds['y'][1])

        return {'x': new_x, 'y': new_y}

    def optimize(self) -> Dict:
        """
        Executes LLM-BayesOpt optimization.
        """
        print("Starting LLM-BayesOpt...")

        # Initial evaluations
        initial_points = [
            {'x': 0.0, 'y': 0.0},
            {'x': 1.0, 'y': 1.0},
            {'x': -1.0, 'y': -1.0},
            {'x': 0.5, 'y': -0.5},
            {'x': -0.5, 'y': 0.5}
        ]

        for i, point in enumerate(initial_points):
            value = self.objective_function(point['x'], point['y'])

            self.history.append({
                'iteration': i + 1,
                'x': point['x'],
                'y': point['y'],
                'target': value,
                'method': 'initial'
            })

            self.X_observed.append([point['x'], point['y']])
            self.y_observed.append(value)

            if self.verbose:
                print(f"  Initial {i+1}/5: f({point['x']:.3f}, {point['y']:.3f}) = {value:.6f}")

        # Main optimization
        for iteration in range(5, self.max_iterations):
            # Select next point
            next_point = self._select_next_point(iteration)

            # Evaluate objective function
            value = self.objective_function(next_point['x'], next_point['y'])

            # Update history
            self.history.append({
                'iteration': iteration + 1,
                'x': next_point['x'],
                'y': next_point['y'],
                'target': value,
                'method': 'llm_bo'
            })

            self.X_observed.append([next_point['x'], next_point['y']])
            self.y_observed.append(value)

            if self.verbose and (iteration + 1) % 5 == 0:
                best_so_far = max(self.history, key=lambda x: x['target'])
                print(f"  Iteration {iteration + 1}: f({next_point['x']:.3f}, {next_point['y']:.3f}) = {value:.6f}")
                print(f"    Best so far: {best_so_far['target']:.6f}")

        # Get best result
        best_result = max(self.history, key=lambda x: x['target'])

        print(f"Optimization completed!")
        print(f"    Best result: f({best_result['x']:.4f}, {best_result['y']:.4f}) = {best_result['target']:.6f}")

        return {
            'best_params': {'x': best_result['x'], 'y': best_result['y']},
            'best_value': best_result['target'],
            'history': self.history,
            'n_evaluations': len(self.history)
        }

# ============================================================================
# COMMON OBJECTIVE FUNCTION FOR THE THREE METHODOLOGIES
# ============================================================================

# Autoencoder models (will be trained once globally)
modelo_1 = None
modelo_2 = None

# Objective function: Negative Rosenbrock in the space projected by AE
def objective_rosenbrock_neuronal(x, y):
    """
    Negative Rosenbrock function applied to the 2D projection
    of a point (x,y) in a latent space, after passing through two Autoencoders.
    Optuna maximizes, so we negate Rosenbrock to search for the minimum of the original.
    The global optimum of the original Rosenbrock is 0 at (1,1).
    """
    global modelo_1, modelo_2

    if modelo_1 is None or modelo_2 is None:
        raise RuntimeError("Autoencoder models have not been trained. Call `main()` first.")

    z = torch.tensor([[x, y]], dtype=torch.float32)
    with torch.no_grad():
        salida_decoder_1 = modelo_1.decode(z)
        proyeccion = modelo_2.encode(salida_decoder_1)
        proy = proyeccion.numpy()[0]

    # Standard Rosenbrock function
    rosenbrock_val = (1 - proy[0])**2 + 100*(proy[1] - proy[0]**2)**2

    # Return the negative for maximization (Optuna and BayesOpt maximize)
    return -rosenbrock_val

# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def main_comparison(n_trials: int = 50):
    global modelo_1, modelo_2 # Access global models

    print(" Starting optimizer comparison: Optuna+LLM vs LLM-BayesOpt vs Traditional BO")
    print("="*80)

    # --- Preparation: Generate Data and Train Autoencoders ---
    print(" Generating data and training Autoencoders...")
    datos = np.random.uniform(-6, 6, size=(1000, 2))
    X = apply_functions(datos)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    modelo_1 = FirstAutoencoder()
    criterio = nn.MSELoss()
    optimizador_1 = torch.optim.Adam(modelo_1.parameters(), lr=0.01)

    for epoca in range(1000):
        salida = modelo_1(X_tensor)
        perdida = criterio(salida, X_tensor)
        optimizador_1.zero_grad()
        perdida.backward()
        optimizador_1.step()
    # print(f"  First AE Lost final: {perdida.item():.6f}")

    modelo_2 = SecondAutoencoder()
    optimizador_2 = torch.optim.Adam(modelo_2.parameters(), lr=0.01)

    with torch.no_grad():
        salidas_decoder_1 = modelo_1.decode(modelo_1.encode(X_tensor))

    for epoca in range(800):
        salida = modelo_2(salidas_decoder_1)
        perdida = criterio(salida, salidas_decoder_1)
        optimizador_2.zero_grad()
        perdida.backward()
        optimizador_2.step()
    # print(f"  Second AE Lost final: {perdida.item():.6f}")
    print("✓ Autoencoders trained.")

    # --- 1. Optimization with Optuna + LLM ---
    print("\n[1/3]  Executing Optuna + LLM...")
    llm_guide_optuna = LLMOptimizationGuide()

    # Wrapper for Optuna objective function
    def objective_optuna_llm(trial):
        x = trial.suggest_float('x', -3, 3)
        y = trial.suggest_float('y', -3, 3)

        value = objective_rosenbrock_neuronal(x, y)

        # Record in history for the LLM
        if hasattr(trial, 'study'):
            sampler = trial.study.sampler
            if isinstance(sampler, LLMGuidedSampler):
                # Calculate projection for sampler history
                z_temp = torch.tensor([[x, y]], dtype=torch.float32)
                with torch.no_grad():
                    salida_decoder_1_temp = modelo_1.decode(z_temp)
                    proyeccion_temp = modelo_2.encode(salida_decoder_1_temp)
                    proy_temp = proyeccion_temp.numpy()[0]

                sampler.trial_history.append({
                    'params': {'x': x, 'y': y},
                    'value': value,
                    'proyeccion': proy_temp.tolist()
                })
        return value

    sampler_optuna = LLMGuidedSampler(llm_guide_optuna, llm_influence=0.4)
    study_optuna = optuna.create_study(
        direction='maximize',
        sampler=sampler_optuna,
        study_name='optuna_llm'
    )
    study_optuna.optimize(objective_optuna_llm, n_trials=n_trials)

    best_optuna = study_optuna.best_trial
    x_optuna, y_optuna = best_optuna.params['x'], best_optuna.params['y']
    valor_optuna = best_optuna.value

    z_optuna = torch.tensor([[x_optuna, y_optuna]], dtype=torch.float32)
    with torch.no_grad():
        salida_optuna = modelo_1.decode(z_optuna)
        proy_optuna = modelo_2.encode(salida_optuna).numpy()[0]

    print(f" Optuna + LLM completed. Best value: {valor_optuna:.6f}")

    # --- 2. Optimization with LLM-BayesOpt (Custom) ---
    print("\n[2/3]  Executing LLM-BayesOpt (Custom)...")
    llm_optimizer_custom = LLMBayesOptimizer(
        objective_function=objective_rosenbrock_neuronal,
        bounds={'x': (-3, 3), 'y': (-3, 3)},
        max_iterations=n_trials, # Same number of evaluations
        verbose=False # Silence detailed output here, summary at the end
    )
    llm_result_custom = llm_optimizer_custom.optimize()

    x_llm_custom, y_llm_custom = llm_result_custom['best_params']['x'], llm_result_custom['best_params']['y']
    valor_llm_custom = llm_result_custom['best_value']

    z_llm_custom = torch.tensor([[x_llm_custom, y_llm_custom]], dtype=torch.float32)
    with torch.no_grad():
        salida_llm_custom = modelo_1.decode(z_llm_custom)
        proy_llm_custom = modelo_2.encode(salida_llm_custom).numpy()[0]

    print(f"✓ LLM-BayesOpt (Custom) completed. Best value: {valor_llm_custom:.6f}")

    # --- 3. Traditional Bayesian Optimization (with bayes_opt) ---
    print("\n[3/3]  Executing Traditional BO (bayes_opt)...")
    opt_tradicional = BayesianOptimization(
        f=objective_rosenbrock_neuronal,
        pbounds={'x': (-3, 3), 'y': (-3, 3)},
        verbose=0,
        random_state=42 # For reproducibility
    )
    opt_tradicional.maximize(init_points=5, n_iter=n_trials - 5) # Total n_trials

    mejor_tradicional = opt_tradicional.max
    x_trad, y_trad = mejor_tradicional['params']['x'], mejor_tradicional['params']['y']
    valor_trad = mejor_tradicional['target']

    z_trad = torch.tensor([[x_trad, y_trad]], dtype=torch.float32)
    with torch.no_grad():
        salida_trad = modelo_1.decode(z_trad)
        proy_trad = modelo_2.encode(salida_trad).numpy()[0]

    print(f" Traditional BO completed. Best value: {valor_trad:.6f}")

    # --- Analysis of Results ---
    print(f"\n{'='*80}")
    print("COMPARATIVE RESULTS ANALYSIS (All 3 Methodologies)")
    print(f"{'='*80}")

    results = {
        'Optuna+LLM': {
            'x_latente': x_optuna, 'y_latente': y_optuna,
            'proy_x': proy_optuna[0], 'proy_y': proy_optuna[1],
            'valor_objetivo': valor_optuna,
            'historial_completo': study_optuna.trials # Optuna already has trials in its study
        },
        'LLM-BayesOpt': {
            'x_latente': x_llm_custom, 'y_latente': y_llm_custom,
            'proy_x': proy_llm_custom[0], 'proy_y': proy_llm_custom[1],
            'valor_objetivo': valor_llm_custom,
            'historial_completo': llm_result_custom['history'] # Our LLM-BayesOpt saves a detailed history
        },
        'BO Tradicional': {
            'x_latente': x_trad, 'y_latente': y_trad,
            'proy_x': proy_trad[0], 'proy_y': proy_trad[1],
            'valor_objetivo': valor_trad,
            'historial_completo': opt_tradicional.res # BayesOpt only saves evaluated points and targets
        }
    }

    # Calculate and display key results
    for name, res in results.items():
        dist_optimo = np.sqrt((res['proy_x']-1)**2 + (res['proy_y']-1)**2)
        rosenbrock_original = -res['valor_objetivo']

        print(f"\n--- {name.upper()} ---")
        print(f"  Latent space: ({res['x_latente']:.4f}, {res['y_latente']:.4f})")
        print(f"  Final projection: ({res['proy_x']:.4f}, {res['proy_y']:.4f})")
        print(f"  Objective value (Negative Rosenbrock): {res['valor_objetivo']:.6f}")
        print(f"  Original Rosenbrock value: {rosenbrock_original:.6f}")
        print(f"  Distance to theoretical optimum (1,1): {dist_optimo:.4f}")

    # Determine the winner
    best_method = max(results, key=lambda k: results[k]['valor_objetivo'])
    print(f"\n THE BEST OPTIMIZER FOUND WAS: {best_method} with an objective value of {results[best_method]['valor_objetivo']:.6f}")
    print(f"   (Remember: we are trying to maximize towards 0, or minimize the original Rosenbrock value towards 0)")

    # --- Comparative Visualizations ---
    if VISUALIZATION_AVAILABLE:
        print(f"\n{'='*80}")
        print("GENERATING COMPARATIVE VISUALIZATIONS (All 3 Methodologies)")
        print(f"{'='*80}")

        fig, axes = plt.subplots(2, 3, figsize=(24, 16)) # Increase size for 3 charts

        # 1. Latent Space Trajectory Comparison
        ax = axes[0, 0]
        # Optuna+LLM
        x_optuna_hist = [t.params['x'] for t in study_optuna.trials if t.state == optuna.trial.TrialState.COMPLETE]
        y_optuna_hist = [t.params['y'] for t in study_optuna.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ax.plot(x_optuna_hist, y_optuna_hist, 'b-o', alpha=0.6, markersize=4, label='Optuna + LLM')
        # LLM-BayesOpt
        x_llm_custom_hist = [h['x'] for h in llm_result_custom['history']]
        y_llm_custom_hist = [h['y'] for h in llm_result_custom['history']]
        ax.plot(x_llm_custom_hist, y_llm_custom_hist, 'r-s', alpha=0.6, markersize=4, label='LLM-BayesOpt')
        # Traditional BO
        x_trad_hist = [p['params']['x'] for p in opt_tradicional.res]
        y_trad_hist = [p['params']['y'] for p in opt_tradicional.res]
        ax.plot(x_trad_hist, y_trad_hist, 'g-^', alpha=0.6, markersize=4, label='Traditional BO')

        ax.scatter([x_optuna], [y_optuna], c='blue', s=250, marker='*', label='Best Optuna+LLM')
        ax.scatter([x_llm_custom], [y_llm_custom], c='red', s=250, marker='*', label='Best LLM-BayesOpt')
        ax.scatter([x_trad], [y_trad], c='green', s=250, marker='*', label='Best Traditional BO')

        ax.set_xlabel('X (Latent Space)')
        ax.set_ylabel('Y (Latent Space)')
        ax.set_title('Optimization Trajectories in Latent Space')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Comparative Convergence (Cumulative Best Value)
        ax = axes[0, 1]

        valores_optuna = [t.value for t in study_optuna.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mejores_optuna = [max(valores_optuna[:i+1]) for i in range(len(valores_optuna))]

        valores_llm_custom = [h['target'] for h in llm_result_custom['history']]
        mejores_llm_custom = [max(valores_llm_custom[:i+1]) for i in range(len(valores_llm_custom))]

        valores_trad = [p['target'] for p in opt_tradicional.res]
        mejores_trad = [max(valores_trad[:i+1]) for i in range(len(valores_trad))]

        ax.plot(range(1, len(mejores_optuna)+1), mejores_optuna, 'b-o', markersize=4, label='Optuna + LLM')
        ax.plot(range(1, len(mejores_llm_custom)+1), mejores_llm_custom, 'r-s', markersize=4, label='LLM-BayesOpt')
        ax.plot(range(1, len(mejores_trad)+1), mejores_trad, 'g-^', markersize=4, label='Traditional BO')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Objective Value (Neg. Rosenbrock)')
        ax.set_title('Convergence of Best Objective Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Distribution of Evaluated Values
        ax = axes[0, 2]
        ax.hist(valores_optuna, bins=15, alpha=0.6, color='blue', label='Optuna + LLM', density=True)
        ax.hist(valores_llm_custom, bins=15, alpha=0.6, color='red', label='LLM-BayesOpt', density=True)
        ax.hist(valores_trad, bins=15, alpha=0.6, color='green', label='Traditional BO', density=True)

        ax.axvline(x=valor_optuna, color='blue', linestyle='--', label=f'Best Optuna+LLM: {valor_optuna:.2f}')
        ax.axvline(x=valor_llm_custom, color='red', linestyle='--', label=f'Best LLM-BayesOpt: {valor_llm_custom:.2f}')
        ax.axvline(x=valor_trad, color='green', linestyle='--', label=f'Best Trad BO: {valor_trad:.2f}')

        ax.set_xlabel('Objective Value (Neg. Rosenbrock)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Evaluated Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Projection in the second autoencoder space
        ax = axes[1, 0]
        # Evaluate projections for a grid of points
        x_grid = np.linspace(-3, 3, 30)
        y_grid = np.linspace(-3, 3, 30)
        proyecciones_x_grid = []
        proyecciones_y_grid = []

        for x_val in x_grid:
            for y_val in y_grid:
                z_temp = torch.tensor([[x_val, y_val]], dtype=torch.float32)
                with torch.no_grad():
                    salida_decoder_1_temp = modelo_1.decode(z_temp)
                    proyeccion_temp = modelo_2.encode(salida_decoder_1_temp)
                    proyecciones_x_grid.append(proyeccion_temp[0, 0].item())
                    proyecciones_y_grid.append(proyeccion_temp[0, 1].item())

        ax.scatter(proyecciones_x_grid, proyecciones_y_grid, alpha=0.2, s=15, c='gray', label='AE Mapping')

        ax.scatter([proy_optuna[0]], [proy_optuna[1]], c='blue', s=250, marker='*', label='Optimum Optuna+LLM')
        ax.scatter([proy_llm_custom[0]], [proy_llm_custom[1]], c='red', s=250, marker='*', label='Optimum LLM-BayesOpt')
        ax.scatter([proy_trad[0]], [proy_trad[1]], c='green', s=250, marker='*', label='Optimum Traditional BO')
        ax.scatter([1], [1], c='black', s=250, marker='x', linewidth=3, label='Theoretical Optimum (1,1)')

        ax.set_xlabel('Projection X (Second AE)')
        ax.set_ylabel('Projection Y (Second AE)')
        ax.set_title('Optimum Points in Projected Space')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Evolution of distance to theoretical optimum
        ax = axes[1, 1]
        distancias_optuna = []
        distancias_llm_custom = []
        distancias_trad = []

        for i in range(len(x_optuna_hist)):
            z_temp = torch.tensor([[x_optuna_hist[i], y_optuna_hist[i]]], dtype=torch.float32)
            with torch.no_grad():
                salida_temp = modelo_1.decode(z_temp)
                proy_temp = modelo_2.encode(salida_temp).numpy()[0]
                distancias_optuna.append(np.sqrt((proy_temp[0]-1)**2 + (proy_temp[1]-1)**2))

        for i in range(len(x_llm_custom_hist)):
            z_temp = torch.tensor([[x_llm_custom_hist[i], y_llm_custom_hist[i]]], dtype=torch.float32)
            with torch.no_grad():
                salida_temp = modelo_1.decode(z_temp)
                proy_temp = modelo_2.encode(salida_temp).numpy()[0]
                distancias_llm_custom.append(np.sqrt((proy_temp[0]-1)**2 + (proy_temp[1]-1)**2))

        for i in range(len(x_trad_hist)):
            z_temp = torch.tensor([[x_trad_hist[i], y_trad_hist[i]]], dtype=torch.float32)
            with torch.no_grad():
                salida_temp = modelo_1.decode(z_temp)
                proy_temp = modelo_2.encode(salida_temp).numpy()[0]
                distancias_trad.append(np.sqrt((proy_temp[0]-1)**2 + (proy_temp[1]-1)**2))

        dist_min_optuna = [min(distancias_optuna[:i+1]) for i in range(len(distancias_optuna))]
        dist_min_llm_custom = [min(distancias_llm_custom[:i+1]) for i in range(len(distancias_llm_custom))]
        dist_min_trad = [min(distancias_trad[:i+1]) for i in range(len(distancias_trad))]

        ax.plot(range(1, len(dist_min_optuna)+1), dist_min_optuna, 'b-o', markersize=4, label='Optuna + LLM')
        ax.plot(range(1, len(dist_min_llm_custom)+1), dist_min_llm_custom, 'r-s', markersize=4, label='LLM-BayesOpt')
        ax.plot(range(1, len(dist_min_trad)+1), dist_min_trad, 'g-^', markersize=4, label='Traditional BO')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Minimum Distance to Theoretical Optimum')
        ax.set_title('Convergence towards Theoretical Optimum (1,1)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Radar chart for performance metrics (simplified)
        ax = fig.add_subplot(2, 3, 6, projection='polar') # Changed subplot to polar projection

        # Metrics to compare (e.g., objective value, distance to optimum, exploration diversity)
        # Normalize for the radar chart
        metrics = {
            'Obj. Value (Max)': [valor_optuna, valor_llm_custom, valor_trad],
            'Opt. Dist. (Min)': [np.sqrt((proy_optuna[0]-1)**2 + (proy_optuna[1]-1)**2),
                                  np.sqrt((proy_llm_custom[0]-1)**2 + (proy_llm_custom[1]-1)**2),
                                  np.sqrt((proy_trad[0]-1)**2 + (proy_trad[1]-1)**2)],
            'Expl. Diversity (Max)': [
                np.std(x_optuna_hist) + np.std(y_optuna_hist),
                np.std(x_llm_custom_hist) + np.std(y_llm_custom_hist),
                np.std(x_trad_hist) + np.std(y_trad_hist)
            ]
        }

        # For normalization: min-max scaling
        normalized_metrics = {}
        for key, vals in metrics.items():
            vals_np = np.array(vals)
            if 'Max' in key: # Maximize the value
                normalized_metrics[key] = (vals_np - np.min(vals_np)) / (np.max(vals_np) - np.min(vals_np) + 1e-6)
            else: # Minimize the value
                normalized_metrics[key] = 1 - (vals_np - np.min(vals_np)) / (np.max(vals_np) - np.min(vals_np) + 1e-6)

        labels = list(normalized_metrics.keys())
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Plot Optuna+LLM
        values = normalized_metrics['Obj. Value (Max)'][0], normalized_metrics['Opt. Dist. (Min)'][0], normalized_metrics['Expl. Diversity (Max)'][0]
        values = list(values) + list(values[:1])
        ax.plot(angles, values, 'b-', linewidth=2, label='Optuna + LLM')
        ax.fill(angles, values, 'blue', alpha=0.1)

        # Plot LLM-BayesOpt
        values = normalized_metrics['Obj. Value (Max)'][1], normalized_metrics['Opt. Dist. (Min)'][1], normalized_metrics['Expl. Diversity (Max)'][1]
        values = list(values) + list(values[:1])
        ax.plot(angles, values, 'r-', linewidth=2, label='LLM-BayesOpt')
        ax.fill(angles, values, 'red', alpha=0.1)

        # Plot Traditional BO
        values = normalized_metrics['Obj. Value (Max)'][2], normalized_metrics['Opt. Dist. (Min)'][2], normalized_metrics['Expl. Diversity (Max)'][2]
        values = list(values) + list(values[:1])
        ax.plot(angles, values, 'g-', linewidth=2, label='Traditional BO')
        ax.fill(angles, values, 'green', alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Performance Analysis (Radar Chart)', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) # Move legend

        plt.tight_layout()
        plt.show()

        print(" Visualizations generated successfully!")

    print(f"\n{'='*80}")
    print("FINAL ANALYSIS AND RECOMMENDATIONS")
    print(f"{'='*80}")
    print("In this experiment, the three optimizers have attempted to maximize the negative of the Rosenbrock function.")
    print("This means that the closer the objective value is to 0, the better the result (ideally, 0).")
    print("The goal is for the projection of the second autoencoder to be (1,1).")

    # Analysis of each method
    print("\n--- Individual Performance ---")
    for name, res in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Objective Value: {res['valor_objetivo']:.6f}")
        print(f"  Original Rosenbrock Value at Optimum: {-res['valor_objetivo']:.6f}")
        print(f"  Distance to Theoretical Optimum (1,1) in projected space: {np.sqrt((res['proy_x']-1)**2 + (res['proy_y']-1)**2):.4f}")
        print(f"  Found Latent Parameters: x={res['x_latente']:.4f}, y={res['y_latente']:.4f}")

    # General conclusion
    print(f"\n--- General Conclusion ---")
    print(f"The best optimizer found in this execution is: {best_method}")
    print(f"It achieved an objective value of {results[best_method]['valor_objetivo']:.6f}, which corresponds to an original Rosenbrock value of {-results[best_method]['valor_objetivo']:.6f}.")
    print(f"Its projection in the final space was ({results[best_method]['proy_x']:.4f}, {results[best_method]['proy_y']:.4f}), with a distance of {np.sqrt((results[best_method]['proy_x']-1)**2 + (results[best_method]['proy_y']-1)**2):.4f} to the theoretical optimum (1,1).")

    print("\nIt is important to remember that the stochastic nature of these algorithms and the complexity of the objective function (Rosenbrock mapped through AEs) can lead to variations in results between executions.")
    print("However, this unified comparison allows for a more direct evaluation under the same conditions.")

    print(f"\n{'='*80}")
    print("COMPARISON FINISHED ✅")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Define the number of trials/iterations for the comparison
    NUM_TRIALS = 50
    main_comparison(n_trials=NUM_TRIALS)