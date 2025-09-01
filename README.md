# Design and Optimization Pipeline for Peptide Synthesis Solvents

This project implements an advanced computational workflow for the design of new solvents for peptide synthesis. It combines a series of machine learning models and computational chemistry calculations to identify molecules with desired chemical and physical properties.

-----

### 🧪 Project Methodology

The project implements a sophisticated computational workflow for the design and optimization of new peptide synthesis solvents. The methodology is structured as a multi-stage pipeline that integrates quantum chemistry calculations (DFT) with advanced machine learning (ML) models and neural networks.

Here is a diagram illustrating the workflow:
Methology of Algotithm.png

1.  **Dataset Generation**: An initial dataset is created from common solvents and databases like **ChEMBL**. This set is expanded by generating molecular variants to create a diverse collection of compounds.
2.  **Descriptor Processing**: For each molecule, over 40 heuristic molecular descriptors are calculated. These descriptors, including physical and chemical properties like molecular weight (MolWt), polarity (LogP), and polar surface area (TPSA), are normalized for use as input into the machine learning models.
3.  **DFT (Density Functional Theory) Calculations**: A diverse subset of molecules is subjected to DFT calculations to obtain high-accuracy reference properties, such as total energy, dipole moment, and HOMO-LUMO orbital energies. This high-quality data is used to train the surrogate models.
4.  **Surrogate Models**: **Random Forest** and **Gaussian Process Regressor** models are trained to quickly predict molecular properties from their molecular descriptors. These surrogate models, trained on DFT data, act as a computationally inexpensive alternative to full quantum chemistry calculations, allowing for a broader exploration of the chemical space.
5.  **Autoencoder Architecture**:
      * **Variational Autoencoder (VAE)**: A VAE is trained to compress the molecular descriptor representation from 50 dimensions to a continuous latent space of 8 dimensions. This latent space allows for smooth navigation to generate new molecules.
      * **Property Autoencoder**: A second autoencoder refines the VAE's representation, further reducing the latent space to a 4-dimensional representation focused on key solvent properties.
6.  **Bayesian Optimization**: Bayesian optimization is used to efficiently search the VAE's latent space, identifying molecular candidates that maximize an objective function. This function rewards desirable properties for peptide synthesis, such as a high dipole moment and a moderate HOMO-LUMO energy gap.
7.  **DFT Validation**: The best candidates identified by the optimization process are validated with a full DFT calculation to obtain their reference properties. The results are compared with the surrogate model predictions to evaluate the pipeline's performance and the models' accuracy.

-----
### Methodologies of ML: Descriptive Equations

1.  $L(\varphi, \theta; x) = E_{q_\varphi(z|x)} [\log p_\theta(x, z) - \log q_\varphi(z|x)]$

2.  $p_\pi(t) = \prod_{r \in t_R} \pi_r$

3.  $p_\pi(x) = \sum_{t \in T_G(x)} p_\pi(t)$

4.  $f(x) = E_\xi[f̆(x; \xi)]$

5.  $k(x, x') = \sigma^2 e^{(-\frac{|x - x'|^2}{2l^2})} + \frac{\delta(x, x')}{\beta}$

6.  $p(Y|X, \theta) = \prod_{i=1}^d \frac{e^{(-\frac{1}{2}y_{:,i}^T K^{-1}y_{:,i})}}{(2\pi)^{n/2} |K|^{1/2}}$

7.  $P_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{(-\frac{(x-\mu)^2}{2\sigma^2})}$

8.  $N(X|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} e^{(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu))}$

9.  $p(w | D) = \frac{p(D | w)p(w)}{p(D)}$

10. $y(x^{(i)}) = \beta_h h_f(x^{(i)}) + \varepsilon^{(i)}$

---

### Equation Descriptions

| Equation Number | Description |
| :--- | :--- |
| 1 | Mathematical description of the **Autoencoder** process. |
| 2 | Mathematical description of the **Context-Free Grammars** process. |
| 3 | Mathematical description of the **Context-Free Grammars** process. |
| 4 | Mathematical description of the **Context-Free Grammars** process. |
| 5 | Mathematical description of the **Dimensionality Reduction using GP** process. |
| 6 | Mathematical description of the **Dimensionality Reduction using GP** process. |
| 7 | Mathematical description of the **GP (Gaussian Processes)** process. |
| 8 | Mathematical description of the **GP (Gaussian Processes)** process. |
| 9 | Mathematical description of the **BO (Bayesian Optimization)** process. |
| 10 | Mathematical description of the **BO (Bayesian Optimization)** process. |
### Symbol Table

| Symbol | Typical Definition |
| :--- | :--- |
| $\varphi$ | Variational parameters, with multiple representations (scalar or vector). |
| $\theta$ | Model parameters, with multiple representations (scalar, vector, or matrix). |
| $x$ | Input data, represented as a vector. |
| $z$ | Latent variable, typically represented as a vector. |
| $p(x,z)$ | Probability distribution that depends on $\theta$, generally a scalar. |
| $\pi_r$ | Probability associated with rule $r$, typically a scalar. |
| $p_\pi(t)$ | Probability of the syntactic tree $t$, typically a scalar. |
| $\xi$ | Random variable in the stochastic function, with multiple representations (scalar, vector, or matrix). |
| $E_\xi$ | Mathematical expectation over the distribution of $\xi$, typically a scalar. |
| $x'$ | Input vector from the data space. |
| $\sigma^2$ | Variance, typically a scalar. |
| $l$ | Scale length of the RBF kernel, typically a scalar. |
| $\delta(x,x')$ | Function that evaluates if $x$ and $x'$ are identical, typically a scalar. |
| $\beta$ | Scalar associated with noise. |
| $Y$ | Output data matrix with multiple dimensions by columns. |
| $X$ | Input data matrix. |
| $K$ | Covariance matrix, can be a matrix or symmetric. |
| $d$ | Number of dimensions in $Y$, typically a scalar. |
| $n$ | Number of data points in the training set, typically a scalar. |
| $\mu$ | Mean, typically a scalar. |
| $P_X(x)$ | Probability density of $x$, typically a scalar. |
| $\Sigma$ | Symmetric covariance matrix. |
| $D$ | Dimensionality of the data, typically a scalar. |
| $w$ | Random model parameter in vector form. |
| $p(w), p(D∣w), p(D), p(w∣D)$ | Multiple probabilities (Prior, Evidence), all scalars. |
| $x^{(i)}$ | Input vector for data point $i$. |
| $\beta_h$ | Unknown parameters to be estimated, with multiple representations (scalars and vectors). |
| $h_f$ | Nonlinear basis function, which maps the input data to a new space. |
### 💻 Installation Requirements

The project requires the following Python libraries. You can install them using `pip`.

```bash
pip install numpy torch scikit-learn pandas bayes-opt tqdm
```

**Optional Dependencies:**

  * **chembl\_webresource\_client**: To access the molecule database.
    ```bash
    pip install chembl_webresource_client
    ```
  * **PySCF**: To perform DFT (Density Functional Theory) calculations. If not installed, DFT calculations will be simulated.
    ```bash
    pip install pyscf
    ```

-----

### 🚀 How to Use

To run the complete workflow, simply execute the main script.

```bash
python Final\ Code\ Molecules.py
```

The script will automatically execute all pipeline steps in the correct order, from data generation to DFT validation of the optimized candidate.

-----

### 📈 Results (Example Output)

The script concludes by printing the key results, showing the surrogate model's performance and the validation of the best candidate.

```
==========================================================================================
 ENHANCED PIPELINE — PEPTIDE SYNTHESIS SOLVENT DESIGN
==========================================================================================

STEP 1 — Generating extended dataset...
...
✓ Dataset generated: 120 compounds

STEP 2 — Building descriptor matrix...
✓ 120 molecules, 50 descriptors

...

STEP 4 — Training surrogate with cross-validation (k=5)...
  CV dipole: R2=0.887±0.052, RMSE=0.556
  CV gap: R2=0.751±0.068, RMSE=0.045
...

==========================================================================================
FINAL RESULTS
==========================================================================================

Dataset: 120 compounds, 50 descriptors
Valid DFT: 25

Best candidate: score=8.1256
  • Similar molecule: O=C1OCCO1_var (dist=0.0321)

DFT vs ML Comparison:
  - energy: DFT=-114.3415 | ML=-113.8824 | error%=0.40
  - dipole: DFT=5.4328 | ML=5.1983 | error%=4.32
  - gap: DFT=0.4851 | ML=0.4776 | error%=1.55
```
