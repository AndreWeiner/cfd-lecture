# %% [markdown]
# ---
# title: Consistency, stability, and convergence
# subtitle: Exercise 4 — theory and code lab
# jupyter: python3
# ---

# %% [markdown]
# ## Learning objectives
#
# This exercise complements [Lecture 4](lecture_04.py) and reuses the periodic grid and spatial stencils from [Exercise 3](exercise_03.py). We separate the space and time contributions to the residual for implicit updates, then derive and test a Courant bound for the explicit Lax–Friedrichs scheme.
#
# After completing the exercise, we should be able to:
#
# - Derive a consistency residual from exact samples and identify its leading space and time terms.
# - Measure the two consistency orders by varying one discretization parameter at a time.
# - Predict and measure the stability of a one-step scheme using its Fourier gain.

# %% [markdown]
# ## Part 1 — Theory
#
# ### Task 1 — Separate space and time consistency
#
# We use periodic convection–diffusion on the unit interval with constant $v>0$ and $\Gamma>0$. The grid has $N_x$ distinct nodes $x_j=j\Delta x$, with $\Delta x=1/N_x$ and periodic indexing. We write the PDE as $\partial_t\varphi=A\varphi$ with $A=-v\partial_x+\Gamma\partial_{xx}$. The two discrete spatial operators are
#
# $$
# A_h^{\mathrm{up}}=-vD^-+\Gamma D^-D^+,
# \qquad
# A_h^{\mathrm{cen}}=-vD^0+\Gamma D^-D^+.
# $$
#
# Here $D^-$ and $D^0$ are the backward and central first differences, and $D^-D^+$ is the compact central second difference. We compare backward Euler with upwind convection, backward Euler with central convection, and Crank–Nicolson with central convection. These numerical updates can be written together as
#
# $$
# \frac{\varphi_j^{n+1}-\varphi_j^n}{\Delta t}
# =(1-\alpha)(A_h\varphi^n)_j+\alpha(A_h\varphi^{n+1})_j,
# \qquad
# \alpha=\begin{cases}
# 1 & \text{backward Euler},\\
# 1/2 & \text{Crank–Nicolson}.
# \end{cases}
# $$
#
# Backward Euler evaluates the spatial right-hand side at the new time level; Crank–Nicolson averages its values at the old and new levels. Both are implicit because the unknown new values occur on the right-hand side.
#
# As in Lecture 4, we evaluate consistency by inserting exact samples into the discrete equation. We write $q_j^n=\varphi(x_j,t^n)$, so $q^n=(\pi_h\varphi)^n$. Moving all terms to the left gives the derivative-scaled consistency residual $\tau$ (tau):
#
# $$
# \tau_j^n=\frac{q_j^{n+1}-q_j^n}{\Delta t}
#  -(1-\alpha)(A_hq^n)_j-\alpha(A_hq^{n+1})_j.
# $$
#
# We assume a sufficiently smooth exact solution for the Taylor expansions below. Temporal expansions keep $x_j$ fixed, and spatial expansions keep time fixed. We use the Courant number $C=v\Delta t/\Delta x$ when discussing coupled refinement. The residual measures a mismatch in the equation; it is distinct from the numerical solution error $\varphi_h-\pi_h\varphi$.
#
# 1. Add and subtract the corresponding weighted values of $\partial_t\varphi$ at the two endpoints of the time step. Write $\tau=\tau_t+\tau_x$, where $\tau_t$ compares the time difference with exact derivatives and $\tau_x$ compares $A\varphi$ with $A_hq$. Use the PDE to justify the split.
# 2. Derive the leading term of $\tau_t$ for backward Euler by expanding about $t^{n+1}$. Derive it for Crank–Nicolson by expanding about the midpoint $t^{n+1/2}$. Show which terms cancel in the midpoint expansion.
# 3. Expand $D^-$, $D^0$, and $D^-D^+$ about $x_j$. Use the results to derive the leading spatial contribution for upwind and central convection with compact central diffusion. Include the signs from $A-A_h$.
# 4. Tabulate the time and space consistency orders of the three combinations. Explain what changing only the convection stencil improves, and what changing only the time integrator improves. Explain why measuring only the sum $\tau_t+\tau_x$ while refining at fixed $C$ does not generally identify the two orders separately.
#
# ### Task 2 — Derive the Lax–Friedrichs stability bound
#
# We now set $\Gamma=0$ and consider pure convection on the same periodic domain. The **Lax–Friedrichs** update replaces the old center value in forward Euler with central convection by the average of its two neighbors:
#
# $$
# \varphi_j^{n+1}
# =\frac{\varphi_{j-1}^n+\varphi_{j+1}^n}{2}
# -\frac{C}{2}(\varphi_{j+1}^n-\varphi_{j-1}^n),
# \qquad C=\frac{v\Delta t}{\Delta x}.
# $$
#
# We use the same Fourier test as in Lecture 4: the linear, constant-coefficient update acts on a uniform periodic grid. A perturbation $w$ satisfies the same update, as follows by subtracting two numerical updates. Here $\theta=\kappa\Delta x$ is the phase change between neighboring nodes, with $\kappa_m=2\pi m/L$ and $L=1$. Thus the represented angles are $\theta_m=2\pi m/N_x$ modulo $2\pi$; we seek a bound valid over the full angle range and independent of mesh refinement.
#
# 1. Substitute $w_j^n=\widehat w^{\,n}\exp(\mathrm{i}j\theta)$ and derive the gain $G(\theta)=\widehat w^{\,n+1}/\widehat w^{\,n}$. Use $G\overline G$ to find the range of $C$ for which all Fourier angles satisfy $|G|\le 1$.
# 2. Rewrite the update as a weighted sum of the two neighboring values. Determine when the weights form a convex combination and use that to bound the maximum norm. Describe the update at $C=1$ for positive velocity.
# 3. Compare $\theta=\pi$ (the alternating mode) with $\theta=\pi/2$ (four nodes per wavelength). Which mode would reveal a violation of the Courant bound? Explain your choice from the gain.

# %% [markdown]
# ## Part 2 — Code lab
#
# Use the Python environment established in Exercise 1. Save your work in the ignored `exercises/` directory, for example as `exercise_04.py`. Reuse your periodic stencils from Exercise 3 and separate exact sampling, residual evaluation, and time stepping into functions. NumPy is sufficient for the required tables and comparisons.
#
# Coding assistance may be used following [Topic 0](lecture_00.qmd). Check the implementation against your derivations and explain the measured results.
#
# ### Step 1 — Measure the two contributions to consistency
#
# Use $v=1$ and $\Gamma=0.01$ with the exact periodic solution
#
# $$
# \varphi(x,t)=\exp(-4\pi^2\Gamma t)\sin\bigl(2\pi(x-vt)\bigr).
# $$
#
# We evaluate the residual using exact samples throughout. The calculation requires no numerical time steps or implicit solves. Measure each contribution in the mesh-weighted two-norm $\|r\|_{2,h}=(\Delta x\sum_j|r_j|^2)^{1/2}$.
#
# 1. Implement the exact solution and its analytical time derivative, then implement the two contributions $\tau_t$ and $\tau_x$ from Task 1. At one nontrivial pair $(N_x,\Delta t)$, verify that their sum agrees to roundoff with the residual computed directly from the stencil and exact samples. Use periodic wrapping for all discrete spatial derivatives.
# 2. **Time refinement.** Keep $N_x=128$ and vary $\Delta t$ over $0.08,0.04,0.02,0.01,0.005$. Measure only $\tau_t$. For backward Euler, fix the right endpoint at $t^{n+1}=0.4$. For Crank–Nicolson, fix the midpoint at $t^{n+1/2}=0.4$. Explain why changing only the convection stencil leaves this measurement unchanged.
# 3. **Space refinement.** Fix $\Delta t=0.04$ and use $N_x=16,32,64,128,256$. Use the same time anchors as in Step 2 of this task. Measure only $\tau_x$, including the endpoint weights of each scheme. This keeps the temporal quadrature in the split fixed while varying the spatial stencil error.
# 4. Tabulate the norms and the observed orders $\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})/\log(2)$ for both refinement studies. Compare the measured orders with your predictions. State which error is being measured in each table.
#
# ### Step 2 — Test the Lax–Friedrichs bound
#
# Use pure convection with $v=1$ and $N_x=64$. For each $C=0.8,1,1.2$, set $\Delta t=C\Delta x/v$. Start from $\varphi_j^0=\cos(j\pi/2)$, a real wave with four nodes per wavelength. The matching positive Fourier mode has index $m=N_x/4$.
#
# 1. Implement the Lax–Friedrichs step with periodic wrapping. Check that it leaves a constant field unchanged. At $C=1$, compare one step with an exact shift by one node on a general periodic array.
# 2. Predict the gain from Task 2 for $\theta=\pi/2$. March each case for 12 steps and extract the complex coefficient at $m=N_x/4$, for example with `numpy.fft.fft`. Compare the measured one-step ratio $\widehat\varphi_m^{n+1}/\widehat\varphi_m^n$ with the predicted complex gain, and compare the amplitude history with $|G|^n|\widehat\varphi_m^0|$. Use the same Fourier normalization throughout.
# 3. Repeat with the alternating initial profile $(-1)^j$ and inspect the coefficient at $m=N_x/2$. Explain why the observation from this mode alone would give a misleading verdict about the scheme at $C=1.2$.
# 4. Summarize the results below, at, and above the Courant bound. Explain the difference between damping a wave and transporting it accurately. The cases have the same number of steps but different physical end times, so compare each amplitude history with its own gain prediction.
