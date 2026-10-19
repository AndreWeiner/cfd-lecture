# %% [markdown]
# ---
# title: Transport phenomena and classification of partial differential equations
# subtitle: Exercise 2 — theory and code lab
# jupyter: python3
# ---
#
# ## Learning objectives
#
# Practice classifying scalar PDEs and systems, distinguishing characteristic speeds from PDE type, and connecting eigenvectors to physical disturbances. Use the notation and classification conventions introduced in [Lecture 2](lecture_02.py).
#
# ## Part 1 — Theory
#
# ### Task 1 — Mixed derivatives and lower-order terms
#
# Consider the scalar equation
#
# $$\partial_{xx} \varphi+4\partial_{xy} \varphi+\partial_{yy} \varphi+3\partial_x \varphi-2\varphi=\sin(x).$$
#
# The notation $\partial_x\varphi$ denotes the partial derivative with respect to $x$; $\partial_{xx}\varphi$ and $\partial_{xy}\varphi$ denote second partial derivatives.
#
# 1. Identify the differential order and determine whether the equation is linear, quasilinear, or fully nonlinear, and whether it is homogeneous.
# 2. Isolate the principal part and identify $A$, $B$, and $C$ using the lecture's convention $A\partial_{xx} \varphi+2B\partial_{xy} \varphi+C\partial_{yy} \varphi$.
# 3. Calculate the discriminant and classify the PDE.
# 4. Use the plane-wave ansatz to obtain the characteristic-normal condition. Determine the characteristic-line orientations and distinguish their tangent directions from their normals.
# 5. Explain whether replacing the right-hand side by another prescribed source, or changing the coefficients of the first derivative and the field itself, would change this classification.
#
# ### Task 2 — A model of changing type in transonic flow
#
# Transonic flow can contain both subsonic and supersonic regions, as illustrated by the airfoil example in Lecture 2. The **Tricomi equation** is a canonical mathematical model used to study such changes of type:
#
# $$y\partial_{xx} \varphi+\partial_{yy} \varphi=0.$$
#
# Here $\varphi$ is an unknown scalar flow function, and $x$ and $y$ are dimensionless mathematical coordinates. In a transonic-flow derivation, these coordinates and the unknown can result from transformations of the original flow problem; they should not automatically be identified with physical position and the physical velocity potential. This task examines the canonical equation rather than reconstructing a complete airfoil flow.
#
# 1. Classify the equation with respect to linearity and identify its principal coefficients.
# 2. Determine the PDE type separately for $y<0$, $y=0$, and $y>0$.
# 3. At a fixed point with $y\neq0$, freeze the coefficients and determine any real characteristic normals and the corresponding local tangent slopes. Explain why these local slopes do not imply globally straight characteristic lines.
# 4. At $y=0$, examine the characteristic-normal condition before dividing by either wave-vector component. Explain how to account for directions omitted by the ratio $n_x/n_y$.
# 5. Discuss whether the classification at $y=0$ alone makes this a heat equation or identifies a well-posed marching direction. State what additional information is needed before formulating an initial- or boundary-value problem.
#
# ### Task 3 — Small surface waves in a shallow-water channel
#
# Consider a long, straight channel with a flat bottom. The water is initially at rest with uniform depth $H>0$. A small surface displacement produces long waves, whose wavelength is much larger than $H$. Neglect viscosity and assume hydrostatic pressure, so that the horizontal flow can be described by its depth-averaged velocity.
#
# For smooth one-dimensional shallow-water flow, conservation of mass and the horizontal momentum balance give
#
# $$\frac{\partial h}{\partial t}+\frac{\partial(hv)}{\partial x}=0,
# \qquad
# \frac{\partial v}{\partial t}+v\frac{\partial v}{\partial x}
# +g\frac{\partial h}{\partial x}=0.$$
#
# Here $x$ is the horizontal coordinate, $t$ is time, $h(x,t)$ is the water depth, $v(x,t)$ is the depth-averaged horizontal velocity, and $g>0$ is gravitational acceleration. Write $h=H+\eta$, where $\eta(x,t)$ is the small displacement of the water surface. Linearization about $h=H$, $v=0$ neglects products of perturbations, such as $\eta v$ and $v \partial_x v$.
#
# The general first-order form used below is
#
# $$\mathbf B\frac{\partial\mathbf r}{\partial t}
# +\mathbf A\frac{\partial\mathbf r}{\partial x}=\mathbf0.$$
#
# Let $\rho_0>0$ denote the constant water density.
#
# 1. Derive the linearized equations from the smooth shallow-water equations above. Explain the physical meaning of each component equation.
# 2. Multiply the linearized momentum equation by $\rho_0H$ to express it per unit horizontal area. Identify an appropriate unknown-field vector $\mathbf r$ and determine the coefficient matrices $\mathbf A$ and $\mathbf B$. State their dimensions and the order of the components in $\mathbf r$.
# 3. Substitute $\mathbf r=\hat{\mathbf r}\exp[\mathrm i(kx-\omega t)]$, with real $k\neq0$. Define $\lambda=\omega/k$ and derive the condition $\det(\mathbf A-\lambda\mathbf B)=0$ for a nonzero amplitude.
# 4. Calculate the characteristic speeds and eigenvectors for $H>0$, $g>0$. Classify the linearized system and interpret the modes as surface waves travelling through water at rest.
# 5. Obtain the same speeds from $\mathbf J=\mathbf B^{-1}\mathbf A$. Explain why the eigenvalues of $\mathbf A$ alone are not the propagation speeds, and why the final speeds do not depend on the constant water density.
# 6. For a finite channel, state the initial fields required and count the incoming modes at each end. Explain why independently prescribing both $\eta$ and $v$ at both ends would generally overconstrain the problem. Discuss why taking $H=0$ is outside the assumptions of this linearization, rather than a reliable model of waves over a dry bed.
#
# ### Task 4 — Isentropic gas flow in a duct
#
# Consider smooth, inviscid gas flow in a straight duct of constant cross-sectional area. Neglect heat transfer and body forces, and assume that the entropy is the same constant throughout the gas. For a calorically perfect ideal gas, pressure can then be expressed as a function of density alone:
#
# $$p(\rho)=K\rho^\gamma,\qquad K>0,\qquad\gamma>1.$$
#
# Here $\rho(x,t)>0$ is density, $u(x,t)$ is velocity, $K$ is a constant fixed by the entropy level, and $\gamma$ is the heat-capacity ratio. Conservation of mass and momentum gives the **isentropic Euler equations**:
#
# $$\frac{\partial\rho}{\partial t}+\frac{\partial(\rho u)}{\partial x}=0,
# \qquad
# \frac{\partial(\rho u)}{\partial t}
# +\frac{\partial\bigl(\rho u^2+p(\rho)\bigr)}{\partial x}=0.$$
#
# The pressure law closes this two-equation model without a separate total-energy equation. The assumption of one constant entropy level restricts the allowed flow states and does not describe arbitrary shocks of the full Euler system.
#
# 1. Define the two-component conserved-state vector and flux vector. Express the fluxes in terms of the independent conserved components and derive the complete $2\times2$ flux Jacobian.
# 2. Define the sound speed by $c^2=\mathrm dp/\mathrm d\rho$. Derive the eigenvalues and right eigenvectors of the Jacobian and classify the system for physical states. Explain why only two modes occur, compared with the three modes in the one-dimensional full Euler equations from the lecture.
# 3. Linearize about a uniform reference state. Examine rightward subsonic, sonic, and supersonic reference velocities. Does a zero characteristic speed imply a repeated eigenvalue or a change of PDE type?
# 4. For a finite interval with a left inlet and a right outlet, count incoming modes in the subsonic and supersonic cases. Use outward normals $-1$ at the inlet and $+1$ at the outlet. Explain how the count differs from that of the full Euler system.
# 5. Contrast this result with the steady small-disturbance potential-flow equation $(1-M_\infty^2)\partial_{xx} \psi+\partial_{yy} \psi=0$. Explain why its Mach-dependent classification does not contradict the classification of the time-dependent isentropic Euler system.
#
# ## Part 2 — Code lab
#
# ### Characteristic propagation with NumPy
#
# Explore the isentropic Euler model from Theory Task 4, linearized about a uniform state. Use exact transport of the characteristic variables to study classification and information propagation without introducing a finite-difference solver. Implement and verify the transformations yourself.
#
# Use the Python environment and dependencies from [Exercise 1](exercise_01.py#part-2-code-lab). Create a script named `exercise_02_characteristics.py` in your ignored local `exercises/` folder. NumPy and Matplotlib are sufficient.
#
# ::: {.callout-note title="Working with an AI tutor"}
# Use an AI tool for hints, explanations, and help investigating failed checks if useful. Make sure that you can explain the transformations and recreate the essential steps independently. Follow the guidance in [Lecture 0](lecture_00.qmd).
# :::
#
# ### Step 1 — Reference states and the flux Jacobian
#
# Use dimensionless quantities and set
#
# $$\bar\rho=1,\qquad\gamma=1.4,\qquad
# \bar c=1,\qquad K=\frac{\bar c^2}{\gamma\bar\rho^{\gamma-1}},
# \qquad\bar p=K\bar\rho^\gamma.$$
#
# 1. Implement a function that constructs the isentropic Euler flux Jacobian from a reference density, velocity, $K$, and $\gamma$. Use your derivation from Theory Task 4.
# 2. Investigate $\bar u\in\{0,\,0.5,\,1,\,1.5\}$. Use `numpy.linalg.eig` to obtain the two eigenvalues and right eigenvectors, and sort the eigenpairs by increasing eigenvalue.
# 3. Compare the computed eigenvalues with your analytical speeds. Check the eigenvector matrix rank and the residual $\mathbf J_0\mathbf R-\mathbf R\boldsymbol\Lambda$. Report the tolerance used in these checks.
#
# Remember that eigenvector scaling and signs are not unique. Keep every eigenvector column paired with its eigenvalue throughout the calculation.
#
# ### Step 2 — A common initial physical disturbance
#
# Use the same initial density and velocity perturbations for every reference velocity:
#
# $$G(x)=\exp[-(x/\ell)^2],\qquad\ell=0.35,\qquad\varepsilon=0.01,$$
#
# $$\rho'(x,0)=\varepsilon\bar\rho G(x),\qquad u'(x,0)=0.$$
#
# Pressure is determined by density in this model; do not prescribe an independent pressure perturbation. Use the interval $-6\leq x\leq8$ as a plotting window and follow the profiles for $0\leq t\leq2$. Primes denote perturbations about the reference state. The symbol $u'$ denotes a velocity perturbation, whereas $\mathbf v$ below is a two-component conserved-state perturbation vector.
#
# 1. Convert the initial density and velocity perturbations into conserved-variable perturbations, $\mathbf v_0=\mathbf q(x,0)-\bar{\mathbf q}$. Explain the conversion of momentum density.
# 2. Obtain the initial characteristic profiles from $\mathbf R\mathbf w_0=\mathbf v_0$, using `numpy.linalg.solve` rather than forming an explicit inverse.
# 3. Reconstruct $\mathbf R\mathbf w_0$ and check that it reproduces the initial conserved perturbations. Check the corresponding physical perturbations as well.
#
# Keep the physical initial profiles fixed when changing $\bar u$. Prescribing identical numerical values of $\mathbf w_0$ for independently normalized eigenvector matrices would generally change the initial physical disturbance.
#
# ### Step 3 — Propagation and reconstruction
#
# Propagate each of the two characteristic profiles using
#
# $$w_j(x,t)=w_{j,0}(x-\lambda_jt),\qquad j=1,2.$$
#
# 1. Evaluate the shifted Gaussian profiles analytically. Do not advance a discretized PDE in time or shift sampled profiles by interpolation.
# 2. Reconstruct the conserved perturbations using $\mathbf v=\mathbf R\mathbf w$.
# 3. Recover $\rho'$, $u'$, and $p'$ by linearizing their relations to the conserved variables about the reference state. Derive the pressure perturbation from the isentropic pressure law. Use the linearized relations consistently; the propagated fields solve the frozen linear system, not the full nonlinear equations.
# 4. Plot the characteristic profiles and the reconstructed physical perturbations at several times. An animation is optional. Label each mode with its propagation speed and use consistent axes when comparing the reference states.
#
# The calculation describes transport on the whole line. The plotting window is large enough to contain the visible profiles during the chosen time interval; it is not a computational boundary where inlet or outlet conditions are imposed.
#
# ### Step 4 — Interpretation and evidence of correctness
#
# 1. Explain which modes move left, remain stationary, or move right for each reference state. Distinguish propagation relative to the gas from propagation in the fixed coordinate system.
# 2. Explain what changes as $\bar u$ passes through $\bar c$, and whether this changes hyperbolicity or the completeness of the eigenvectors.
# 3. For one subsonic and one supersonic state, draw an $x$–$t$ diagram through $P=(1,1)$. Mark the initial footpoints, dependence bounds, and influence bounds using the speeds computed by your script. Explain the role of the two individual paths and distinguish the enclosing bounds from the information carried along each path.
# 4. Verify at least one additional property beyond agreement with a reference plot. For example, check that each characteristic profile preserves its shape and amplitude, or that its value remains constant along its characteristic. Explain why your check would detect a plausible implementation error.
# 5. Connect the propagation plots to the incoming boundary-condition counts in Theory Task 4. Explain why there is no independent entropy/contact profile in this model, in contrast to the full Euler example in Lecture 2.
#
# Retain the script, clearly labeled plots, and a short explanation of your observations and checks. Include enough information to reproduce the reference states and plotting times.
