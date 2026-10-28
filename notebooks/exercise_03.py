# %% [markdown]
# ---
# title: Finite difference method foundations
# subtitle: Exercise 3 — theory and code lab
# jupyter: python3
# ---
#

# %% [markdown]
# ## Learning objectives
#
# This exercise complements [Lecture 3](lecture_03.py). We derive additional boundary formulas, measure approximation orders rather than assume them, and investigate combined space–time error in a transport calculation and approximation orders on nonuniform grids.
#
# After completing the exercise, we should be able to:
#
# - Construct finite difference coefficients by matching Taylor terms.
# - Distinguish interior accuracy from boundary accuracy and special exactness.
# - Estimate observed approximation orders from systematic mesh refinement.
# - Implement periodic transport and assess the combined space–time error against an analytical solution.
# - Compare approximation errors on smoothly graded and alternating-width meshes, including a coordinate-mapped derivative.
#
# Use $N$ for the number of stored nodes and lowercase $n$ for the time-level index. State all assumptions and show the intermediate steps of each derivation. The tasks intentionally leave the formulas and results to be established by the reader.
#

# %% [markdown]
# ## Part 1 — Theory
#
# ### Task 1 — First derivatives at a boundary
#
# Consider a smooth function on $[0,1]$ with $N$ uniformly spaced nodes, $x_i=ih$, $h=1/(N-1)$.
#
# 1. Starting from Taylor expansions about $x_0$, derive the leading error of the two-point forward approximation to $\varphi'(x_0)$. Define clearly whether your error is approximation minus exact derivative or the opposite.
# 2. Seek an approximation of the form
#
#    $$
#    \varphi'(x_0)\approx\frac{a\varphi_0+b\varphi_1+c\varphi_2}{h}.
#    $$
#
#    Determine $a,b,c$ by requiring exactness for constant, linear, and quadratic polynomials. Explain how these conditions follow from the Taylor expansions.
# 3. Derive the leading error term and its order. Which derivative determines its coefficient?
# 4. Construct the corresponding formula at $x_{N-1}$. How do the stencil and error change under reflection?
# 5. Propose a complete first-derivative approximation using central differences in the interior and your formulas at the endpoints. Compare the expected maximum error when the boundary formulas are first order versus second order.
#
# ### Task 2 — Second derivatives and stencil width
#
# 1. Derive the compact central approximation to $\varphi''(x_i)$ by choosing coefficients in
#
#    $$
#    \frac{a\varphi_{i-1}+b\varphi_i+c\varphi_{i+1}}{h^2}.
#    $$
#
#    Identify its leading error using Taylor expansion through sufficiently high degree.
# 2. At the left boundary, consider $(\varphi_0-2\varphi_1+\varphi_2)/h^2$. Determine its approximation order **at $x_0$**. Why does its resemblance to the interior formula not guarantee the same accuracy there?
# 3. Add a fourth point and derive a second-order one-sided approximation of $\varphi''(x_0)$ using $x_0,x_1,x_2,x_3$. State the minimum $N$, derive the leading error coefficient, and obtain the reflected right-boundary formula.
# 4. Compare $D^0D^0$ and $D^-D^+$ on the uniform mesh. List the nodes with nonzero coefficients and apply both operators to $\varphi_i=(-1)^i$. Explain the difference without interpreting this sequence as a smooth resolved wave.
#
# ### Task 3 — Predict what a refinement study can reveal
#
# Suppose an error behaves as $E(h)\approx Kh^p$, with $K\ne0$.
#
# 1. Derive the observed-order estimator
#
#    $$
#    p_{\mathrm{obs}}=\frac{\log(E(h)/E(h/2))}{\log2}.
#    $$
#
# 2. Explain why this estimate is unreliable when an error is zero, near floating-point precision, or outside the range where the leading Taylor term dominates.
# 3. For the compact central second derivative, predict what changes when the test function is quadratic rather than exponential. Explain why a polynomial can verify exactness without providing a useful measurement of the generic approximation order.
# 4. On a nonuniform mesh, explain why an “order” requires a specified family of refined meshes. Contrast a smooth mapped grid with neighboring widths that retain a fixed unequal ratio as they shrink.
#

# %% [markdown]
# ## Part 2 — Code lab
#
# Use the Python environment established in Exercise 1, with NumPy and Matplotlib. Save your work in the ignored `exercises/` directory, for example as `exercise_03.py`, and use functions to separate approximation, error measurement, and plotting. No additional package installation is required.
#
# Coding assistance may be used following [Topic 0](lecture_00.qmd), but check the generated formulas against your derivations and explain each result independently. Do not accept a fitted slope or a visually plausible curve as the sole verification of an implementation.
#
# ### Step 1 — Implement and check derivative operators
#
# Implement the first- and second-derivative approximations derived in the theory tasks. For full-domain derivatives, use central formulas in the interior and second-order one-sided formulas at both endpoints.
#
# Before running refinement studies:
#
# - Verify array lengths, node locations, and the minimum number of nodes for each stencil.
# - Check exactness on constant, linear, and quadratic polynomials where expected. Include a cubic test to distinguish the boundary second-derivative formulas.
# - Verify the reflected right-boundary formulas separately.
# - Ensure that endpoint approximations use available domain values rather than silently wrapping indices. Periodic wrapping will be introduced explicitly in the transport calculation.
#
# ### Step 2 — Measure spatial approximation orders
#
# Use $\varphi(x)=e^x$ on $[0,1]$, whose first and second derivatives are also $e^x$. Use $N=17,33,65,129,257$, so that $h$ halves between successive meshes.
#
# 1. Compare forward, backward, and central first differences where their stencils are available. For a fair interior comparison, measure all three on the common set $i=1,\dots,N-2$.
# 2. Compare the compact central second derivative and the twice-applied central first derivative on their common valid set $i=2,\dots,N-3$.
# 3. For each approximation, tabulate $h$, the maximum absolute error, and $p_{\mathrm{obs}}$. Plot error against $h$ using logarithmic axes. Include reference lines with slopes one and two, clearly distinguished from measured data.
# 4. Test the full-domain first and second derivatives, including boundary formulas. Report interior, boundary, and overall maximum errors separately. Replace the second-order boundary formulas with the lower-order alternatives and explain the change in the overall error.
# 5. Repeat the compact second-derivative test with $\varphi(x)=x^2$. Explain why the observed-order estimator may fail here. Avoid taking logarithms of zero; report such cases explicitly rather than assigning an order.
#
# If desired, continue refinement until rounding becomes visible, but retain the original moderate-grid results. The approximation-order claim should be based on the range showing the predicted asymptotic behavior, not a fit across an error plateau.
#
# ### Step 3 — Construct the periodic transport problem
#
# Consider **pure convection**, rather than the convection–diffusion example animated in the lecture:
#
# $$
# \partial_t\varphi+v\partial_x\varphi=0,\qquad
# v=1,\qquad \varphi(x,0)=\sin(2\pi x),\qquad x\in[0,1].
# $$
#
# Use periodic continuation and the exact solution
#
# $$
# \varphi(x,t)=\sin\bigl(2\pi(x-vt)\bigr).
# $$
#
# Use $N$ distinct nodes $x_i=i\Delta x$, $i=0,\dots,N-1$, with $\Delta x=1/N$. Do not store $x=1$ as an independent unknown. For plotting only, repeat the first value at the right endpoint.
#
# 1. Implement the periodic semi-discrete right-hand sides for central and positive-velocity upwind convection. Use `numpy.roll` or explicitly wrapped indices, and verify which direction each shift represents.
# 2. Apply each operator to a constant field and check that the derivative is zero. Check that the discrete spatial mean of the right-hand side vanishes for a nonconstant field.
# 3. Compare your implementation with explicitly assembled periodic matrices for central and upwind convection using a small mesh with $N=8$. Verify that matrix–vector multiplication reproduces the corresponding right-hand-side function.
# 4. **Visualize both matrices side by side** as heatmaps, for example using Matplotlib's `imshow`. Label rows and columns with node indices $0,\dots,7$, annotate each cell with its coefficient, and include a colorbar. Use the same diverging color scale, centered at zero, for both matrices so that positive, negative, and zero entries can be compared directly. Inspect the diagonal, neighboring diagonals, and periodic corner entries. Which nodal values contribute to each row? How does the positive transport direction appear in the upwind matrix? Is either matrix symmetric? Check both row and column sums numerically; the heatmap complements these checks rather than replacing them.
# 5. Verify that the exact solution satisfies the PDE and periodic conditions. Explain why it provides a useful reference for the numerical calculation.
#
# ### Step 4 — Advance the periodic solution in time
#
# For the central spatial operator $\mathbf L$, implement Crank–Nicolson:
#
# $$
# \left(\mathbf I-\frac{\Delta t}{2}\mathbf L\right)
# \underline\varphi^{n+1}
# =\left(\mathbf I+\frac{\Delta t}{2}\mathbf L\right)
# \underline\varphi^n.
# $$
#
# For these small systems, `numpy.linalg.solve` is sufficient; do not explicitly form a matrix inverse. Check the linear-system residual. Efficient solvers will be treated later in the course.
#
# 1. Visualize the two Crank–Nicolson matrices for $N=8$ and $C=0.5$ using the labeled and annotated heatmap format from Step 3. Identify the identity contribution and the sign changes between the two matrices.
# 2. For $N=40$, run Crank–Nicolson with central convection and forward Euler with upwind convection to $T=1$, using $C=|v|\Delta t/\Delta x=0.5$. Plot both numerical solutions alongside the exact solution at $t=0,1/4,1/2,1$.
# 3. Calculate the maximum absolute nodal error at $T=1$ and check that the discrete spatial mean remains close to its initial value. Describe the differences between the computed profiles and the exact solution without separating the error into phase and amplitude contributions.
# 4. Repeat for $N=20,40,80,160$ with fixed $C=0.5$ and estimate the observed order of each **combined space–time scheme**. Since $\Delta t$ changes with $\Delta x$, these results do not isolate spatial or temporal error. Formal analysis of the different error contributions follows in a later lecture.
#
# The chosen time step is suitable for these two schemes. Do not transfer it to forward Euler with central convection: removing the physical diffusion present in the lecture example changes the stability properties.
#
# ### Step 5 — Approximation errors on nonuniform grids
#
# We now isolate spatial approximation errors by differentiating the known function $\varphi(x)=e^x$ on $[0,1]$, without time integration. Use $N$ nodes including both endpoints and compare two explicitly defined families of nonuniform meshes.
#
# **Smoothly graded meshes:** use $N=17,33,65,129,257$, uniform computational nodes $\xi_i=i/(N-1)$, and
#
# $$
# x_i=g(\xi_i),\qquad g(\xi)=\frac{e^{2\xi}-1}{e^2-1}.
# $$
#
# **Alternating-width meshes:** use $K=8,16,32,64,128$ pairs of intervals and $N=2K+1$ nodes. Define widths
#
# $$
# \Delta x_j=\begin{cases}
# 1/(3K),&j\text{ odd},\\
# 2/(3K),&j\text{ even},
# \end{cases}\qquad j=1,\dots,2K,
# $$
#
# and construct the nodes by cumulative summation from $x_0=0$. The widths sum to one. Here $K$ counts interval pairs; it is unrelated to the time-level index $n$.
#
# 1. Plot representative meshes side by side. For every refinement level, report $H=\max_j\Delta x_j$ and $S=\max_{1\leq i\leq N-2}|\Delta x_{i+1}-\Delta x_i|$. Estimate how $S$ scales with $H$ for each family.
# 2. Implement the weighted nonuniform central first derivative and the compact nonuniform second derivative derived in the lecture. Evaluate their maximum absolute errors on all interior nodes, using the analytical derivatives of $e^x$. Keep boundary approximations out of this comparison.
# 3. Tabulate errors and estimate orders using $\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})/\log(H_{\mathrm{coarse}}/H_{\mathrm{fine}})$. Use the measured ratio of $H$, rather than assuming it is exactly two for the mapped meshes.
# 4. Explain why the weighted first derivative can remain second order on both families, while the compact second derivative can have different orders. Relate the measurements to the products and differences of neighboring widths in the Taylor error formulas.
# 5. On the smoothly graded meshes, also approximate the first derivative using the chain-rule formula and the **analytical** Jacobian $J(\xi)=g'(\xi)$. Compare its errors and observed order with those of the direct weighted formula. Are the numerical values identical? Why does the same approximation order not imply identical coefficients or errors?
#
# As checks, both direct operators should differentiate constant and linear functions exactly up to rounding; the compact second derivative should also be exact for a quadratic. For the mapped formula, a physically linear function becomes nonlinear in $\xi$, so exactness at finite spacing is not generally expected. This is a useful distinction between the two constructions.
