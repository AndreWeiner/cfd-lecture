# %% [markdown]
# ---
# title: Finite difference method foundations
# subtitle: Lecture 3
# jupyter: python3
# ---
#
# ## Learning objectives
#
# After this lecture, we will be able to:
#
# - Describe the finite difference workflow and define one-dimensional meshes, grid widths, and nodal solution values, including periodic indexing.
# - Derive forward, backward, and central first-derivative approximations and compact second-derivative formulas using Taylor expansions; identify their leading errors and approximation orders.
# - Explain how stencil width, even–odd coupling, and nonuniform spacing affect derivative approximations, and relate coordinate-mapped derivatives through the chain rule.
# - Interpret time integration as an approximation of the mean rate of change and formulate forward Euler, backward Euler, Crank–Nicolson, and implicit midpoint updates.
# - Distinguish mean-rate, one-step, and global temporal errors, and explain the construction and starting-value requirements of AB2 and BDF2 multistep methods.
# - Construct central and upwind semi-discrete approximations of periodic convection–diffusion, explain numerical diffusion, and assemble the corresponding vector–matrix systems.
# - Combine spatial discretization and time integration in the weighted matrix update, interpret the Courant and diffusion numbers, and distinguish explicit updates from implicit linear-system solves.
# - Use the analytical traveling, decaying sine wave to interpret numerical damping and transport errors, recognizing that stability and convergence require further analysis.
#

# %% [markdown]
# ## What is the finite difference method?
#
# The finite difference method is one of the oldest numerical methods for solving partial differential equations. It provides a conceptual foundation for understanding other discretization approaches, including finite volume and finite element methods, as well as higher-order schemes.
#
# The finite difference method replaces a differential equation on a continuous domain by algebraic equations for approximate solution values at discrete points. We proceed in three steps:
#
# 1. **Discretize the domain.** Subdivide the computational domain into regular discrete elements (intervals in one dimension), whose endpoints define the mesh points or nodes.
# 2. **Represent the solution at the nodes.** Associate an unknown approximate solution value with each mesh point.
# 3. **Approximate the derivatives.** Replace derivatives in the differential equation by finite differences involving neighboring nodal values. Together with the initial and boundary conditions, these approximations produce equations for the discrete solution.
#
# We develop these ideas mainly for **one-dimensional problems**. Multidimensional finite difference methods are covered separately in Lecture 6, following the lectures on consistency, stability, and convergence and on conservation laws.
#

# %% [markdown]
# ## Discretizing a one-dimensional domain
#
# Consider the interval $\Omega=[a,b]$ with coordinate $x$. We subdivide it into $N-1$ intervals using $N$ ordered nodes,
#
# $$
# a=x_0<x_1<\dots<x_{N-1}=b.
# $$
#
# The **mesh**, or discrete representation of the domain, is the set of nodes
#
# $$
# \Omega_h=\{x_i\}_{i=0}^{N-1}.
# $$
#
# ![The continuous interval Omega and its discrete mesh, with nodes from x zero to x capital N minus one and the width between x i minus one and x i.](media/fd-mesh-light.svg){.light-content width=100%}
# ![The continuous interval Omega and its discrete mesh, with nodes from x zero to x capital N minus one and the width between x i minus one and x i.](media/fd-mesh-dark.svg){.dark-content width=100%}
#
# The **grid width** of interval $i$ is
#
# $$
# \Delta x_i=x_i-x_{i-1},\qquad i=1,\dots,N-1.
# $$
#
# For a **uniform mesh**, all intervals have the same width:
#
# $$
# \Delta x_i=h=\frac{b-a}{N-1},\qquad x_i=a+ih.
# $$
#
# Here $h$ is the discretization parameter. More generally, for a nonuniform mesh we can use $h=\max_i\Delta x_i$ to characterize its resolution. The sketch shows a uniform mesh; the definition of $\Delta x_i$ also applies to nonuniform meshes.
#
# For a generic exact solution $\varphi(x)$, we seek nodal approximations
#
# $$
# \varphi_i\approx\varphi(x_i),\qquad i=0,\dots,N-1.
# $$
#
# We denote the entire collection of discrete solution values by
#
# $$
# \varphi_h=\{\varphi_i\}_{i=0}^{N-1}.
# $$
#
# Each value $\varphi_i$ is associated with its node $x_i$. Thus $\Omega_h$ specifies **where** we approximate the solution, while $\varphi_h$ collects its approximate values. For time-dependent problems, the nodal values also depend on time; we first focus on the spatial approximation.
#

# %% [markdown]
# ## Approximating first derivatives
#
# ### From a limit to a finite difference
#
# For a differentiable function, the first derivative at $x_i$ is defined by
#
# $$
# \varphi'(x_i)=\lim_{h\to0}\frac{\varphi(x_i+h)-\varphi(x_i)}{h}.
# $$
#
# On a mesh, we retain a finite spacing instead of taking the limit. For the following formulas, let $h>0$ be the constant spacing of a uniform mesh, so that $x_{i\pm1}=x_i\pm h$.
#
# ### Forward difference
#
# Using the node to the right of $x_i$, we define the **forward-difference operator** $D^+$ by
#
# $$
# D^+\varphi_i:=\frac{\varphi_{i+1}-\varphi_i}{h}
# \approx\varphi'(x_i),\qquad i=0,\dots,N-2.
# $$
#
# ### Backward difference
#
# The same derivative can be written as
#
# $$
# \varphi'(x_i)=\lim_{h\to0^+}\frac{\varphi(x_i)-\varphi(x_i-h)}{h}.
# $$
#
# Using the node to the left of $x_i$, we define the **backward-difference operator** $D^-$ by
#
# $$
# D^-\varphi_i:=\frac{\varphi_i-\varphi_{i-1}}{h}
# \approx\varphi'(x_i),\qquad i=1,\dots,N-1.
# $$
#
# Both expressions approximate the slope at $x_i$ using two neighboring nodal values. Their index ranges reflect which neighbors are available on the mesh.
#
# **How good are these approximations for a finite grid width $h$?** To calculate the exact error of a derivative approximation directly, we would need to know the true underlying function and its derivative. In a numerical solution of a differential equation, that function is generally unknown. However, assuming sufficient smoothness, we can represent it locally by a Taylor polynomial plus a remainder. This allows us to derive how the approximation error depends on the grid width and on derivatives of the underlying function, even without knowing the function explicitly.
#

# %% [markdown]
# ## Taylor expansion and approximation error
#
# ### Building an approximation term by term
#
# A Taylor polynomial approximates a smooth function near an expansion point $x_i$ using the function value and its derivatives at that point. Retaining the first four terms gives
#
# $$
# \varphi(x_i+h)=\varphi(x_i)
# +h\varphi'(x_i)
# +\frac{h^2}{2!}\varphi''(x_i)
# +\frac{h^3}{3!}\varphi'''(x_i)
# +R_4(h).
# $$
#
# These terms account for the function value, slope, curvature, and third derivative at $x_i$; $R_4(h)$ collects the remainder. In general, retaining terms through degree $m$ gives
#
# $$
# \varphi(x_i+h)=\sum_{k=0}^{m}\frac{h^k}{k!}\varphi^{(k)}(x_i)+R_{m+1}(h).
# $$
#
# Here $\varphi^{(0)}=\varphi$, and $R_{m+1}$ is the remainder. If $\varphi$ has $m+1$ continuous derivatives between $x_i$ and $x_i+h$, Taylor's theorem gives
#
# $$
# R_{m+1}(h)=\frac{h^{m+1}}{(m+1)!}\varphi^{(m+1)}(\xi),
# $$
#
# for some point $\xi$ between $x_i$ and $x_i+h$. A bounded derivative therefore gives $R_{m+1}(h)=\mathcal O(h^{m+1})$ as $h\to0$.
#
# In practical terms, the error decreases predictably as the distance $h$ from the expansion point becomes smaller. We generally do not know the exact location of $\xi$, but we do not need it: if the magnitude of the $(m+1)$-st derivative remains below a constant $M$ throughout the interval, then the remainder is bounded by $M|h|^{m+1}/(m+1)!$. The notation $\mathcal O(h^{m+1})$ means that, for sufficiently small $h$, the error is bounded by a constant times $|h|^{m+1}$.
#
# For example, retaining the first four terms gives a cubic Taylor polynomial with remainder $\mathcal O(h^4)$. Halving $h$ reduces this error bound by a factor of $2^4=16$, provided the same derivative bound applies. The actual error typically follows this scaling when the fourth-order term dominates.
#
# The animation uses the dimensionless example $\varphi(x)=e^x$ about $x_i=1$. The powers therefore involve the distance $x-1$ from the expansion point:
#
# $$
# T_m(x)=\varphi(1)+(x-1)\varphi'(1)+\frac{(x-1)^2}{2!}\varphi''(1)+\dots+\frac{(x-1)^m}{m!}\varphi^{(m)}(1).
# $$
#
# For $\varphi(x)=e^x$, all derivative values $\varphi^{(k)}(1)$ equal $e$.
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Taylor polynomials of increasing degree approximating the exponential function" autoplay muted><source src="media/fd-taylor-light.webm" type="video/webm"><source src="media/fd-taylor-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Taylor polynomials of increasing degree approximating the exponential function" autoplay muted><source src="media/fd-taylor-dark.webm" type="video/webm"><source src="media/fd-taylor-dark.mp4" type="video/mp4"></video>
#
# The constant term matches the function value. Adding the linear term also matches its slope; the quadratic term incorporates curvature. Further terms use higher derivatives. For this example, retaining more terms improves the approximation over the plotted interval. In general, a finite Taylor expansion is a local statement controlled by its remainder; an infinite Taylor series does not necessarily converge to the function.
#
# ### Expanding about a mesh point
#
# For a sufficiently smooth solution and a small increment $h$, expansion about $x_i$ gives
#
# $$
# \varphi(x_i+h)=\varphi(x_i)
# +h\varphi'(x_i)
# +\frac{h^2}{2}\varphi''(x_i)
# +\frac{h^3}{6}\varphi'''(x_i)
# +\mathcal O(h^4).
# $$
#
# Subtracting $\varphi(x_i)$ and dividing by $h$ yields
#
# $$
# \frac{\varphi(x_i+h)-\varphi(x_i)}{h}
# =\varphi'(x_i)
# +\frac{h}{2}\varphi''(x_i)
# +\frac{h^2}{6}\varphi'''(x_i)
# +\mathcal O(h^3).
# $$
#
# Rearranging for the exact derivative, we obtain
#
# $$
# \boxed{
# \varphi'(x_i)=
# \frac{\varphi(x_i+h)-\varphi(x_i)}{h}
# -\frac{h}{2}\varphi''(x_i)
# -\frac{h^2}{6}\varphi'''(x_i)
# +\mathcal O(h^3).
# }
# $$
#
# The first term on the right is the forward difference applied to **exact nodal samples**. The remaining terms show the error introduced when we replace the derivative by that difference. Defining the difference error as the approximation minus the exact derivative gives
#
# $$
# \frac{\varphi(x_i+h)-\varphi(x_i)}{h}-\varphi'(x_i)
# =\frac{h}{2}\varphi''(x_i)+\mathcal O(h^2).
# $$
#
# The leading error is proportional to both $h$ and the local second derivative $\varphi''(x_i)$. Thus, local curvature controls the leading error coefficient: stronger curvature gives a larger leading error at the same spacing. An approximation whose leading error scales proportionally to $h$ is said to be **first-order accurate**. The forward difference therefore has first-order accuracy. For a smooth function with a nonzero leading error coefficient, halving $h$ approximately halves this error once $h$ is sufficiently small.
#
# This is the **derivative approximation error evaluated on the exact solution**. The computed nodal values $\varphi_i$ also contain solution errors; those errors are distinct from the Taylor remainder derived here.
#
# ### Applying the same procedure to the backward difference
#
# For the backward difference, we follow the same steps, but expand $\varphi(x_i-h)$ about $x_i$. Replacing $h$ by $-h$ changes the signs of the odd-power terms:
#
# $$
# \varphi(x_i-h)=\varphi(x_i)-h\varphi'(x_i)
# +\frac{h^2}{2}\varphi''(x_i)
# -\frac{h^3}{6}\varphi'''(x_i)+\mathcal O(h^4).
# $$
#
# Subtracting this expansion from $\varphi(x_i)$ and dividing by $h$ gives
#
# $$
# \frac{\varphi(x_i)-\varphi(x_i-h)}{h}
# =\varphi'(x_i)-\frac{h}{2}\varphi''(x_i)
# +\frac{h^2}{6}\varphi'''(x_i)+\mathcal O(h^3).
# $$
#
# Thus, using the same definition of error (approximation minus exact derivative),
#
# $$
# \frac{\varphi(x_i)-\varphi(x_i-h)}{h}-\varphi'(x_i)
# =-\frac{h}{2}\varphi''(x_i)+\mathcal O(h^2).
# $$
#
# The backward difference is likewise said to be **first-order accurate**, meaning that its leading error scales proportionally to $h$. Its leading error is proportional to both $h$ and the local second derivative $\varphi''(x_i)$, with the opposite sign to that of the forward difference. For a locally linear function, both one-sided formulas are exact; when the second derivative vanishes only at a point, higher-order terms determine the error there.

# %% [markdown]
# ## Central differences on a nonuniform mesh
#
# ### Combining forward and backward differences
#
# The leading errors of the forward and backward differences have opposite signs. This suggests combining the two approximations so that their second-derivative contributions cancel. On a nonuniform mesh, the two adjacent widths are
#
# $$
# \Delta x_i=x_i-x_{i-1},\qquad
# \Delta x_{i+1}=x_{i+1}-x_i.
# $$
#
# Both widths are positive. We extend the one-sided operators to these local spacings:
#
# $$
# D^+\varphi_i=\frac{\varphi_{i+1}-\varphi_i}{\Delta x_{i+1}},
# \qquad
# D^-\varphi_i=\frac{\varphi_i-\varphi_{i-1}}{\Delta x_i}.
# $$
#
# To analyze their errors, we temporarily apply them to exact samples, denoted by $\varphi_i^*=\varphi(x_i)$. Taylor expansion about $x_i$ gives
#
# $$
# \begin{aligned}
# D^+\varphi_i^*&=\varphi'(x_i)
# +\frac{\Delta x_{i+1}}{2}\varphi''(x_i)
# +\frac{\Delta x_{i+1}^2}{6}\varphi'''(x_i)
# +\mathcal O(\Delta x_{i+1}^3),\\
# D^-\varphi_i^*&=\varphi'(x_i)
# -\frac{\Delta x_i}{2}\varphi''(x_i)
# +\frac{\Delta x_i^2}{6}\varphi'''(x_i)
# +\mathcal O(\Delta x_i^3).
# \end{aligned}
# $$
#
# Consider a weighted combination $\alpha D^+ +\beta D^-$. Multiplying the two Taylor expansions by their respective weights and adding them gives
#
# $$
# \begin{aligned}
# \alpha D^+\varphi_i^*+\beta D^-\varphi_i^*
# ={}&(\alpha+\beta)\varphi'(x_i)\\
# &+\frac{\alpha\Delta x_{i+1}-\beta\Delta x_i}{2}\varphi''(x_i)\\
# &+\frac{\alpha\Delta x_{i+1}^2+\beta\Delta x_i^2}{6}\varphi'''(x_i)
# +\text{higher-order terms}.
# \end{aligned}
# $$
#
# Both one-sided expansions contain $\varphi'(x_i)$ with coefficient one. Their weighted sum therefore contains it with coefficient $\alpha+\beta$. We want to approximate the derivative itself, so this coefficient must equal one: $\alpha+\beta=1$.
#
# The second-derivative contributions have opposite signs. After weighting, their combined coefficient is $(\alpha\Delta x_{i+1}-\beta\Delta x_i)/2$. To remove this leading error contribution for a general smooth function, we set its coefficient to zero. We therefore require
#
# $$
# \alpha+\beta=1,\qquad
# \alpha\Delta x_{i+1}-\beta\Delta x_i=0.
# $$
#
# Hence
#
# $$
# \alpha=\frac{\Delta x_i}{\Delta x_i+\Delta x_{i+1}},\qquad
# \beta=\frac{\Delta x_{i+1}}{\Delta x_i+\Delta x_{i+1}}.
# $$
#
# We define the **central-difference operator** $D^0$ at an interior node, $i=1,\dots,N-2$, by
#
# $$
# \boxed{
# D^0\varphi_i:=
# \frac{\Delta x_i}{\Delta x_i+\Delta x_{i+1}}D^+\varphi_i
# +\frac{\Delta x_{i+1}}{\Delta x_i+\Delta x_{i+1}}D^-\varphi_i.
# }
# $$
#
# Substituting the one-sided differences gives the three-point formula
#
# $$
# D^0\varphi_i=
# -\frac{\Delta x_{i+1}}{\Delta x_i(\Delta x_i+\Delta x_{i+1})}\varphi_{i-1}
# +\frac{\Delta x_{i+1}-\Delta x_i}{\Delta x_i\Delta x_{i+1}}\varphi_i
# +\frac{\Delta x_i}{\Delta x_{i+1}(\Delta x_i+\Delta x_{i+1})}\varphi_{i+1}.
# $$
#
# On a nonuniform mesh, the weights must account for the unequal spacings. A simple arithmetic average $(D^++D^-)/2$ would leave the second-derivative contribution $(\Delta x_{i+1}-\Delta x_i)\varphi''(x_i)/4$ and is not generally second-order accurate.
#
# ### Error analysis
#
# Inserting the Taylor expansions into the weighted combination cancels the second-derivative terms. The coefficient multiplying the third derivative is
#
# $$
# \frac{\Delta x_i\Delta x_{i+1}^2+\Delta x_{i+1}\Delta x_i^2}
# {6(\Delta x_i+\Delta x_{i+1})}
# =\frac{\Delta x_i\Delta x_{i+1}}{6}.
# $$
#
# Consequently, for a sufficiently smooth function,
#
# $$
# D^0\varphi_i^*-\varphi'(x_i)
# =\frac{\Delta x_i\Delta x_{i+1}}{6}\varphi'''(x_i)
# +\mathcal O(H_i^3),
# \qquad H_i=\max(\Delta x_i,\Delta x_{i+1}).
# $$
#
# The leading truncation error is proportional to the **product of the neighboring grid widths** and to the **third derivative**. Since $\Delta x_i\Delta x_{i+1}\leq H_i^2$, this approximation is second-order accurate as both spacings tend to zero, provided the required derivatives remain bounded. If both spacings are halved together and the leading coefficient is nonzero, the leading error decreases by a factor of four.
#
# ### Uniform-grid limit
#
# For $\Delta x_i=\Delta x_{i+1}=h$, both weights equal $1/2$. The central difference becomes
#
# $$
# \boxed{
# D^0\varphi_i=\frac12(D^+\varphi_i+D^-\varphi_i)
# =\frac{\varphi_{i+1}-\varphi_{i-1}}{2h}.
# }
# $$
#
# On this symmetric stencil, the even-power terms in the Taylor expansions of $\varphi(x_i+h)$ and $\varphi(x_i-h)$ cancel when we subtract them. For a function with sufficiently many bounded derivatives,
#
# $$
# D^0\varphi_i^*-\varphi'(x_i)
# =\frac{h^2}{6}\varphi'''(x_i)+\mathcal O(h^4).
# $$
#
# Thus, the central approximation has a leading error proportional to **$h^2$ and $\varphi'''(x_i)$**, whereas the one-sided approximations have leading errors proportional to **$h$ and $\varphi''(x_i)$**. The central formula is exact for quadratic polynomials, whose third and higher derivatives vanish.

# %% [markdown]
# ## Comparing first-derivative approximations
#
# The sketch summarizes all three approximations on a uniform mesh using exact samples of a dimensionless example function. The forward and backward differences are slopes of secants joining $x_i$ to its right or left neighbor. The central difference is the slope of the secant joining the two outer nodes, $x_{i-1}$ and $x_{i+1}$. The dotted line is the exact tangent at $x_i$.
#
# ![Forward, backward, and central secants compared with the exact tangent at the middle node of a uniform mesh.](media/fd-first-derivatives-light.svg){.light-content width=100%}
# ![Forward, backward, and central secants compared with the exact tangent at the middle node of a uniform mesh.](media/fd-first-derivatives-dark.svg){.dark-content width=100%}
#
# The central secant does not generally pass through the middle sample: its **slope** approximates the derivative there. For the convex exponential shown, the backward slope underestimates the exact derivative and the forward slope overestimates it. Their average gives the central slope, which is closer to the exact derivative in this example. On a nonuniform mesh, we instead use the weighted combination derived above; the simple chord between the outer nodes does not generally give that operator.

# %% [markdown]
# ## Approximating second derivatives
#
# ### The derivative of a derivative
#
# The second derivative is the first derivative of the first derivative:
#
# $$
# \varphi''(x)=\frac{\mathrm d}{\mathrm dx}\left(\frac{\mathrm d\varphi}{\mathrm dx}\right).
# $$
#
# This suggests applying our discrete first-derivative operators successively. Since we have forward, backward, and central operators, several combinations are possible. Their accuracy, stencil width, and coupling between nodes can differ. We first assume a uniform mesh with constant spacing $h$.
#
# ### Applying the central operator twice
#
# Among the first-derivative approximations considered above, the central difference achieved the highest order of accuracy: second order, compared with first order for the forward and backward differences. This motivates using the central operator for both successive differentiations when approximating the second derivative.
#
# Apply $D^0$ to the discrete first derivative $D^0\varphi_i$:
#
# $$
# \begin{aligned}
# D^0D^0\varphi_i
# &=\frac{D^0\varphi_{i+1}-D^0\varphi_{i-1}}{2h}\\
# &=\frac{1}{2h}\left(
# \frac{\varphi_{i+2}-\varphi_i}{2h}
# -\frac{\varphi_i-\varphi_{i-2}}{2h}\right)\\
# &=\boxed{\frac{\varphi_{i+2}-2\varphi_i+\varphi_{i-2}}{4h^2}}.
# \end{aligned}
# $$
#
# Here the rightmost operator acts first. The formula uses nodes two grid spacings away on either side. It spans **five grid positions**, from $i-2$ to $i+2$, although the coefficients at $i-1$ and $i+1$ vanish: only three nodes contribute.
#
# To analyze the truncation error, we again insert exact function samples $\varphi_i^*=\varphi(x_i)$. Adding Taylor expansions at offsets $+2h$ and $-2h$ cancels the odd-power terms:
#
# $$
# \varphi(x_i+2h)+\varphi(x_i-2h)-2\varphi(x_i)
# =4h^2\varphi''(x_i)
# +\frac{4h^4}{3}\varphi^{(4)}(x_i)
# +\mathcal O(h^6).
# $$
#
# Dividing by $4h^2$ gives
#
# $$
# D^0D^0\varphi_i^*-
# \varphi''(x_i)
# =\frac{h^2}{3}\varphi^{(4)}(x_i)+\mathcal O(h^4).
# $$
#
# The approximation is therefore **second-order accurate**: its leading error scales proportionally to $h^2$. The leading error also depends on the **fourth derivative** of the underlying function. For second-derivative approximations, it is this fourth derivative that appears here; the third derivative appeared in our central approximation of the first derivative.
#
# There are two potential disadvantages:
#
# - **Boundary treatment.** The formula requires $i-2$ and $i+2$, so it applies directly only for $i=2,\dots,N-3$. At interior nodes next to the boundaries, we need a different formula or additional values outside the domain, with suitable relations to impose the boundary conditions.
# - **Even–odd decoupling.** At an even-indexed node, the formula uses only even-indexed values; at an odd-indexed node, it uses only odd-indexed values. Thus this operator alone does not couple the two interlaced grids. For example, the alternating sequence $\varphi_i=(-1)^i$ gives $D^0D^0\varphi_i=0$ despite its rapid variation. This can allow spurious alternating, or checkerboard, modes if other parts of the discretization and boundary treatment do not control them.
#
# ### Combining forward and backward differences
#
# We can instead apply $D^-$ to $D^+\varphi_i$:
#
# $$
# \begin{aligned}
# D^-D^+\varphi_i
# &=\frac{D^+\varphi_i-D^+\varphi_{i-1}}{h}\\
# &=\frac{1}{h}\left(
# \frac{\varphi_{i+1}-\varphi_i}{h}
# -\frac{\varphi_i-\varphi_{i-1}}{h}\right)\\
# &=\boxed{\frac{\varphi_{i+1}-2\varphi_i+\varphi_{i-1}}{h^2}}.
# \end{aligned}
# $$
#
# On a uniform mesh, reversing the order gives the same result: $D^+D^-=D^-D^+$. This is the usual **compact three-point approximation** of the second derivative. It couples neighboring nodes and applies at every interior node, $i=1,\dots,N-2$, using prescribed boundary values where appropriate.
#
# We already derived this same stencil in the belt-flow example of [Lecture 1](lecture_01.py#example-flow-between-a-belt-and-a-stationary-wall) and used it in [Exercise 1, Step 6](exercise_01.py#step-6-implement-and-assess-the-belt-flow-solution). There, the reduced momentum equation is a one-dimensional Poisson problem, $\mu u''(y)=\Delta p/L$, with prescribed velocities at the two walls.
#
# For the error analysis, adding the Taylor expansions at offsets $+h$ and $-h$ gives
#
# $$
# \varphi(x_i+h)+\varphi(x_i-h)-2\varphi(x_i)
# =h^2\varphi''(x_i)
# +\frac{h^4}{12}\varphi^{(4)}(x_i)
# +\mathcal O(h^6).
# $$
#
# Hence
#
# $$
# D^-D^+\varphi_i^*-
# \varphi''(x_i)
# =\frac{h^2}{12}\varphi^{(4)}(x_i)+\mathcal O(h^4).
# $$
#
# This approximation is also **second-order accurate**, with a leading error proportional to $h^2$ and the **fourth derivative**. At the same spacing $h$, its leading error coefficient is one quarter of that for $D^0D^0$. It also uses a narrower stencil and does not have the same even–odd decoupling: for $\varphi_i=(-1)^i$, it gives $D^-D^+\varphi_i=-4\varphi_i/h^2$ rather than zero.
#
# ### Connection to the mesh-refinement results in Exercise 1
#
# The fourth derivative in the error formula explains a potentially surprising observation from Exercise 1. The exact belt-flow velocity profile is quadratic in $y$, so its fourth and all higher derivatives vanish. The three-point second-derivative formula is therefore **exact for this profile**, rather than merely second-order accurate.
#
# With the prescribed boundary values and constant source term represented exactly, the exact nodal samples satisfy the discrete Poisson equations. Since this Dirichlet system has a unique solution, solving it in exact arithmetic would recover the exact velocity at every node on any uniform mesh.
#
# In floating-point arithmetic, the measured nodal error can consequently remain almost zero, near the numerical precision of the calculation, instead of showing the usual decrease proportional to $h^2$. The remaining error comes from rounding, parameter representation, matrix assembly, the linear-system solve, and evaluation of the analytical reference. It need not be strictly constant and can increase on finer meshes as the system becomes more ill-conditioned. Thus a nearly flat error curve at a very small level is consistent with an exact stencil for this particular solution; it does not contradict the general second-order error analysis.
#
# This exactness concerns the **nodal values**. Straight segments joining the nodes still differ from the quadratic analytical profile between nodes, and numerical integration of that profile can introduce its own error, as investigated in the mass-flow calculation of Exercise 1.
#
# ### What changes on a nonuniform mesh?
#
# When $\Delta x_i$ and $\Delta x_{i+1}$ differ, we must account for where the intermediate derivative approximations are located. The neighboring interval slopes
#
# $$
# s_{i-1/2}=\frac{\varphi_i-\varphi_{i-1}}{\Delta x_i},\qquad
# s_{i+1/2}=\frac{\varphi_{i+1}-\varphi_i}{\Delta x_{i+1}}
# $$
#
# approximate the first derivative at the respective interval midpoints. Those midpoints are separated by $(\Delta x_i+\Delta x_{i+1})/2$. Taking their slope difference over this distance yields the three-point formula
#
# $$
# L_h\varphi_i:=\frac{2}{\Delta x_i+\Delta x_{i+1}}
# \left(
# \frac{\varphi_{i+1}-\varphi_i}{\Delta x_{i+1}}
# -\frac{\varphi_i-\varphi_{i-1}}{\Delta x_i}
# \right).
# $$
#
# On a uniform mesh, $L_h$ reduces to $D^-D^+$. On a nonuniform mesh, this construction uses the midpoint separation as its outer denominator; simply composing the nodal one-sided operators with their earlier definitions does not in general produce this formula.
#
# Taylor expansion about $x_i$ gives, with $H_i=\max(\Delta x_i,\Delta x_{i+1})$,
#
# $$
# \begin{aligned}
# L_h\varphi_i^*-\varphi''(x_i)
# ={}&\frac{\Delta x_{i+1}-\Delta x_i}{3}\varphi'''(x_i)\\
# &+\frac{\Delta x_i^2-\Delta x_i\Delta x_{i+1}+\Delta x_{i+1}^2}{12}
# \varphi^{(4)}(x_i)+\mathcal O(H_i^3).
# \end{aligned}
# $$
#
# The unequal spacings leave a **third-derivative term** that cancels on a uniform mesh. The fourth-derivative contribution scales as $H_i^2$, but the third-derivative contribution scales with the difference $\Delta x_{i+1}-\Delta x_i$. The overall accuracy therefore depends on how this difference decreases as we refine the mesh.
#
# For example, if $\Delta x_i=h$ and $\Delta x_{i+1}=2h$, their difference is $h$. The leading error is then $h\varphi'''(x_i)/3$: halving both widths only halves this contribution, so the approximation is generally **first-order accurate**. By contrast, if $\Delta x_i=h$ and $\Delta x_{i+1}=h+Ch^2$, with fixed $C$, their difference is $Ch^2$. Both displayed error contributions then scale as $h^2$, giving **second-order accuracy**.
#
# Thus, second-order accuracy is recovered when neighboring widths differ by $\mathcal O(H_i^2)$, or when they are equal. Such behavior occurs on suitably smooth graded meshes, where neighboring cell widths become increasingly similar under refinement. Unequal spacing alone does not imply first-order accuracy; what matters is how the difference between neighboring widths scales during refinement. These statements assume bounded derivatives and a nonzero leading error coefficient; if $\varphi'''(x_i)=0$, the third-derivative contribution vanishes at that node.

# %% [markdown]
# ## Mapping a nonuniform grid to a uniform grid
#
# Mapping provides a systematic way to separate geometry from differentiation, especially for higher-order methods and multidimensional structured grids. Direct finite difference formulas remain attractive for simple one-dimensional problems.
#
# ### Physical and computational coordinates
#
# A general strategy for handling nonuniform grids is to introduce a **computational coordinate** $\xi$ in which the nodes are uniformly spaced. We perform finite differences on this uniform grid and use the chain rule to relate derivatives in $\xi$ to derivatives in the physical coordinate $x$.
#
# Let a smooth, invertible mapping
#
# $$
# x=g(\xi),\qquad 0\leq\xi\leq1,\qquad
# g(0)=a,\quad g(1)=b
# $$
#
# describe the physical domain. We assume $g'(\xi)>0$, so the node order is preserved and the mapping has no singular points. Using $N$ computational nodes,
#
# $$
# \Delta\xi=\frac{1}{N-1},\qquad \xi_i=i\Delta\xi,
# \qquad x_i=g(\xi_i),\qquad i=0,\dots,N-1.
# $$
#
# The corresponding meshes are
#
# $$
# \widehat\Omega_{\Delta\xi}=\{\xi_i\}_{i=0}^{N-1},\qquad
# \Omega_h=\{x_i\}_{i=0}^{N-1}.
# $$
#
# ![Uniform computational mesh and nonuniform physical mesh linked node by node by a smooth coordinate mapping.](media/fd-mapping-light.svg){.light-content width=100%}
# ![Uniform computational mesh and nonuniform physical mesh linked node by node by a smooth coordinate mapping.](media/fd-mapping-dark.svg){.dark-content width=100%}
#
# The sketch uses the dimensionless example $g(\xi)=(e^{2\xi}-1)/(e^2-1)$. Equal computational intervals map to progressively larger physical intervals. The inverse mapping $\xi=g^{-1}(x)$ maps the physical nodes back to the uniform computational grid.
#
# We express the same solution in computational coordinates as
#
# $$
# \widehat\varphi(\xi)=\varphi(g(\xi)),\qquad
# \widehat\varphi(\xi_i)=\varphi(x_i).
# $$
#
# The nodal values are unchanged by the coordinate transformation; their coordinate labels change. Derivatives, however, measure changes in the solution relative to a coordinate increment. To relate derivatives on the two grids, we therefore need to know how a small computational increment $\mathrm d\xi$ translates into a physical increment $\mathrm dx$. This local stretching is described by the **mapping Jacobian**
#
# $$
# J(\xi)=\frac{\mathrm dx}{\mathrm d\xi}=g'(\xi)>0.
# $$
#
# It measures the local stretching of the physical grid relative to the computational grid. The physical width is exactly $\Delta x_i=\int_{\xi_{i-1}}^{\xi_i}J(\xi)\,\mathrm d\xi$.
#
# ### First derivatives: a geometry-dependent factor
#
# The chain rule gives
#
# $$
# \frac{\mathrm d\widehat\varphi}{\mathrm d\xi}
# =\frac{\mathrm d\varphi}{\mathrm dx}\frac{\mathrm dx}{\mathrm d\xi},
# \qquad
# \boxed{\frac{\mathrm d\varphi}{\mathrm dx}
# =\frac{1}{J}\frac{\mathrm d\widehat\varphi}{\mathrm d\xi}}.
# $$
#
# Thus, we differentiate on the uniform computational grid and multiply by the geometry-dependent factor $1/J$. For example, central differencing gives
#
# $$
# \varphi'(x_i)\approx
# \frac{1}{J_i}\frac{\varphi_{i+1}-\varphi_{i-1}}{2\Delta\xi},
# \qquad J_i=J(\xi_i).
# $$
#
# For a fixed smooth mapping with $J$ bounded away from zero, exact metric values, and sufficiently smooth $\widehat\varphi$, this approximation has error $\mathcal O(\Delta\xi^2)$.
#
# ### Second derivatives: differentiate the geometry factor too
#
# Applying the transformed derivative a second time yields
#
# $$
# \boxed{
# \frac{\mathrm d^2\varphi}{\mathrm dx^2}
# =\frac{1}{J}\frac{\mathrm d}{\mathrm d\xi}
# \left(\frac{1}{J}\frac{\mathrm d\widehat\varphi}{\mathrm d\xi}\right)
# =\frac{1}{J^2}\frac{\mathrm d^2\widehat\varphi}{\mathrm d\xi^2}
# -\frac{J_\xi}{J^3}\frac{\mathrm d\widehat\varphi}{\mathrm d\xi},
# \qquad J_\xi=\frac{\mathrm dJ}{\mathrm d\xi}=g''(\xi).
# }
# $$
#
# For a nonlinear mapping, $J$ varies with position. We must therefore differentiate $1/J$ as well; multiplying the computational second derivative by $1/J^2$ alone would omit a term.
#
# Using the compact central second derivative and central first derivative on the uniform computational grid gives
#
# $$
# \varphi''(x_i)\approx
# \frac{1}{J_i^2}\frac{\varphi_{i+1}-2\varphi_i+\varphi_{i-1}}{\Delta\xi^2}
# -\frac{(J_\xi)_i}{J_i^3}\frac{\varphi_{i+1}-\varphi_{i-1}}{2\Delta\xi}.
# $$
#
# With the same smoothness and nondegeneracy assumptions, and exact metric values, both terms are second-order accurate in $\Delta\xi$. For a linear mapping, $J$ is constant and $J_\xi=0$, recovering the uniform physical-grid formulas with $h=J\Delta\xi$.
#
# ### Geometry and accuracy
#
# The mapping separates grid geometry from the difference formulas: the computational grid is uniform, while $J$ and $J_\xi$ account for physical stretching. If the mapping is known analytically, we can evaluate these factors directly. If only physical node coordinates are available, we must construct a suitable smooth mapping or approximate its derivatives consistently on the computational grid, for example
#
# $$
# J_i\approx\frac{x_{i+1}-x_{i-1}}{2\Delta\xi},\qquad
# (J_\xi)_i\approx\frac{x_{i+1}-2x_i+x_{i-1}}{\Delta\xi^2}.
# $$
#
# These geometry approximations and the boundary formulas must be included in the accuracy assessment. A fixed smooth mapping with nonzero bounded $J$ produces neighboring physical widths that differ by $\mathcal O(\Delta\xi^2)$, connecting this strategy to the smooth-grid refinement discussed above. Merely assigning uniform computational indices to an arbitrary irregular mesh does not guarantee a smooth mapping or second-order accuracy.

# %% [markdown]
# ## Time integration
#
# ### A general evolution problem
#
# We now consider a time-dependent field $\varphi(x,t)$ governed by
#
# $$
# \dot\varphi=\frac{\partial\varphi}{\partial t}=f[\varphi].
# $$
#
# The right-hand side represents the spatial terms and any sources. For example, constant-velocity convection and constant-coefficient diffusion give
#
# $$
# f[\varphi]=-v\frac{\partial\varphi}{\partial x}
# +\Gamma\frac{\partial^2\varphi}{\partial x^2}.
# $$
#
# After spatial discretization, these terms become a function of the nodal solution values. We write the resulting evolution equation schematically as $\dot\varphi=f(\varphi)$; here $\varphi$ may represent the entire discrete solution. The following formulas apply componentwise. Boundary data must also be included consistently. For clarity, we consider an autonomous right-hand side; explicit time dependence can be included as $f(\varphi,t)$.
#
# Let $t^n$ and $t^{n+1}$ be consecutive time levels, with
#
# $$
# \Delta t=t^{n+1}-t^n>0,\qquad \varphi^n\approx\varphi(t^n).
# $$
#
# The lowercase index $n$ counts time steps. A spatial and temporal sample is denoted by $\varphi_i^n\approx\varphi(x_i,t^n)$.
#
# ### The time step as an average rate of change
#
# Integrating both sides of the evolution equation over one time interval gives the exact identity
#
# $$
# \int_{t^n}^{t^{n+1}}\dot\varphi(t)\,\mathrm dt
# =\int_{t^n}^{t^{n+1}}f(\varphi(t))\,\mathrm dt,
# $$
#
# and hence
#
# $$
# \varphi(t^{n+1})-\varphi(t^n)
# =\int_{t^n}^{t^{n+1}}f(\varphi(t))\,\mathrm dt.
# $$
#
# Dividing by $\Delta t$, we obtain
#
# $$
# \boxed{
# \frac{\varphi(t^{n+1})-\varphi(t^n)}{\Delta t}
# =\overline f^{\,n+1/2},
# \qquad
# \overline f^{\,n+1/2}:=
# \frac{1}{\Delta t}\int_{t^n}^{t^{n+1}}f(\varphi(t))\,\mathrm dt.
# }
# $$
#
# In words, the change over a time step equals the step duration times the **mean right-hand side along the solution trajectory**. The superscript $n+1/2$ labels the interval mean; it does not imply that this mean equals the instantaneous midpoint value.
#
# We do not know the solution throughout the interval, so we approximate this mean. A numerical time step therefore has the form
#
# $$
# \frac{\varphi^{n+1}-\varphi^n}{\Delta t}
# \approx\overline f^{\,n+1/2}.
# $$
#
# Different time-integration methods correspond to different ways of evaluating this average.
#
# ### Four approximations of the mean
#
# ![Four panels comparing left-endpoint, right-endpoint, trapezoidal, and midpoint approximations of the time integral of the right-hand side.](media/fd-time-integration-light.svg){.light-content width=100%}
# ![Four panels comparing left-endpoint, right-endpoint, trapezoidal, and midpoint approximations of the time integral of the right-hand side.](media/fd-time-integration-dark.svg){.dark-content width=100%}
#
# The sketch shows an illustrative scalar history $f(\varphi(t))$, with $f^n=f(\varphi(t^n))$ and similarly for $f^{n+1}$. The shaded area in each panel approximates its time integral; dividing this area by $\Delta t$ gives the approximate mean. The dotted horizontal line marks the exact mean for the illustrated history. In a numerical calculation, the required solution states are themselves approximated.
#
# #### Forward Euler: evaluate at the beginning
#
# Approximate the mean by the known rate at $t^n$:
#
# $$
# \overline f^{\,n+1/2}\approx f(\varphi^n),\qquad
# \boxed{\varphi^{n+1}=\varphi^n+\Delta t\,f(\varphi^n)}.
# $$
#
# The right-hand side uses only the known solution, so the method is **explicit**. Forward Euler is first-order accurate in time under the usual smoothness and stability assumptions.
#
# #### Backward Euler: evaluate at the end
#
# Approximate the mean by the rate at $t^{n+1}$:
#
# $$
# \overline f^{\,n+1/2}\approx f(\varphi^{n+1}),\qquad
# \boxed{\varphi^{n+1}=\varphi^n+\Delta t\,f(\varphi^{n+1})}.
# $$
#
# The new solution appears inside $f$, so the update generally requires solving an algebraic system. This method is **implicit** and first-order accurate in time.
#
# #### Crank–Nicolson: average the endpoint rates
#
# Use the trapezoidal approximation:
#
# $$
# \overline f^{\,n+1/2}\approx
# \frac{f(\varphi^n)+f(\varphi^{n+1})}{2},\qquad
# \boxed{\varphi^{n+1}=\varphi^n+
# \frac{\Delta t}{2}\left[f(\varphi^n)+f(\varphi^{n+1})\right]}.
# $$
#
# This method averages the **two endpoint rates**, is generally implicit, and is second-order accurate in time.
#
# #### Implicit midpoint: evaluate at a midpoint state
#
# The midpoint quadrature rule suggests using $f(\varphi(t^{n+1/2}))$, where $t^{n+1/2}=(t^n+t^{n+1})/2$. In the **implicit midpoint method**, we approximate the midpoint state by the average of the endpoint states:
#
# $$
# \overline f^{\,n+1/2}\approx
# f\left(\frac{\varphi^n+\varphi^{n+1}}{2}\right),\qquad
# \boxed{\varphi^{n+1}=\varphi^n+
# \Delta t\,f\left(\frac{\varphi^n+\varphi^{n+1}}{2}\right)}.
# $$
#
# For $f(\varphi,t)$, this evaluation also uses the midpoint time $t^{n+1/2}$. The method is generally implicit and second-order accurate in time. It evaluates the **rate at the average state**, whereas Crank–Nicolson averages the **rates at the endpoint states**. These are generally different for nonlinear $f$, but coincide for an autonomous linear or affine right-hand side.
#
# Here **autonomous** means that $f$ has no explicit dependence on time: the rate depends on the current state, although that state changes with time. A scalar **linear** right-hand side has the form $f(\varphi)=a\varphi$, while an **affine** right-hand side has the form $f(\varphi)=a\varphi+b$, with time-independent coefficients $a$ and $b$. For example, $\dot\varphi=-k\varphi$ describes linear decay, and $\dot\varphi=-k\varphi+s$ adds a constant source. For a vector of nodal values, the corresponding affine form is $\underline f(\underline\varphi)=\mathbf A\underline\varphi+\underline b$, with a fixed matrix $\mathbf A$ and vector $\underline b$; the linear case has $\underline b=\underline 0$.
#
# For an affine right-hand side, averaging states before evaluation gives exactly the same result as averaging the evaluated rates:
#
# $$
# f\left(\frac{\varphi^n+\varphi^{n+1}}{2}\right)
# =a\frac{\varphi^n+\varphi^{n+1}}{2}+b
# =\frac{f(\varphi^n)+f(\varphi^{n+1})}{2}.
# $$
#
# This explains why implicit midpoint and Crank–Nicolson coincide in this case.
#
# There is also an **explicit midpoint method**: it first predicts a midpoint state using a half forward-Euler step, then evaluates $f$ at that predicted state. The midpoint method shown here uses the implicit endpoint-average formulation.

# %% [markdown]
# ### Accuracy of the time-integration methods
#
# We can establish the temporal orders using Taylor expansion, just as we did for spatial differences. To simplify the notation, let
#
# $$
# q(t)=f(\varphi(t)),\qquad q_n=q(t^n).
# $$
#
# Here $q$ denotes the right-hand side evaluated **along the exact solution trajectory**. Its time derivatives therefore include the changing solution; for an autonomous scalar problem, for example, $\dot q=f'(\varphi)\dot\varphi$.
#
# Introduce the elapsed time $\tau=t-t^n$, so that $\tau$ runs from $0$ to $\Delta t$ during the step. Taylor expansion about the beginning of the step gives
#
# $$
# q(t^n+\tau)=q_n+\tau\dot q_n
# +\frac{\tau^2}{2}\ddot q_n+\mathcal O(\tau^3).
# $$
#
# The quantities $q_n$, $\dot q_n$, and $\ddot q_n$ are evaluated at the fixed time $t^n$, so they are constants with respect to the integration variable $\tau$. Substituting this expansion into the definition of the interval mean gives
#
# $$
# \begin{aligned}
# \overline f^{\,n+1/2}
# &=\frac{1}{\Delta t}\int_0^{\Delta t}q(t^n+\tau)\,\mathrm d\tau\\
# &=\frac{1}{\Delta t}\left[
# q_n\tau+\frac{\tau^2}{2}\dot q_n
# +\frac{\tau^3}{6}\ddot q_n
# \right]_0^{\Delta t}
# +\mathcal O(\Delta t^3).
# \end{aligned}
# $$
#
# Integrating each power raises its exponent by one and divides by that new exponent. For example, the quadratic term contributes $\ddot q_n\Delta t^3/6$ to the integral; dividing by $\Delta t$ gives $\ddot q_n\Delta t^2/6$ to the mean. With a bounded third time derivative, integrating the Taylor remainder gives $\mathcal O(\Delta t^4)$, which becomes $\mathcal O(\Delta t^3)$ after division by $\Delta t$. Thus the exact interval mean has the expansion
#
# $$
# \overline f^{\,n+1/2}
# =q_n+\frac{\Delta t}{2}\dot q_n
# +\frac{\Delta t^2}{6}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# We compare each approximation with this mean, inserting exact solution states to isolate the error introduced by the time-integration formula.
#
# #### Forward Euler
#
# Forward Euler uses $q_n$, so its mean-rate error is
#
# $$
# q_n-\overline f^{\,n+1/2}
# =-\frac{\Delta t}{2}\dot q_n+\mathcal O(\Delta t^2).
# $$
#
# The leading error is proportional to $\Delta t$: forward Euler is said to be **first-order accurate in time**.
#
# #### Backward Euler
#
# Taylor expansion of the endpoint rate gives
#
# $$
# q(t^{n+1})=q_n+\Delta t\,\dot q_n
# +\frac{\Delta t^2}{2}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# Subtracting the exact mean yields
#
# $$
# q(t^{n+1})-\overline f^{\,n+1/2}
# =\frac{\Delta t}{2}\dot q_n+\mathcal O(\Delta t^2).
# $$
#
# Backward Euler is also **first-order accurate in time**. Its leading mean-rate error has the opposite sign to that of forward Euler.
#
# #### Crank–Nicolson
#
# Averaging the exact endpoint rates gives
#
# $$
# \frac{q_n+q(t^{n+1})}{2}
# =q_n+\frac{\Delta t}{2}\dot q_n
# +\frac{\Delta t^2}{4}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# Therefore,
#
# $$
# \frac{q_n+q(t^{n+1})}{2}-\overline f^{\,n+1/2}
# =\frac{\Delta t^2}{12}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# Averaging cancels the first-order mean-rate error. The remaining leading error is proportional to $\Delta t^2$, so Crank–Nicolson is **second-order accurate in time**.
#
# #### Midpoint rule and implicit midpoint
#
# Expanding the exact midpoint rate gives
#
# $$
# q\left(t^n+\frac{\Delta t}{2}\right)
# =q_n+\frac{\Delta t}{2}\dot q_n
# +\frac{\Delta t^2}{8}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# Hence the midpoint quadrature error in the mean is
#
# $$
# q\left(t^n+\frac{\Delta t}{2}\right)-\overline f^{\,n+1/2}
# =-\frac{\Delta t^2}{24}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# This is second order, but the implicit midpoint method also approximates the midpoint **state** by the average of the endpoint states. Taylor expansion shows that
#
# $$
# \frac{\varphi(t^n)+\varphi(t^{n+1})}{2}
# -\varphi\left(t^n+\frac{\Delta t}{2}\right)
# =\frac{\Delta t^2}{8}\ddot\varphi(t^n)+\mathcal O(\Delta t^3).
# $$
#
# For smooth $f$, an $\mathcal O(\Delta t^2)$ change in its state argument changes its value by $\mathcal O(\Delta t^2)$. Thus this substitution preserves the **second-order mean-rate accuracy** of implicit midpoint, although it generally changes the leading error coefficient. The coefficient $-\ddot q_n/24$ above belongs to quadrature using the exact midpoint state, rather than to the complete implicit midpoint formula.
#
# #### Mean-rate error, one-step error, and global error
#
# The time update multiplies the approximate mean by $\Delta t$. A mean-rate error of order $\mathcal O(\Delta t^p)$ therefore introduces a one-step solution error of order $\mathcal O(\Delta t^{p+1})$, when starting from the exact state. Over a fixed time interval, the number of steps grows proportionally to $1/\Delta t$. For a stable integration of a sufficiently smooth problem, propagation and accumulation of these errors give a global solution error of order $\mathcal O(\Delta t^p)$.
#
# | Method | Mean-rate error | One-step solution error, starting exactly | Global solution error over fixed time |
# |:--|:--|:--|:--|
# | Forward/backward Euler | $\mathcal O(\Delta t)$ | $\mathcal O(\Delta t^2)$ | $\mathcal O(\Delta t)$ |
# | Crank–Nicolson/implicit midpoint | $\mathcal O(\Delta t^2)$ | $\mathcal O(\Delta t^3)$ | $\mathcal O(\Delta t^2)$ |
#
# These orders assume sufficient smoothness, stability under time-step refinement, and adequately accurate solutions of the implicit algebraic equations. They describe temporal errors for the spatially discretized problem; spatial discretization introduces a separate error. A vanishing leading coefficient can give higher accuracy for a special solution.

# %% [markdown]
# ## Linear multistep methods
#
# ### Reusing previous time levels
#
# The methods considered so far advance from the current solution to the next time level. Another strategy is to reuse information from several previous time levels. **Linear multistep methods** combine solution values and/or right-hand-side evaluations at these levels to compute the new solution.
#
# For example, a method using the two known levels $n-1$ and $n$ to compute level $n+1$ is called a **two-step method**, although three time levels appear in the formula. The term “linear” describes how the solution values and right-hand-side evaluations are combined; the function $f$ itself may be nonlinear.
#
# We consider a constant time step $\Delta t$ and write $f^n=f(\varphi^n)$.
#
# The two panels illustrate the different constructions. AB2 extrapolates known rates and integrates that extrapolation over the new step. BDF2 differentiates a quadratic through the solution states, requiring its endpoint slope to agree with the evolution equation. The BDF2 panel uses the dimensionless example $\dot\varphi=\varphi$ with $\Delta t=1$; its open marker represents the state to be determined by the implicit solve, rather than an exact future sample.
#
# ![AB2 rate extrapolation with a shaded future-step integral, and BDF2 quadratic interpolation with its endpoint tangent and an unknown new state.](media/fd-multistep-light.svg){.light-content width=100%}
# ![AB2 rate extrapolation with a shaded future-step integral, and BDF2 quadratic interpolation with its endpoint tangent and an unknown new state.](media/fd-multistep-dark.svg){.dark-content width=100%}
#
# ### Explicit second-order Adams–Bashforth method (AB2)
#
# The mean-rate interpretation also motivates multistep integration. Given the two known rates $f^{n-1}$ and $f^n$, extrapolate their linear interpolant over the next interval:
#
# $$
# f(\varphi(t^n+\tau))\approx
# f^n+\frac{\tau}{\Delta t}(f^n-f^{n-1}),
# \qquad 0\leq\tau\leq\Delta t.
# $$
#
# Integrating this expression and dividing by the step duration gives
#
# $$
# \overline f^{\,n+1/2}\approx
# \frac{1}{\Delta t}\int_0^{\Delta t}
# \left[f^n+\frac{\tau}{\Delta t}(f^n-f^{n-1})\right]\,\mathrm d\tau
# =\frac32 f^n-\frac12 f^{n-1}.
# $$
#
# Thus the **second-order Adams–Bashforth method** is
#
# $$
# \boxed{
# \varphi^{n+1}=\varphi^n+
# \Delta t\left[\frac32 f(\varphi^n)-\frac12 f(\varphi^{n-1})\right].
# }
# $$
#
# Only known states enter the right-hand side, so AB2 is **explicit**. It uses past rates to predict the mean rate over the next interval. The weights sum to one, but one is negative: this is extrapolation rather than an average with positive weights.
#
# For a smooth exact rate history $q(t)=f(\varphi(t))$, Taylor expansion about $t^n$ gives
#
# $$
# \frac32 q(t^n)-\frac12 q(t^{n-1})
# =q_n+\frac{\Delta t}{2}\dot q_n
# -\frac{\Delta t^2}{4}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# Comparing with the exact interval mean yields
#
# $$
# \frac32 q(t^n)-\frac12 q(t^{n-1})-\overline f^{\,n+1/2}
# =-\frac{5\Delta t^2}{12}\ddot q_n+\mathcal O(\Delta t^3).
# $$
#
# The mean-rate error is second order, giving a one-step solution error of $\mathcal O(\Delta t^3)$ and, under suitable stability and starting-value assumptions, a global error of $\mathcal O(\Delta t^2)$. After initialization, the previous rate can be stored and reused, so only one new evaluation of $f$ is needed per step. Explicit stability restrictions still limit the allowable time step.
#
# ### Implicit second-order backward differentiation formula (BDF2)
#
# Instead of approximating the interval mean, we can approximate the time derivative at the new time level using the three solution values at $t^{n-1}$, $t^n$, and $t^{n+1}$. Differentiating the quadratic polynomial through these values at $t^{n+1}$ gives
#
# $$
# \dot\varphi(t^{n+1})\approx
# \frac{3\varphi^{n+1}-4\varphi^n+\varphi^{n-1}}{2\Delta t}.
# $$
#
# Substituting the evolution equation $\dot\varphi=f(\varphi)$ at the new level gives the **second-order backward differentiation formula**:
#
# $$
# \boxed{
# \frac{3\varphi^{n+1}-4\varphi^n+\varphi^{n-1}}{2\Delta t}
# =f(\varphi^{n+1}).
# }
# $$
#
# Equivalently,
#
# $$
# \varphi^{n+1}=\frac43\varphi^n-\frac13\varphi^{n-1}
# +\frac{2\Delta t}{3}f(\varphi^{n+1}).
# $$
#
# Since the unknown new state appears inside $f$, BDF2 is **implicit** and generally requires solving an algebraic system. For nonlinear $f$, this system is generally nonlinear.
#
# To check the accuracy, expand the exact previous states about $t^{n+1}$. The resulting derivative approximation satisfies
#
# $$
# \frac{3\varphi(t^{n+1})-4\varphi(t^n)+\varphi(t^{n-1})}{2\Delta t}
# -\dot\varphi(t^{n+1})
# =-\frac{\Delta t^2}{3}\varphi^{(3)}(t^{n+1})
# +\mathcal O(\Delta t^3).
# $$
#
# Here $\varphi^{(3)}$ denotes the third **time** derivative. The residual is second order in $\Delta t$; with sufficiently accurate starting values, stable integration, and adequately converged algebraic solves, BDF2 has second-order global temporal accuracy. BDF2 is often useful for stiff evolution problems, where fast time scales strongly restrict explicit integration. Implicitness does not remove time-step requirements for accuracy.
#
# ### Starting values and higher-order schemes
#
# Both methods require $\varphi^0$ and $\varphi^1$ before their first two-step update. The initial condition supplies $\varphi^0$; a suitable one-step method supplies $\varphi^1$ with sufficient accuracy. Stored solution or rate histories must be available when restarting the calculation.
#
# Both families have **higher-order members**. For example, the third-order Adams–Bashforth method uses three previous rates:
#
# $$
# \varphi^{n+1}=\varphi^n+\Delta t\left[
# \frac{23}{12}f^n-\frac{16}{12}f^{n-1}+\frac{5}{12}f^{n-2}
# \right].
# $$
#
# The third-order backward differentiation formula uses three previous solution states:
#
# $$
# \frac{11\varphi^{n+1}-18\varphi^n+9\varphi^{n-1}-2\varphi^{n-2}}
# {6\Delta t}=f(\varphi^{n+1}).
# $$
#
# Higher-order methods require more history and suitable initialization, and their stability properties differ from those of the second-order schemes. Thus a higher formal order is not automatically preferable. All coefficients shown here assume constant time steps; variable time steps require coefficients adapted to the actual time-level spacings.

# %% [markdown]
# ## Combining space and time discretization: periodic convection–diffusion
#
# ### Mathematical problem
#
# Consider a scalar field $\varphi(x,t)$ on the unit-length domain $\Omega=[0,1]$, governed by the one-dimensional convection–diffusion equation
#
# $$
# \frac{\partial\varphi}{\partial t}
# +v\frac{\partial\varphi}{\partial x}
# =\Gamma\frac{\partial^2\varphi}{\partial x^2},
# \qquad v>0,\quad\Gamma>0.
# $$
#
# The transport velocity $v$ and diffusion coefficient $\Gamma$ are constant. We prescribe the initial sine profile
#
# $$
# \varphi(x,0)=\sin(2\pi x)
# $$
#
# and periodic boundary conditions. The field is continued periodically beyond the interval, so, for a smooth solution,
#
# $$
# \varphi(0,t)=\varphi(1,t),\qquad
# \partial_x\varphi(0,t)=\partial_x\varphi(1,t).
# $$
#
# The two ends are identified: a profile leaving at the right re-enters at the left. The unit-length coordinate is dimensionless; $v$ and $\Gamma$ are expressed consistently with the chosen time scale.
#
# ![Initial sine profile on the unit interval, positive transport velocity, and periodic identification of the two endpoints.](media/fd-convection-diffusion-light.svg){.light-content width=100%}
# ![Initial sine profile on the unit interval, positive transport velocity, and periodic identification of the two endpoints.](media/fd-convection-diffusion-dark.svg){.dark-content width=100%}
#
# This problem has the exact solution
#
# $$
# \boxed{\varphi(x,t)=e^{-4\pi^2\Gamma t}\sin\bigl(2\pi(x-vt)\bigr)}.
# $$
#
# Convection transports the wave to the right at speed $v$, while diffusion decreases its amplitude exponentially. The exact solution provides a reference for assessing spatial and temporal errors separately.
#
# ### Uniform periodic mesh
#
# Use $N\geq3$ distinct periodic nodes,
#
# $$
# \Delta x=\frac{1}{N},\qquad
# x_i=i\Delta x,\quad i=0,\dots,N-1,\qquad
# \Omega_h=\{x_i\}_{i=0}^{N-1}.
# $$
#
# Unlike a mesh with two separate boundary nodes, a periodic mesh identifies $x=1$ with $x=0$. We therefore store $N$ distinct values and have $N$ intervals around the periodic domain. The endpoint at $x=1$ can be repeated for plotting, but is not an additional unknown. Indices wrap around:
#
# $$
# \varphi_{-1}=\varphi_{N-1},\qquad
# \varphi_N=\varphi_0,\qquad
# \varphi_i(0)=\sin(2\pi x_i).
# $$
#
# ### Spatial discretization
#
# For diffusion, use the compact central second derivative:
#
# $$
# \Gamma\,\partial_{xx}\varphi(x_i,t)
# \approx\frac{\Gamma}{\Delta x^2}
# (\varphi_{i+1}-2\varphi_i+\varphi_{i-1}).
# $$
#
# For convection, recall the hyperbolic transport prototype from Lecture 2: its characteristics satisfy $\mathrm dx/\mathrm dt=v$. With $v>0$, information travels from left to right. The full equation with $\Gamma>0$ is parabolic, but its convective part retains this directional transport. We compare two approximations of that part.
#
# #### Central convection
#
# Using the central first derivative gives
#
# $$
# v\,\partial_x\varphi(x_i,t)
# \approx\frac{v}{2\Delta x}(\varphi_{i+1}-\varphi_{i-1}).
# $$
#
# This second-order approximation uses both neighboring nodes symmetrically. It does not select a preferred upstream direction; its suitability depends on diffusion, resolution, and the time-integration method.
#
# #### Upwind convection
#
# For positive $v$, the upstream neighbor is $i-1$. The backward difference therefore gives the **upwind approximation**
#
# $$
# v\,\partial_x\varphi(x_i,t)
# \approx\frac{v}{\Delta x}(\varphi_i-\varphi_{i-1}).
# $$
#
# “Upwind” means choosing the one-sided stencil according to the direction of information transport. For negative velocity, the upstream neighbor would instead be $i+1$, requiring a forward difference. The approximation used here is first-order accurate:
#
# $$
# -vD^-\varphi_i^*
# =-v\varphi'(x_i)+\frac{v\Delta x}{2}\varphi''(x_i)
# +\mathcal O(\Delta x^2).
# $$
#
# The Taylor analysis shows that the leading error contribution to the evolution equation is a coefficient, $v\Delta x/2$, times the second derivative $\varphi''(x_i)$. This has exactly the same mathematical form as the physical diffusion term $\Gamma\varphi''(x_i)$. To leading order, the spatially discretized equation therefore behaves like
#
# $$
# \partial_t\varphi+v\partial_x\varphi
# =\left(\Gamma+\frac{v\Delta x}{2}\right)\partial_{xx}\varphi
# +\mathcal O(\Delta x^2).
# $$
#
# The additional positive diffusion coefficient $v\Delta x/2$ comes from the approximation, rather than from the physical model. It is therefore called **numerical diffusion**. Like physical diffusion, it smooths spatial variations and damps the amplitude of the sine wave; here it causes extra damping beyond that prescribed by $\Gamma$. This leading spatial error decreases as the mesh is refined. The interpretation concerns the semi-discrete equation; time integration can introduce additional errors.
#
# ### Semi-discrete equations
#
# We discretize space while keeping time continuous. Each nodal value $\varphi_i(t)$ satisfies an ordinary differential equation. For central convection,
#
# $$
# \frac{\mathrm d\varphi_i}{\mathrm dt}
# =\frac{\Gamma}{\Delta x^2}(\varphi_{i+1}-2\varphi_i+\varphi_{i-1})
# -\frac{v}{2\Delta x}(\varphi_{i+1}-\varphi_{i-1}).
# $$
#
# For convenience, we introduce the constant coefficients
#
# $$
# A=\frac{\Gamma}{\Delta x^2},\qquad
# B=\frac{v}{2\Delta x}.
# $$
#
# Collecting terms gives
#
# $$
# \boxed{\frac{\mathrm d\varphi_i}{\mathrm dt}
# =(A+B)\varphi_{i-1}-2A\varphi_i+(A-B)\varphi_{i+1}}
# \qquad\text{(central convection)}.
# $$
#
# For upwind convection, the corresponding equation is
#
# $$
# \begin{aligned}
# \frac{\mathrm d\varphi_i}{\mathrm dt}
# &=A(\varphi_{i+1}-2\varphi_i+\varphi_{i-1})
# -2B(\varphi_i-\varphi_{i-1})\\
# &=\boxed{(A+2B)\varphi_{i-1}-(2A+2B)\varphi_i+A\varphi_{i+1}}
# \qquad\text{(upwind convection)}.
# \end{aligned}
# $$
#
# These equations apply at every node using periodic index wrapping. Their right-hand sides have the common form
#
# $$
# \frac{\mathrm d\varphi_i}{\mathrm dt}
# =c_-\varphi_{i-1}+c_0\varphi_i+c_+\varphi_{i+1},
# $$
#
# with coefficients
#
# | Convection approximation | $c_-$ | $c_0$ | $c_+$ |
# |:--|:--|:--|:--|
# | Central | $A+B$ | $-2A$ | $A-B$ |
# | Upwind, $v>0$ | $A+2B$ | $-2A-2B$ | $A$ |
#
# ### Vector–matrix form
#
# Next, we collect the time-dependent nodal values in the solution vector
#
# $$
# \underline{\varphi}(t)=
# \begin{bmatrix}
# \varphi_0(t)\\\varphi_1(t)\\\vdots\\\varphi_{N-1}(t)
# \end{bmatrix}
# \in\mathbb R^N.
# $$
#
# The entire semi-discrete problem becomes
#
# $$
# \boxed{\frac{\mathrm d\underline{\varphi}}{\mathrm dt}
# =\mathbf L\underline{\varphi},\qquad
# \underline{\varphi}(0)=\begin{bmatrix}\sin(2\pi x_0)\\\vdots\\\sin(2\pi x_{N-1})\end{bmatrix}.}
# $$
#
# The matrix $\mathbf L$ is defined by
#
# $$
# (\mathbf L\underline{\varphi})_i
# =c_-\varphi_{(i-1)\bmod N}
# +c_0\varphi_i
# +c_+\varphi_{(i+1)\bmod N}.
# $$
#
# Here $\bmod N$ means taking the remainder modulo $N$, represented by an index in $\{0,\dots,N-1\}$. This wraps indices around the periodic mesh. For example, with $N=5$, the left neighbor of node $i=0$ has index $(-1)\bmod5=4$, and the right neighbor of node $i=4$ has index $5\bmod5=0$. Thus the first and last stored nodes are neighbors.
#
# For example, with $N=5$,
#
# $$
# \mathbf L=
# \begin{bmatrix}
# c_0&c_+&0&0&c_-\\
# c_-&c_0&c_+&0&0\\
# 0&c_-&c_0&c_+&0\\
# 0&0&c_-&c_0&c_+\\
# c_+&0&0&c_-&c_0
# \end{bmatrix}.
# $$
#
# The corner entries implement the periodic connection between the first and last stored nodes. Both coefficient choices satisfy $c_-+c_0+c_+=0$, so constant fields have zero time derivative. On this uniform periodic grid, the column sums also vanish, preserving the discrete spatial mean in the semi-discrete equations.
#
# We have now reached the autonomous linear evolution form introduced in the time-integration section, with $\underline f(\underline\varphi)=\mathbf L\underline\varphi$. Any of the time-integration schemes discussed above can be applied to this system, subject to its stability and accuracy requirements.

# %% [markdown]
# ### From the semi-discrete system to a fully discrete scheme
#
# We now apply time integration to the semi-discrete system
#
# $$
# \frac{\mathrm d\underline\varphi}{\mathrm dt}
# =\mathbf L\underline\varphi.
# $$
#
# The vector $\underline\varphi^n$ approximates the nodal solution at time $t^n$. The constant matrix $\mathbf L$ contains either the central or the upwind convection coefficients, together with central diffusion and periodic index wrapping. Discretizing time as well as space gives a **fully discrete scheme**.
#
# #### Forward Euler
#
# Evaluating the right-hand side at the known time level gives
#
# $$
# \frac{\underline\varphi^{n+1}-\underline\varphi^n}{\Delta t}
# =\mathbf L\underline\varphi^n,
# \qquad
# \boxed{\underline\varphi^{n+1}
# =(\mathbf I+\Delta t\,\mathbf L)\underline\varphi^n}.
# $$
#
# Here $\mathbf I$ is the $N\times N$ identity matrix. This update is explicit: the new solution is computed directly from the old solution.
#
# #### Backward Euler
#
# Evaluating the right-hand side at the new time level gives
#
# $$
# \frac{\underline\varphi^{n+1}-\underline\varphi^n}{\Delta t}
# =\mathbf L\underline\varphi^{n+1},
# \qquad
# \boxed{(\mathbf I-\Delta t\,\mathbf L)\underline\varphi^{n+1}
# =\underline\varphi^n}.
# $$
#
# The new solution is obtained by solving a linear system at each step.
#
# #### Crank–Nicolson
#
# Averaging the right-hand sides at the two time levels gives
#
# $$
# \frac{\underline\varphi^{n+1}-\underline\varphi^n}{\Delta t}
# =\frac12\mathbf L\underline\varphi^n
# +\frac12\mathbf L\underline\varphi^{n+1},
# $$
#
# or, after collecting the unknown new values on the left,
#
# $$
# \boxed{\left(\mathbf I-\frac{\Delta t}{2}\mathbf L\right)
# \underline\varphi^{n+1}
# =\left(\mathbf I+\frac{\Delta t}{2}\mathbf L\right)
# \underline\varphi^n}.
# $$
#
# This method also requires a linear-system solve, but has second-order temporal accuracy rather than the first-order accuracy of the Euler methods.
#
# #### A single weighted formula
#
# All three updates can be expressed by weighting the old and new right-hand sides with a parameter $\alpha$:
#
# $$
# \frac{\underline\varphi^{n+1}-\underline\varphi^n}{\Delta t}
# =(1-\alpha)\mathbf L\underline\varphi^n
# +\alpha\mathbf L\underline\varphi^{n+1}.
# $$
#
# Rearranging gives the common matrix equation
#
# $$
# \boxed{
# (\mathbf I-\alpha\Delta t\,\mathbf L)\underline\varphi^{n+1}
# =\left[\mathbf I+(1-\alpha)\Delta t\,\mathbf L\right]
# \underline\varphi^n.
# }
# $$
#
# The parameter $\alpha$ is the weight assigned to the **new-time-level rate**; it is unrelated to the weights used earlier to combine spatial difference operators.
#
# | $\alpha$ | Time-integration scheme | Update type | Temporal order |
# |:--|:--|:--|:--|
# | $0$ | Forward Euler | Explicit | First |
# | $1/2$ | Crank–Nicolson | Implicit | Second |
# | $1$ | Backward Euler | Implicit | First |
#
# This family is often called the **theta method**, with $\alpha$ taking the role commonly denoted by $\theta$. The same equation works for either spatial discretization through the choice of $\mathbf L$.
#
# #### Absorbing the time step into the spatial matrix
#
# For convenience, we introduce the dimensionless matrix
#
# $$
# \mathbf M:=\Delta t\,\mathbf L.
# $$
#
# The weighted time step then becomes
#
# $$
# \boxed{(\mathbf I-\alpha\mathbf M)\underline\varphi^{n+1}
# =[\mathbf I+(1-\alpha)\mathbf M]\underline\varphi^n}.
# $$
#
# Multiplication by $\Delta t$ turns the spatial coefficients into dimensionless combinations. We define the **diffusion number** $D$ (also called the Fourier number) and the **Courant number** $C$ by
#
# $$
# D:=\frac{\Gamma\Delta t}{\Delta x^2}=A\Delta t,
# \qquad C:=\frac{|v|\Delta t}{\Delta x}.
# $$
#
# The diffusion number compares the step duration with the diffusion time scale $\Delta x^2/\Gamma$. The Courant number is the distance traveled by convection during one step, measured in grid spacings. Both are nonnegative. We use the name diffusion number here to distinguish $D$ from the convective Courant number $C$.
#
# Although our example assumes $v>0$, we can write the coefficients for either transport direction by introducing
#
# $$
# s:=\operatorname{sgn}(v),\qquad v=s|v|,
# \qquad B\Delta t=\frac{sC}{2}.
# $$
#
# For positive velocity $s=1$, for negative velocity $s=-1$, and for zero velocity $s=0$ and $C=0$.
#
# For **central convection**, the coefficients of $\mathbf M_{\mathrm{cen}}$ are
#
# $$
# m_-=D+\frac{sC}{2},\qquad
# m_0=-2D,\qquad
# m_+=D-\frac{sC}{2}.
# $$
#
# For **upwind convection**, the stencil must switch with the transport direction. Positive velocity uses the left neighbor; negative velocity uses the right neighbor. The coefficients of $\mathbf M_{\mathrm{up}}$ are therefore
#
# $$
# m_-=D+\frac{1+s}{2}C,\qquad
# m_0=-2D-C,\qquad
# m_+=D+\frac{1-s}{2}C.
# $$
#
# In particular, for the positive velocity of our example, these reduce to $(D+C,-2D-C,D)$. For negative velocity they become $(D,-2D-C,D+C)$.
#
# For either choice, the matrix action is
#
# $$
# (\mathbf M\underline\varphi)_i
# =m_-\varphi_{(i-1)\bmod N}+m_0\varphi_i
# +m_+\varphi_{(i+1)\bmod N}.
# $$
#
# For example, with $N=5$, the two matrices are
#
# $$
# \mathbf M_{\mathrm{cen}}=
# \begin{bmatrix}
# -2D&D-sC/2&0&0&D+sC/2\\
# D+sC/2&-2D&D-sC/2&0&0\\
# 0&D+sC/2&-2D&D-sC/2&0\\
# 0&0&D+sC/2&-2D&D-sC/2\\
# D-sC/2&0&0&D+sC/2&-2D
# \end{bmatrix},
# $$
#
# $$
# \mathbf M_{\mathrm{up}}=
# \begin{bmatrix}
# -2D-C&D+\frac{1-s}{2}C&0&0&D+\frac{1+s}{2}C\\
# D+\frac{1+s}{2}C&-2D-C&D+\frac{1-s}{2}C&0&0\\
# 0&D+\frac{1+s}{2}C&-2D-C&D+\frac{1-s}{2}C&0\\
# 0&0&D+\frac{1+s}{2}C&-2D-C&D+\frac{1-s}{2}C\\
# D+\frac{1-s}{2}C&0&0&D+\frac{1+s}{2}C&-2D-C
# \end{bmatrix}.
# $$
#
# The periodic corner entries remain present. The numbers $D$ and $C$ combine the physical coefficients, spatial resolution, and time step in a form useful for the stability analysis in the next lecture.
#
# At an individual node, using $m_-=\Delta t\,c_-$, $m_0=\Delta t\,c_0$, and $m_+=\Delta t\,c_+$, the weighted equation reads
#
# $$
# \begin{aligned}
# &-\alpha m_-\varphi_{i-1}^{n+1}
# +(1-\alpha m_0)\varphi_i^{n+1}
# -\alpha m_+\varphi_{i+1}^{n+1}\\
# &\qquad=(1-\alpha)m_-\varphi_{i-1}^{n}
# +\left[1+(1-\alpha)m_0\right]\varphi_i^{n}
# +(1-\alpha)m_+\varphi_{i+1}^{n}.
# \end{aligned}
# $$
#
# The indices remain periodic. Thus the same dimensionless coefficients $m_-$, $m_0$, and $m_+$ enter all three methods; $\alpha$ determines how their contributions are distributed between the known and unknown time levels. Their stability and resulting errors will be examined in the lecture on consistency, stability, and convergence.

# %% [markdown]
# ### Solving the time step
#
# For forward Euler ($\alpha=0$), the matrix multiplying the new solution is the identity, and the update is explicit. For backward Euler and Crank–Nicolson, the new nodal values are coupled through $\mathbf I-\alpha\mathbf M$, so each time step requires solving a **linear system of equations**. We will study direct and iterative solution methods later in the course. An iterative solve is one option, rather than a requirement for every system. With fixed coefficients, mesh, and time step, the system matrix stays the same, allowing a suitable factorization or preconditioner to be reused.
#
# ### Example: forward Euler with central and upwind convection
#
# The animation compares the analytical solution with forward Euler applied to the two spatial discretizations. Both numerical solutions use compact central diffusion, the same periodic mesh, and the same time step:
#
# $$
# N=40,\quad v=1,\quad\Gamma=0.01,\quad
# \Delta x=\frac1{40},\quad\Delta t=0.001,
# \qquad C=0.04,\quad D=0.016.
# $$
#
# The computation runs from $t=0$ to $t=3$, covering three passages around the periodic domain. The time step is stable for both fully discrete schemes in this example; stability conditions will be derived in the next lecture. Markers show computed nodal values, with the periodic endpoint repeated for plotting.
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Forward Euler solutions with central and upwind convection compared with the decaying traveling analytical sine wave" autoplay muted><source src="media/fd-euler-comparison-light.webm" type="video/webm"><source src="media/fd-euler-comparison-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Forward Euler solutions with central and upwind convection compared with the decaying traveling analytical sine wave" autoplay muted><source src="media/fd-euler-comparison-dark.webm" type="video/webm"><source src="media/fd-euler-comparison-dark.mp4" type="video/mp4"></video>
#
# Both solutions travel in the correct direction, but the upwind approximation damps the wave more strongly because of its numerical diffusion. Central convection more closely follows the analytical wave for these parameters, although it also has spatial and temporal errors. This comparison motivates the next questions: under what conditions is each combination stable, and how do its errors change when we refine the mesh and the time step?
