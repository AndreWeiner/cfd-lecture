# %% [markdown]
# ---
# title: Transport phenomena and classification of partial differential equations
# subtitle: Lecture 2
# jupyter: python3
# ---
#
# ## Learning objectives
#
# After this lecture, we will be able to:
#
# - Distinguish hyperbolic, parabolic, and elliptic behavior and relate each type to examples in fluid mechanics.
# - Classify a scalar second-order PDE using its principal part, characteristic directions, and discriminant, and explain when local linearization is needed.
# - Rewrite a second-order PDE as a first-order system and derive its characteristic condition from a determinant.
# - Construct the conserved-state and flux vectors of the one-dimensional Euler equations and derive the flux Jacobian.
# - Use eigenvalues and eigenvectors to identify characteristic variables, propagation speeds, and hyperbolicity.
# - Interpret domains of dependence and ranges of influence, distinguishing steady spatial models from time-dependent propagation.
# - Select appropriate initial and boundary data for prototype PDEs and explain their implications for Euler and Navier–Stokes flow problems.
#
# ## What does classification mean?
#
# ::: {.callout-note title="Classification"}
# **Classification means grouping problems according to shared properties. For partial differential equations (PDEs), we distinguish hyperbolic, parabolic, and elliptic types according to their mathematical structure and the characteristic behavior of their solutions.**
# :::
#
# ## Why is classification helpful?
#
# Classification provides three useful perspectives on a flow problem:
#
# 1. **Understanding.** A complex flow can involve several interacting phenomena. By identifying these phenomena and relating them to simpler, well-understood prototype problems, we can understand their individual roles before considering their interaction. The PDE types provide a framework for distinguishing these behaviors. We will return to this perspective later using flow past an airfoil as an example.
#
# 2. **Guidance for numerical methods.** Different PDE types have different numerical requirements. We can develop and study methods that work well for each type using prototype problems, then combine suitable methods to solve more complex flow problems. Their coupling must also be considered when assessing the complete numerical method.
#
# 3. **Well-posedness.** Classification helps us determine which initial and boundary conditions are appropriate and assess whether a problem is well posed. A problem is well posed if a solution exists, is unique, and depends continuously on the prescribed data: small changes in those data should produce small changes in the solution. The PDE type guides this assessment, but does not by itself guarantee well-posedness.

# %% [markdown]
# ## Three PDE types and their physical behavior
#
# We introduce each type through a prototype problem. The field $\varphi$ is a generic scalar quantity. The examples below use constant coefficients and Cartesian coordinates; their coordinate-free forms use the gradient $\nabla$ and the divergence $\nabla\cdot$.
#
# ### Hyperbolic behavior — transport and wave propagation
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Convection in x, time and phi coordinates, with characteristics"><source src="media/pde-convection-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Convection in x, time and phi coordinates, with characteristics"><source src="media/pde-convection-dark.mp4" type="video/mp4"></video>
#
# The animation shows an initial profile moving at a constant velocity without changing its shape. The three axes represent position $x$, time $t$, and the field $\varphi$. The highlighted curve is the current profile; the faint curves show its history. The dashed lines are **characteristics**: paths along which the value of $\varphi$ remains constant. All plotted quantities are dimensionless.
#
# The one-dimensional convection equation and its coordinate-free form are
#
# $$
# \frac{\partial\varphi}{\partial t}+u\frac{\partial\varphi}{\partial x}=0,
# \qquad
# \frac{\partial\varphi}{\partial t}+\mathbf{u}\cdot\nabla\varphi=0.
# $$
#
# Here $u$ is a constant transport velocity in one dimension, and $\mathbf{u}$ is its vector counterpart. Given an initial profile $\varphi_0(x)$, the one-dimensional solution is
#
# $$
# \varphi(x,t)=\varphi_0(x-ut),\qquad x(t)=x_0+ut.
# $$
#
# Information travels along the characteristics at a finite speed. In the animation, $u=1$, and the initial profile is a localized Gaussian pulse.
#
# Examples in fluid mechanics include:
#
# - **Convection of a passive scalar.** A concentration pattern is transported by a prescribed flow when diffusion is neglected:
#
#   $$\frac{\partial c}{\partial t}+\mathbf{u}\cdot\nabla c=0.$$
#
#   Here $c$ is the scalar concentration, $\mathbf{u}$ is the prescribed fluid velocity, and $t$ is time.
# - **Time-domain acoustics.** Small pressure disturbances in a homogeneous fluid at rest satisfy the linear acoustic wave equation:
#
#   $$\frac{\partial^2p'}{\partial t^2}-c_0^2\nabla^2p'=0.$$
#
#   Here $p'$ is the pressure perturbation about the undisturbed state, and $c_0$ is the constant sound speed. This inviscid model is hyperbolic, although it differs from the one-way convection prototype.
# - **Steady supersonic potential flow.** For small disturbances to a uniform two-dimensional compressible flow, the linearized potential equation is
#
#   $$(1-M_\infty^2)\frac{\partial^2\psi}{\partial x^2}+\frac{\partial^2\psi}{\partial y^2}=0.$$
#
#   Here $\psi$ is the perturbation velocity potential, so the velocity perturbation is $\nabla\psi$. The coordinate $x$ follows the uniform flow, $y$ is transverse, and $M_\infty=U_\infty/c_\infty$ is the free-stream Mach number, defined using the free-stream speed and sound speed. We consider supersonic flow, $M_\infty>1$, for which the equation is hyperbolic. This example assumes smooth, small disturbances and does not describe the internal structure of shocks.
#
# ### Parabolic behavior — diffusion and smoothing
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Diffusion in a fixed x and phi plane approaching a uniform state"><source src="media/pde-diffusion-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Diffusion in a fixed x and phi plane approaching a uniform state"><source src="media/pde-diffusion-dark.mp4" type="video/mp4"></video>
#
# The animation shows $\varphi$ against $x$ in a fixed plane, with the current time displayed in the corner. The dotted curve marks the initial profile. Diffusion smooths spatial differences. Here the initial profile gradually approaches a uniform distribution rather than moving unchanged. The animation uses a finite domain with zero flux at both ends and no sources. These conditions conserve the spatial mean, which determines the uniform equilibrium. All plotted quantities are dimensionless.
#
# For a constant diffusion coefficient $\alpha>0$, the one-dimensional diffusion equation and its coordinate-free form are
#
# $$
# \frac{\partial\varphi}{\partial t}=\alpha\frac{\partial^2\varphi}{\partial x^2},
# \qquad
# \frac{\partial\varphi}{\partial t}=\nabla\cdot(\alpha\nabla\varphi)=\alpha\nabla^2\varphi.
# $$
#
# For the dimensionless domain $0\leq x\leq1$, the animation uses the exact solution
#
# $$
# \varphi(x,t)=0.5+0.4\cos\bigl(2\pi(x-0.5)\bigr)\exp\bigl(-4\pi^2\alpha t\bigr),\qquad\alpha=0.08.
# $$
#
# Its spatial variations decay with time, while its mean remains $0.5$. A uniform final state depends on the boundary conditions and sources; diffusion does not always lead to a uniform distribution.
#
# Examples in fluid mechanics include:
#
# - **Unsteady heat conduction.** Without convection or heat sources, a material with constant properties satisfies
#
#   $$\frac{\partial T}{\partial t}=\alpha_T\nabla^2T,\qquad\alpha_T=\frac{k}{\rho c_p}.$$
#
#   Here $T$ is temperature, $k$ is thermal conductivity, $\rho$ is density, $c_p$ is specific heat capacity at constant pressure, and $\alpha_T$ is thermal diffusivity.
# - **Momentum or species diffusion.** In a unidirectional flow without convection or a pressure gradient, viscous diffusion gives
#
#   $$\frac{\partial u}{\partial t}=\nu\frac{\partial^2u}{\partial y^2},\qquad\nu=\frac{\mu}{\rho}.$$
#
#   Here $u$ is the streamwise velocity, $y$ is the transverse coordinate, $\mu$ is dynamic viscosity, and $\nu$ is kinematic viscosity. For a species without convection, reactions, or sources, the analogous equation is
#
#   $$\frac{\partial c}{\partial t}=D\nabla^2c.$$
#
#   Here $c$ is concentration and $D$ is the constant molecular diffusivity.
# - **A steady attached boundary layer.** For a two-dimensional incompressible laminar boundary layer with constant properties, the governing equations are
#
#   $$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0,\qquad
#   u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}
#   =-\frac{1}{\rho}\frac{\mathrm{d}p_e}{\mathrm{d}x}+\nu\frac{\partial^2u}{\partial y^2}.$$
#
#   Here $x$ is the downstream coordinate, $y$ is normal to the wall, $u$ and $v$ are the corresponding velocity components, and $p_e(x)$ is the pressure imposed by the outer flow. Density $\rho$ and kinematic viscosity $\nu$ are constant. Downstream convection is coupled to transverse diffusion, giving parabolic character with $x$ as the marching direction. This description assumes attached forward flow and breaks down near separation or reversed flow.
#
# ### Elliptic behavior — equilibrium and boundary influence
#
# ![Steady Poiseuille velocity profile between stationary walls.](media/pde-equilibrium-light.png){.light-content width=100%}
# ![Steady Poiseuille velocity profile between stationary walls.](media/pde-equilibrium-dark.png){.dark-content width=100%}
#
# An elliptic problem describes a spatial balance rather than an initial profile evolving in time. Its solution is determined by conditions on the boundary together with any sources within the domain. A useful prototype is the Poisson equation; without sources, it becomes the Laplace equation:
#
# $$
# -\frac{\mathrm{d}}{\mathrm{d}x}\left(\alpha\frac{\mathrm{d}\varphi}{\mathrm{d}x}\right)=s,
# \qquad
# -\nabla\cdot(\alpha\nabla\varphi)=s,\qquad\alpha>0.
# $$
#
# Here $s$ is a prescribed source. The first expression is the one-dimensional reduction, an ordinary differential equation; the second is the corresponding elliptic PDE in multiple spatial dimensions.
#
# The sketch shows fully developed, steady **plane Poiseuille flow** between stationary walls. A constant pressure gradient balances viscous stresses:
#
# $$
# \mu\frac{\mathrm{d}^2u}{\mathrm{d}y^2}=\frac{\mathrm{d}p}{\mathrm{d}x},
# \qquad u(0)=u(H)=0,
# \qquad
# u(y)=-\frac{1}{2\mu}\frac{\mathrm{d}p}{\mathrm{d}x}\,y(H-y).
# $$
#
# For $\mathrm{d}p/\mathrm{d}x<0$, the velocity is positive inside the channel. Here $H$ is the channel height, $\mu$ is the dynamic viscosity, and $u_{\max}$ is the maximum velocity. Note that the velocity profile is a parabola, but this does **not** make the governing steady boundary-value problem a parabolic PDE. The words describe different things: the profile's shape and the equation's type.
#
# Further examples include:
#
# - **Steady heat conduction.** For constant thermal conductivity, the equilibrium temperature satisfies
#
#   $$-\nabla\cdot(k\nabla T)=\dot q.$$
#
#   Here $T$ is temperature, $k>0$ is thermal conductivity, and $\dot q$ is the volumetric heat-generation rate. Boundary conditions prescribe temperatures or heat fluxes.
# - **Incompressible, irrotational potential flow.** Writing the velocity as the gradient of a potential gives
#
#   $$\mathbf{u}=\nabla\psi,\qquad\nabla\cdot\mathbf{u}=0\quad\Rightarrow\quad\nabla^2\psi=0.$$
#
#   Here $\mathbf{u}$ is fluid velocity and $\psi$ is the velocity potential. Incompressibility gives the zero-divergence condition, while the potential representation describes an irrotational velocity field.
# - **Pressure determination in incompressible flow.** For constant density and viscosity, with no body forces, taking the divergence of the momentum equation gives
#
#   $$\nabla^2p=-\rho\,\nabla\cdot\bigl[\nabla\cdot(\mathbf{u}\otimes\mathbf{u})\bigr].$$
#
#   Here $p$ is pressure, $\rho$ is density, $\mathbf{u}$ is the divergence-free velocity field, and $\otimes$ denotes the outer product. The inner divergence acts on the velocity-flux tensor; the outer divergence acts on the resulting vector. Given the velocity field and appropriate boundary conditions, this is an elliptic equation for pressure.

# %% [markdown]
# ### One application, several behaviors — transonic flow past an airfoil
#
# Complex real-world applications, such as flow past an airfoil, can combine all the behaviors introduced above. We consider a transonic case with a subsonic free stream, a locally supersonic region above the airfoil, and a shock that may interact with the boundary layer.
#
# ![Schematic transonic airfoil showing subsonic outer flow, a supersonic pocket and shock, attached boundary layers, an unsteady wake, and acoustic radiation.](media/transonic-airfoil-light.png){.light-content width=100%}
# ![Schematic transonic airfoil showing subsonic outer flow, a supersonic pocket and shock, attached boundary layers, an unsteady wake, and acoustic radiation.](media/transonic-airfoil-dark.png){.dark-content width=100%}
#
# The sketch is qualitative: the region boundaries, boundary-layer thickness, and acoustic wavefronts are schematic rather than simulation results. The arrow $U_\infty$ indicates the free-stream velocity.
#
# - **Elliptic behavior in the subsonic outer flow.** Away from viscous layers, a steady, approximately irrotational flow can be described using a compressible potential-flow model. In subsonic regions, its equation is elliptic. This approximation does not describe the entire transonic flow: it excludes viscous layers and may fail where shocks generate appreciable entropy and vorticity.
# - **Hyperbolic behavior in the supersonic pocket.** The flow can accelerate locally above the sound speed even though the free stream is subsonic. In a smooth supersonic region, the steady potential-flow equation is hyperbolic. Thus the outer-flow model itself has mixed elliptic–hyperbolic character. The shock is a discontinuity connecting different flow states and requires appropriate conservation and admissibility conditions.
# - **Parabolic behavior in the attached boundary layer.** Close to the surface, viscosity matters. Where the flow remains attached and a steady boundary-layer approximation is suitable, the equations can be marched downstream and have parabolic character. This description should not be extended unchanged into a separated or strongly unsteady region.
# - **Hyperbolic behavior in acoustic propagation.** Unsteady separation, wake motion, or shock motion can generate pressure disturbances that radiate as sound. The propagation of small acoustic disturbances is described by a hyperbolic wave equation, or a convected wave equation when the mean flow matters. In the subsonic outer flow, sound can propagate upstream as well as downstream: its upstream speed relative to the airfoil is approximately $c_\infty-U_\infty>0$. Here $c_\infty$ is the free-stream sound speed and $U_\infty$ is the free-stream speed. This identifies the acoustic behavior; it does not classify the whole separated shear layer as purely hyperbolic.
#
# These are interacting physical mechanisms and modeling perspectives, rather than independent pieces that can always be solved separately. They explain why classification is useful: we can understand prototype behaviors and choose suitable numerical methods, while still accounting for their coupling in the complete flow problem.

# %% [markdown]
# ## Techniques for classifying PDEs and systems of PDEs
#
# In the previous section, we encountered PDEs with different mathematical forms that exhibit the same characteristic behavior: transport and wave propagation, diffusion, or spatial equilibrium. Their physical variables and coefficients differ, but they share features that allow us to group them as hyperbolic, parabolic, or elliptic problems.
#
# Conversely, the same physical problem can sometimes be expressed either as a system of equations or as a single equation. For example, in incompressible, irrotational potential flow, the velocity field satisfies
#
# $$
# \nabla\cdot\mathbf{u}=0,\qquad\nabla\times\mathbf{u}=\mathbf{0}.
# $$
#
# Introducing a velocity potential $\psi$ with $\mathbf{u}=\nabla\psi$ satisfies the irrotationality condition and reduces the system to the single equation
#
# $$
# \nabla^2\psi=0.
# $$
#
# This representation is valid locally and, for a smooth irrotational field, globally on a simply connected domain. The change of variables alters the mathematical form while preserving the underlying flow description under these assumptions.
#
# We therefore need reliable classification techniques that go beyond recognizing a familiar equation. They should reveal the characteristic behavior encoded in a PDE or a system, even when it is written in an unfamiliar form. In the following, we will examine systematic criteria for individual PDEs and systems of PDEs, keeping track of the assumptions required when moving between equivalent formulations.

# %% [markdown]
# ### Classification of second-order PDEs
#
# We consider a scalar field $\varphi(x,y)$ with independent variables $x$ and $y$. A second-order equation with real coefficients that is linear in its highest-order derivatives can be written as
#
# $$
# A\frac{\partial^2\varphi}{\partial x^2}
# +2B\frac{\partial^2\varphi}{\partial x\partial y}
# +C\frac{\partial^2\varphi}{\partial y^2}
# =R\left(x,y,\varphi,\frac{\partial\varphi}{\partial x},\frac{\partial\varphi}{\partial y}\right).
# $$
#
# The coefficients $A$, $B$, and $C$ may depend on $x$, $y$, $\varphi$, and its first derivatives, but not on its second derivatives. At least one of them must be nonzero for the equation to be second order at the point considered. The right-hand side $R$ collects sources and lower-order terms. Here we use the convention **$2B$ for the mixed-derivative coefficient**. For a sufficiently smooth field, the two mixed derivatives are equal: $\partial_{xy} \varphi=\partial_{yx} \varphi$.
#
# #### Linear, quasilinear, and nonlinear equations
#
# - **Linear.** All coefficients depend only on the independent variables, and the field and all its derivatives enter linearly. In the form above, this means $A=A(x,y)$, $B=B(x,y)$, $C=C(x,y)$, and
#
#   $$R=f(x,y)-D(x,y)\partial_x \varphi-E(x,y)\partial_y \varphi-F(x,y)\varphi.$$
#
#   Here $D$, $E$, and $F$ are prescribed coefficients, and $f$ is a prescribed source. The notation $\partial_x\varphi$ denotes the partial derivative of $\varphi$ with respect to $x$. A nonzero source makes the equation inhomogeneous, but does not make it nonlinear.
# - **Quasilinear.** The highest-order derivatives enter linearly, but their coefficients may depend on the field or its first derivatives. The right-hand side can also depend nonlinearly on these lower-order quantities. For example, a coefficient $A=1+\varphi^2$ makes the equation nonlinear overall while retaining a linear dependence on $\partial_{xx} \varphi$.
# - **Nonlinear.** Any nonlinear dependence on the field or its derivatives makes the equation nonlinear overall. Quasilinear equations therefore form a subclass of nonlinear equations. If the highest-order derivatives themselves enter nonlinearly, as in $(\partial_{xx} \varphi)^2+\partial_{yy} \varphi=0$, the equation is called **fully nonlinear** and is not covered by the coefficient form above.
#
# This distinction matters because it determines which mathematical and numerical techniques can be applied directly. For a linear differential operator, linear combinations of solutions solve the correspondingly combined forcing and initial and boundary data. For a fixed homogeneous problem, this gives the superposition principle. Nonlinear equations generally do not obey this principle; their discretization typically requires updating solution-dependent coefficients or solving a nonlinear system.
#
# **The classification techniques introduced below apply directly to linear equations with constant coefficients.** For linear equations with variable coefficients, we evaluate and freeze the coefficients of the highest-order derivatives at a point to obtain a local classification.
#
# For quasilinear and fully nonlinear equations, the linear wave ansatz used below is not directly applicable to the complete equation. We first linearize about a sufficiently smooth reference solution and evaluate the resulting coefficients of the highest-order derivatives locally. The classification may therefore depend on the solution and vary across the domain. For a quasilinear equation, the original coefficients $A$, $B$, and $C$ already provide the coefficients of the highest-order derivatives of this local linearization; for a fully nonlinear equation, we first linearize about a reference solution to obtain these coefficients.
#
# #### The highest-order terms determine the PDE type
#
# We consider a diffusion equation with a linear reaction term:
#
# $$
# \frac{\partial\varphi}{\partial t}=\alpha\frac{\partial^2\varphi}{\partial x^2}+\lambda\varphi,\qquad\alpha>0.
# $$
#
# Here $\alpha$ is the diffusivity, while $\lambda$ is a reaction rate: a positive value produces growth, and a negative value produces decay. On a periodic domain, an initial disturbance $\varphi(x,0)=\varepsilon\cos(kx)$ has the exact solution
#
# $$
# \varphi(x,t)=\varepsilon\exp\bigl[(\lambda-\alpha k^2)t\bigr]\cos(kx).
# $$
#
# The amplitude $\varepsilon$ sets the initial disturbance size, and the wavenumber $k$ determines its wavelength, $2\pi/k$. The second derivative of $\cos(kx)$ is $-k^2\cos(kx)$, so diffusion contributes a decay rate $\alpha k^2$, whereas the reaction contributes a growth rate $\lambda$.
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Broad and short-wavelength disturbances under diffusion and reaction"><source src="media/pde-reaction-diffusion-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Broad and short-wavelength disturbances under diffusion and reaction"><source src="media/pde-reaction-diffusion-dark.mp4" type="video/mp4"></video>
#
# The animation compares two disturbances on the dimensionless periodic domain $0\leq x\leq2\pi$, with $\varepsilon=0.5$, $\alpha=0.1$, and $\lambda=0.5$. The solid curves show the full equation. For comparison, the dashed curves show reaction alone, and the dotted curves show diffusion alone. Both panels use the same vertical scale.
#
# - For the broad disturbance, $k=1$, the net growth rate is $0.5-0.1=0.4$: reaction outweighs diffusion and the amplitude grows.
# - For the shorter-wavelength disturbance, $k=4$, the net growth rate is $0.5-1.6=-1.1$: diffusion outweighs reaction and the amplitude decays.
#
# As the wavelength decreases, the contribution from the second derivative increases with $k^2$, while the reaction rate stays unchanged. This illustrates how the highest-order term controls the response at sufficiently small spatial scales. Changing $\lambda$ alters growth or decay, but the equation remains parabolic as long as $\alpha>0$.
#
# The growing broad disturbance shows that lower-order terms can dominate at larger spatial scales and must still be retained when solving the problem.
#
# For the general second-order equation, we classify its **principal part**, which consists of the highest-order terms. We therefore examine
#
# $$
# \underbrace{A\partial_{xx} \varphi+2B\partial_{xy} \varphi+C\partial_{yy} \varphi}_{\text{principal part}}=0.
# $$
#
# One way to understand this is to consider a smooth, invertible change of independent variables. The chain rule transforms the second derivatives into combinations of second derivatives in the new coordinates, together with additional lower-order terms. A term containing only the field or its first derivatives cannot produce a second derivative under this transformation. Thus $A$, $B$, and $C$ determine the structure of the highest-order part in any such coordinate system. This structure also determines the characteristic directions that we will use for classification.
#
# Identifying a degenerate principal part as parabolic does not by itself determine a well-posed evolution direction: the remaining terms must also be examined.
#
# If the principal coefficients depend on position or on the solution, classification is local and the type may change across the domain.
#
# #### Classification using a wave vector
#
# We recall the one-dimensional advection example above. Its characteristic lines are $x=x_0+ut$: the initial profile is transported along these lines, and the field remains constant along each one. There, we already knew the characteristic lines from the analytical solution.
#
# We now want to find the characteristic lines directly from the PDE, without first solving it. We look for admissible orientations of lines along which a wave profile has constant phase. A vector normal to such a line describes its orientation; this vector points across the line, not along it.
#
# We consider constant coefficients, or coefficients frozen at a particular point. To determine the admissible orientations, we introduce the wave vector $\mathbf{n}=(n_x,n_y)$ and the plane-wave ansatz
#
# $$
# \varphi(x,y)=\hat\varphi\exp\bigl[\mathrm{i}(n_xx+n_yy)\bigr],\qquad\mathbf{n}\neq\mathbf{0}.
# $$
#
# Here $\hat\varphi\neq0$ is a constant amplitude, $\mathrm{i}^2=-1$, and $n_x,n_y$ are the wave-vector components. For real components, the real part describes a sinusoidal field. Lines of constant phase satisfy $n_xx+n_yy=\mathrm{const.}$, and $\mathbf{n}$ is normal to these lines. The wave vector specifies spatial variation; it does not by itself specify a propagation speed. One of the independent variables may represent time when classifying an evolution equation.
#
# ![Parallel constant-phase lines in the x-y plane and their normal wave vector, with components n x and n y.](media/pde-wave-vector-light.png){.light-content width=100%}
# ![Parallel constant-phase lines in the x-y plane and their normal wave vector, with components n x and n y.](media/pde-wave-vector-dark.png){.dark-content width=100%}
#
# We differentiate the ansatz to obtain
#
# $$
# \begin{aligned}
# \partial_x \varphi&=\mathrm{i}n_x\varphi,&\qquad\partial_y \varphi&=\mathrm{i}n_y\varphi,\\
# \partial_{xx} \varphi&=-n_x^2\varphi,&\partial_{yy} \varphi&=-n_y^2\varphi,\\
# \partial_{xy} \varphi&=\mathrm{i}n_x\mathrm{i}n_y\varphi=-n_xn_y\varphi.
# \end{aligned}
# $$
#
# We substitute into the homogeneous principal-part equation to obtain
#
# $$
# A\partial_{xx} \varphi+2B\partial_{xy} \varphi+C\partial_{yy} \varphi
# =-\left(An_x^2+2Bn_xn_y+Cn_y^2\right)\hat\varphi\exp\bigl[\mathrm{i}(n_xx+n_yy)\bigr]=0.
# $$
#
# The amplitude and exponential are nonzero, so we divide them out:
#
# $$An_x^2+2Bn_xn_y+Cn_y^2=0.$$
#
# This condition determines the orientations of the characteristic lines. The wave vector is perpendicular to each such line. We count distinct directions, not the infinitely many possible vector magnitudes: multiplying $\mathbf{n}$ by a nonzero real number leaves its direction unchanged.
#
# For $n_y\neq0$, we define $\tau=n_x/n_y$ and divide by $n_y^2$:
#
# $$A\tau^2+2B\tau+C=0.$$
#
# If $A\neq0$, the quadratic formula gives
#
# $$\tau_{\pm}=\frac{-B\pm\sqrt{B^2-AC}}{A}.$$
#
# We recall how the quadratic formula determines the number of real roots. The two expressions for $\tau$ differ only in the sign preceding the square root, so the quantity **inside the square root** is decisive:
#
# - If $B^2-AC>0$, the square root is a positive real number. Adding or subtracting it gives two distinct real roots.
# - If $B^2-AC=0$, the square root vanishes. The two expressions coincide, giving one repeated real root, $\tau=-B/A$.
# - If $B^2-AC<0$, the square root is not real. There are no real roots, although two complex roots exist.
#
# Geometrically, the roots are the intersections of the graph $q(\tau)=A\tau^2+2B\tau+C$ with the horizontal axis. Two distinct roots correspond to two crossings; a repeated root corresponds to the parabola touching the axis; no real roots means no intersections.
#
# For a quadratic $a\tau^2+b\tau+c=0$, the quantity $b^2-4ac$ is called the **discriminant** because it distinguishes these three cases. Here $a=A$, $b=2B$, and $c=C$, so the discriminant is
#
# $$ (2B)^2-4AC=4(B^2-AC). $$
#
# The positive factor $4$ does not change the sign. We therefore use the reduced discriminant $\Delta=B^2-AC$ to count the real characteristic directions and classify the PDE:

#
# | Condition | Real characteristic directions | PDE type |
# |:---|:---|:---|
# | $\Delta>0$ | Two distinct directions | Hyperbolic |
# | $\Delta=0$ | One repeated direction | Parabolic |
# | $\Delta<0$ | No real directions | Elliptic |
#
# This classification applies when the principal part is nonzero. The ratio $\tau$ omits directions with $n_y=0$. Those must be checked in the original condition: they are possible when $A=0$. If $A=0$ and $B\neq0$, one direction comes from the remaining linear equation for $\tau$ and the other has $n_y=0$. If $A=B=0$ but $C\neq0$, the single repeated direction has $n_y=0$. Thus the discriminant criterion still applies even when the displayed quadratic formula cannot be used.
#
#
# The names elliptic, hyperbolic, and parabolic originate from the classification of conic sections; their geometric interpretation is not otherwise needed or relevant for the PDE classification considered here.

# %% [markdown]
# #### Example — compressible potential flow past a slender body
#
# We consider steady, two-dimensional, inviscid, irrotational flow past a slender body in a uniform free stream. The body thickness and surface slopes are small, so the velocity disturbances are small compared with the free-stream speed. We write the total velocity potential as
#
# $$\Phi(x,y)=U_\infty x+\psi(x,y),\qquad
# u=U_\infty+\partial_x \psi,\qquad v=\partial_y \psi.$$
#
# Here $x$ is the free-stream direction, $y$ is transverse, $U_\infty$ is the uniform incoming speed, $\Phi$ is the total velocity potential, and $\psi$ is the disturbance potential. The components $u$ and $v$ form the velocity vector $\mathbf{u}$. The small-disturbance assumption requires $|\partial_x \psi|,|\partial_y \psi|\ll U_\infty$.
#
# ![Analytical incompressible potential flow around a slender ellipse, with streamlines and calculated velocity-disturbance vectors.](media/slender-body-light.png){.light-content width=100%}
# ![Analytical incompressible potential flow around a slender ellipse, with streamlines and calculated velocity-disturbance vectors.](media/slender-body-dark.png){.dark-content width=100%}
#
# The figure shows an analytical **incompressible potential-flow solution** around a slender ellipse to illustrate the geometry and disturbance field. Blue streamlines follow the calculated total velocity. The red arrows show the full disturbance $\nabla\psi=\mathbf{u}-U_\infty\mathbf{e}_x$, including its streamwise and transverse components. Their lengths use one common scale, indicated by the reference arrow. Coordinates are normalized by the body length $L$; the ellipse thickness is $0.11L$. The solution satisfies zero normal velocity on the body and approaches the uniform free stream far away. The figure illustrates the disturbance field; it is not a compressible or supersonic simulation.
#
# Linearizing the compressible potential-flow equation about the uniform free stream gives
#
# $$\boxed{(1-M_\infty^2)\partial_{xx} \psi+\partial_{yy} \psi=0},\qquad
# M_\infty=\frac{U_\infty}{c_\infty},$$
#
# where $c_\infty$ is the free-stream sound speed and $M_\infty$ is the free-stream Mach number. The model describes smooth small disturbances; it does not resolve shock structure or viscous layers. Near sonic conditions, the linearization ceases to be a reliable physical approximation because terms discarded during linearization can become important.
#
# Comparing with $A\partial_{xx} \psi+2B\partial_{xy} \psi+C\partial_{yy} \psi=0$, we identify
#
# $$A=1-M_\infty^2,\qquad B=0,\qquad C=1.$$
#
# We substitute these coefficients into the reduced discriminant to obtain
#
# $$\Delta=B^2-AC=0-(1-M_\infty^2)\cdot1=M_\infty^2-1.$$
#
# We must therefore distinguish three cases: supersonic flow, the sonic limit approached from the supersonic side, and subsonic flow.
#
# ##### Supersonic case: $M_\infty>1$
#
# Here $\Delta>0$, so the equation is hyperbolic. We define $\beta=\sqrt{M_\infty^2-1}>0$. The equation becomes $-\beta^2\partial_{xx} \psi+\partial_{yy} \psi=0$. The wave-vector condition is
#
# $$-\beta^2n_x^2+n_y^2=0\quad\Rightarrow\quad n_y=\pm\beta n_x.$$
#
# Equivalently, the two solutions for the ratio are
#
# $$\tau_\pm=\frac{-B\pm\sqrt\Delta}{A}
# =\frac{\pm\beta}{-\beta^2}=\mp\frac{1}{\beta}.$$
#
# Thus the two characteristic families are
#
# $$x+\beta y=\mathrm{const.},\qquad x-\beta y=\mathrm{const.},\qquad
# \frac{\mathrm{d}y}{\mathrm{d}x}=\mp\frac{1}{\beta}.$$
#
# A general local smooth solution is the superposition
#
# $$\boxed{\psi(x,y)=F(x+\beta y)+G(x-\beta y)},$$
#
# where $F$ and $G$ are arbitrary twice-differentiable functions, determined by the prescribed data. Each contribution is constant along one characteristic family; their sum need not be constant along either family. Substituting the expression gives $\partial_{xx} \psi=F''+G''$ and $\partial_{yy} \psi=\beta^2(F''+G'')$, confirming that it solves the PDE.
#
# The characteristics make the Mach angle $\mu_M$ with the free-stream direction:
#
# $$\tan\mu_M=\frac{1}{\sqrt{M_\infty^2-1}},\qquad\sin\mu_M=\frac{1}{M_\infty}.$$
#
# ![Characteristic wedges upstream and downstream of P, showing the domain of dependence and range of influence for supersonic downstream marching.](media/potential-hyperbolic-light.png){.light-content width=100%}
# ![Characteristic wedges upstream and downstream of P, showing the domain of dependence and range of influence for supersonic downstream marching.](media/potential-hyperbolic-dark.png){.dark-content width=100%}
#
# In a downstream-marching formulation, the upstream wedge bounded by characteristics through $P=(x_P,y_P)$ is its **domain of dependence**: data prescribed upstream can affect $P$ only through this wedge. The downstream wedge is its **range of influence**: a disturbance at $P$ can affect downstream locations within it. On an upstream data line $x=x_0<x_P$, the relevant interval is
#
# $$y_P-\frac{x_P-x_0}{\beta}\leq y\leq y_P+\frac{x_P-x_0}{\beta}.$$
#
# These are spatial diagrams for a steady problem, with $x$ playing the role of a marching variable. They are not time histories. The diagram illustrates $M_\infty=2$, with $P$ upstream of a faintly drawn slender body. The body lies outside the upstream dependence region and cannot influence $P$ in this supersonic model. The shaded downstream wedge applies only in the fluid; the body surface imposes boundary conditions, and interactions there are not shown.
#
# ##### Sonic limit: $M_\infty\to1^+$
#
# We now approach sonic conditions from the supersonic side. For every $M_\infty>1$, the equation remains hyperbolic, but
#
# $$\beta=\sqrt{M_\infty^2-1}\to0^+,\qquad
# \Delta=M_\infty^2-1\to0^+,\qquad
# \mu_M=\arcsin(1/M_\infty)\to90^\circ.$$
#
# The characteristic slopes $\mathrm{d}y/\mathrm{d}x=\pm1/\beta$ become unbounded. The two families therefore approach the same transverse orientation, $x=\mathrm{const.}$. The upstream dependence wedge and downstream influence wedge open out toward the upstream and downstream half-planes, respectively.
#
# ![Limiting upstream domain of dependence and downstream range of influence as the supersonic Mach number approaches one, with P upstream of a faint slender body.](media/potential-sonic-light.png){.light-content width=100%}
# ![Limiting upstream domain of dependence and downstream range of influence as the supersonic Mach number approaches one, with P upstream of a faint slender body.](media/potential-sonic-dark.png){.dark-content width=100%}
#
# The sketch shows the **limiting characteristic geometry**, not an evolution problem obtained by substituting $M_\infty=1$. In the supersonic limit, a point at any fixed transverse distance and strictly upstream or downstream of $P$ eventually lies inside the corresponding wedge. The body is excluded from the fluid domain, and boundary interactions are not shown.
#
# At exactly $M_\infty=1$, the discriminant vanishes and the principal part is algebraically parabolic. However, the linearized equation degenerates to $\partial_{yy} \psi=0$, with solutions $\psi=a(x)y+b(x)$; it does not itself determine downstream evolution. Moreover, the linear small-disturbance approximation is not reliable near sonic conditions, where discarded nonlinear terms can become important. The limiting sketch therefore illustrates the classification geometry rather than a quantitative prediction of sonic flow.
#
# ##### Subsonic case: $M_\infty<1$
#
# Here $\Delta<0$, so the equation is elliptic. We set $\eta=\sqrt{1-M_\infty^2}>0$. Then
#
# $$\eta^2\partial_{xx} \psi+\partial_{yy} \psi=0,\qquad
# \tau_\pm=\pm\frac{\mathrm{i}}{\eta}.$$
#
# There are no real characteristic directions. Using the scaled coordinate $X=x/\eta$ transforms the equation into $\partial_{XX} \psi+\partial_{yy} \psi=0$, the Laplace equation.
#
# ![Global coupling of an elliptic boundary-value problem, with P connected to the surrounding boundary and no characteristic wedge.](media/potential-elliptic-light.png){.light-content width=100%}
# ![Global coupling of an elliptic boundary-value problem, with P connected to the surrounding boundary and no characteristic wedge.](media/potential-elliptic-dark.png){.dark-content width=100%}
#
# The solution is determined as a boundary-value problem. In general, boundary data around the entire connected domain influence the value at $P$; there is no finite upstream dependence wedge. In this spatial sense, the domain of dependence and the range of influence coincide: the solution is globally coupled throughout the connected fluid domain. Likewise, a localized change in the boundary data or an introduced source can affect the solution throughout the domain, including upstream locations. The arrows illustrate spatial coupling, not instantaneous physical propagation in time. This is a steady model, and dependence and influence are understood through the complete boundary-value problem.

# %% [markdown]
# ### From second-order PDEs to first-order systems of PDEs
#
# We have classified scalar second-order PDEs through their characteristic directions. Fluid mechanics also gives us coupled systems of first-order equations, for example when expressing conservation of mass, momentum, and energy. These descriptions are closely connected: we can rewrite a second-order equation as a first-order system by introducing its first derivatives as additional unknowns.
#
# #### Introducing the first derivatives as unknowns
#
# We consider again
#
# $$A\partial_{xx} \varphi+2B\partial_{xy} \varphi+C\partial_{yy} \varphi
# =R(x,y,\varphi,\partial_x \varphi,\partial_y \varphi).$$
#
# We define two new fields
#
# $$p=\partial_x \varphi,\qquad q=\partial_y \varphi.$$
#
# Here $p$ and $q$ are auxiliary derivative fields, not pressure or mass flux.
#
# The second derivatives of $\varphi$ now become first derivatives of $p$ and $q$:
#
# $$\partial_{xx} \varphi=\partial_x p,\qquad\partial_{xy} \varphi=\partial_y p=\partial_x q,\qquad\partial_{yy} \varphi=\partial_y q.$$
#
# The original equation can therefore be replaced by the first-order system
#
# $$
# \begin{aligned}
# A \partial_x p+2B \partial_y p+C \partial_y q&=R(x,y,\varphi,p,q),\\
# \partial_y p-\partial_x q&=0,\\
# \partial_x \varphi&=p,\qquad\partial_y \varphi=q.
# \end{aligned}
# $$
#
# The **compatibility condition** $\partial_y p=\partial_x q$ follows from equality of the mixed derivatives. It ensures locally that the two fields can be derivatives of the same scalar potential. The last two relations retain the definition of that potential and are needed when the right-hand side or the coefficients depend on $\varphi$ itself.
#
# If the equation depends only on derivatives of $\varphi$, the first two equations form a closed system for $p$ and $q$. We can then reconstruct $\varphi$ by integration, with an appropriate reference value. The boundary or initial conditions must also be translated consistently; the new derivative fields cannot be prescribed independently of the original field.
#
# For the derivative fields, the first two equations have the vector–matrix form
#
# $$
# \underbrace{\begin{pmatrix}A&0\\0&-1\end{pmatrix}}_{\mathbf{M}_x}
# \frac{\partial\mathbf{r}}{\partial x}
# +\underbrace{\begin{pmatrix}2B&C\\1&0\end{pmatrix}}_{\mathbf{M}_y}
# \frac{\partial\mathbf{r}}{\partial y}
# =\underbrace{\begin{pmatrix}R\\0\end{pmatrix}}_{\mathbf{s}},
# \qquad\mathbf{r}=\begin{pmatrix}p\\q\end{pmatrix}.
# $$
#
# Equivalently, $\mathbf{M}_x\partial_x \mathbf{r}+\mathbf{M}_y\partial_y \mathbf{r}=\mathbf{s}$. The vector $\mathbf{r}$ contains the two derivative fields, $\mathbf{M}_x$ and $\mathbf{M}_y$ are $2\times2$ coefficient matrices, and $\mathbf{s}$ is the right-hand-side vector. Their entries retain any dependence on position or the unknown fields. If the coefficients or $R$ depend on $\varphi$ itself, the derivative definitions must also be retained to complete the system.
#
# #### Classification from the determinant of a first-order system
#
# We again examine the homogeneous principal part, using constant coefficients or coefficients frozen locally:
#
# $$\mathbf{M}_x\partial_x \mathbf{r}+\mathbf{M}_y\partial_y \mathbf{r}=\mathbf{0}.$$
#
# As in the classification of second-order PDEs, we now use a plane-wave ansatz, this time with a vector-valued amplitude:
#
# $$\mathbf{r}(x,y)=\hat{\mathbf{r}}\exp\bigl[\mathrm{i}(n_xx+n_yy)\bigr],\qquad
# \hat{\mathbf{r}}=\begin{pmatrix}\hat p\\\hat q\end{pmatrix}\neq\mathbf{0}.$$
#
# The amplitude is now a vector because the system has two unknown fields. Differentiation gives
#
# $$\partial_x \mathbf{r}=\mathrm{i}n_x\hat{\mathbf{r}}e^{\mathrm{i}(n_xx+n_yy)},\qquad
# \partial_y \mathbf{r}=\mathrm{i}n_y\hat{\mathbf{r}}e^{\mathrm{i}(n_xx+n_yy)}.$$
#
# We substitute and divide out the nonzero factor $\mathrm{i}e^{\mathrm{i}(n_xx+n_yy)}$ to obtain a homogeneous linear algebraic system for the amplitude:
#
# $$\underbrace{(n_x\mathbf{M}_x+n_y\mathbf{M}_y)}_{\mathbf{K}(\mathbf{n})}\hat{\mathbf{r}}=\mathbf{0}.$$
#
# If $\mathbf{K}$ is invertible, multiplication by its inverse gives only the trivial solution $\hat{\mathbf{r}}=\mathbf{0}$. A **nontrivial amplitude exists if and only if the coefficient matrix is singular**, so
#
# $$\boxed{\det(n_x\mathbf{M}_x+n_y\mathbf{M}_y)=0.}$$
#
# For the matrices obtained from our second-order scalar equation,
#
# $$
# \mathbf{K}(\mathbf{n})=
# n_x\begin{pmatrix}A&0\\0&-1\end{pmatrix}
# +n_y\begin{pmatrix}2B&C\\1&0\end{pmatrix}
# =\begin{pmatrix}An_x+2Bn_y&Cn_y\\n_y&-n_x\end{pmatrix}.
# $$
#
# Expanding the determinant gives
#
# $$
# \begin{aligned}
# \det\mathbf{K}
# &=(An_x+2Bn_y)(-n_x)-(Cn_y)n_y\\
# &=-An_x^2-2Bn_xn_y-Cn_y^2.
# \end{aligned}
# $$
#
# Setting it to zero and multiplying by $-1$ recovers exactly the condition found for the scalar second-order equation:
#
# $$An_x^2+2Bn_xn_y+Cn_y^2=0.$$
#
# For $n_y\neq0$, division by $n_y^2$ and the substitution $\tau=n_x/n_y$ give
#
# $$\det(\tau\mathbf{M}_x+\mathbf{M}_y)=0
# \quad\Longleftrightarrow\quad A\tau^2+2B\tau+C=0.$$
#
# Thus, for this two-field system obtained from a second-order scalar PDE, the same reduced discriminant $\Delta=B^2-AC$ gives the same classification:
#
# | Condition | Characteristic directions | Type inherited from the scalar PDE |
# |:---|:---|:---|
# | $\Delta>0$ | Two distinct real directions | Hyperbolic |
# | $\Delta=0$ | One repeated real direction | Parabolic principal part |
# | $\Delta<0$ | No real directions | Elliptic |
#
# Directions with $n_y=0$ must still be checked before division, as in the scalar derivation. Rewriting the equation as a system has preserved its characteristic condition. For each admissible wave vector, solving $\mathbf{K}\hat{\mathbf{r}}=\mathbf{0}$ also determines the allowed relative amplitudes of $p$ and $q$. In particular, its second row gives $n_y\hat p-n_x\hat q=0$, consistent with both fields being derivatives of the same scalar potential.
#
# For a general square first-order system with more unknowns, the determinant condition remains the starting point, but the polynomial can have higher degree. The two-root discriminant table above is specific to this reduction of a scalar second-order equation. For general evolution systems, hyperbolicity also requires examining whether the real characteristic speeds have a complete set of independent eigenvectors; a repeated root alone does not establish parabolic behavior.

# %% [markdown]
# #### Special case: conservation equations as first-order systems
#
# Conservation equations have a particular structure: the local change of a conserved quantity is balanced by transport through fluxes and by sources. In one spatial dimension, a system of conservation equations can be written as
#
# $$\frac{\partial\mathbf{q}}{\partial t}
# +\frac{\partial\mathbf{f}(\mathbf{q})}{\partial x}
# =\mathbf{s}.$$
#
# Here $\mathbf{q}(x,t)\in\mathbb{R}^m$ contains the conserved quantities per unit volume, $\mathbf{f}(\mathbf{q})\in\mathbb{R}^m$ contains their fluxes in the $x$ direction, and $\mathbf{s}\in\mathbb{R}^m$ contains source terms. We consider fluxes that depend on the state $\mathbf{q}$, without additional explicit dependence on position or time. A system with nonzero sources is also called a system of balance laws; setting $\mathbf{s}=\mathbf{0}$ gives the source-free conservation laws.
#
# For a sufficiently smooth solution, the chain rule gives
#
# $$\frac{\partial\mathbf{f}(\mathbf{q})}{\partial x}
# =\underbrace{\frac{\partial\mathbf{f}}{\partial\mathbf{q}}}_{\mathbf{J}(\mathbf{q})}
# \frac{\partial\mathbf{q}}{\partial x},\qquad
# J_{ij}=\frac{\partial f_i}{\partial q_j},\quad i,j=1,\ldots,m.$$
#
# Thus the conservation equations take the first-order quasilinear form
#
# $$\boxed{\partial_t \mathbf{q}+\mathbf{J}(\mathbf{q})\partial_x \mathbf{q}=\mathbf{s}.}$$
#
# The $m\times m$ matrix $\mathbf{J}$ is the **flux Jacobian**. Comparing with the general first-order system, the coefficient of $\partial_t \mathbf{q}$ is the identity matrix, while the coefficient of $\partial_x \mathbf{q}$ is the flux Jacobian. If the flux is linear in the state, the Jacobian is constant and the system is linear. Otherwise, its coefficients depend on the solution, and classification is performed locally by evaluating the Jacobian at a reference state.
#
# This conversion uses differentiability. At a discontinuity, such as a shock, the expanded quasilinear form cannot replace the conservative formulation; the integral conservation laws supply the required jump conditions. In the following classification, we examine smooth states or a local linearization about such a state.
#
# ##### Example: the one-dimensional compressible Euler equations
#
# We consider compressible gas flow in a straight duct of constant cross-sectional area. We neglect viscosity, heat conduction, body forces, and other sources. Conservation of mass, momentum, and total energy gives the one-dimensional Euler equations:
#
# $$
# \begin{aligned}
# \frac{\partial\rho}{\partial t}+\frac{\partial(\rho u)}{\partial x}&=0,\\
# \frac{\partial(\rho u)}{\partial t}+\frac{\partial(\rho u^2+p)}{\partial x}&=0,\\
# \frac{\partial(\rho E)}{\partial t}+\frac{\partial\bigl[u(\rho E+p)\bigr]}{\partial x}&=0.
# \end{aligned}
# $$
#
# Here $\rho$ is density, $u$ is velocity in the $x$ direction, $p$ is pressure, and $E=e+u^2/2$ is total energy per unit mass, consisting of specific internal energy $e$ and specific kinetic energy. We close the system with the calorically perfect ideal-gas relation
#
# $$p=(\gamma-1)\rho e=(\gamma-1)\left(\rho E-\frac12\rho u^2\right),\qquad\gamma=\frac{c_p}{c_v}>1,$$
#
# where $c_p$ and $c_v$ are the constant specific heat capacities. Unlike the auxiliary derivative field $p$ used earlier, $p$ now denotes physical pressure.
#
# ::: {.callout-note collapse="true" title="Additional reading: where does the gas relation come from?"}
# The closure combines the ideal-gas law with the calorically perfect gas assumption. The ideal-gas law is
#
# $$p=\rho R_sT,$$
#
# where $T$ is absolute temperature and $R_s$ is the specific gas constant. For a calorically perfect gas, the specific heat capacities are constant. Choosing the internal-energy reference so that the additive constant vanishes gives
#
# $$e=c_vT.$$
#
# For an ideal gas, $c_p-c_v=R_s$. Using $\gamma=c_p/c_v$, we obtain
#
# $$R_s=(\gamma-1)c_v.$$
#
# We substitute into the ideal-gas law to obtain
#
# $$p=\rho(\gamma-1)c_vT=(\gamma-1)\rho e.$$
#
# Finally, $E=e+u^2/2$ gives
#
# $$p=(\gamma-1)\left(\rho E-\frac12\rho u^2\right).$$
#
# This is a material model that closes the Euler equations; it does not follow from the conservation laws alone.
# :::
#
# **Conserved-state and flux vectors.** We define
#
# $$
# \mathbf{q}=\begin{pmatrix}\rho\\\rho u\\\rho E\end{pmatrix},\qquad
# \mathbf{f}(\mathbf{q})=\begin{pmatrix}\rho u\\\rho u^2+p\\u(\rho E+p)\end{pmatrix}.
# $$
#
# Both vectors have three components. The entries of $\mathbf{q}$ are mass, momentum, and total energy per unit volume; the entries of $\mathbf{f}$ are the corresponding fluxes. Their conservation equations become
#
# $$\partial_t \mathbf{q}+\partial_x\mathbf{f}(\mathbf{q})=\mathbf{0}.$$
#
# **Deriving the flux Jacobian.** To differentiate with respect to the conserved variables, we write everything in terms of the independent components $q_1$, $q_2$, and $q_3$:
#
# We start from the definitions of the conserved components:
#
# $$q_1=\rho,\qquad q_2=\rho u,\qquad q_3=\rho E.$$
#
# These are the independent variables for differentiation. Velocity and specific total energy are derived quantities, obtained by dividing momentum density and energy density by mass density:
#
# $$\rho=q_1,\qquad u=\frac{q_2}{q_1},\qquad E=\frac{q_3}{q_1},\qquad q_1>0.$$
#
# First, we express the kinetic-energy density and pressure in these variables:
#
# $$
# \frac12\rho u^2=\frac12q_1\left(\frac{q_2}{q_1}\right)^2
# =\frac{q_2^2}{2q_1},\qquad
# p(\mathbf{q})=(\gamma-1)\left(q_3-\frac{q_2^2}{2q_1}\right).
# $$
#
# We can now convert each flux component separately:
#
# 1. **Mass flux.** The product $\rho u$ is already the second conserved component:
#
#    $$f_1=\rho u=q_1\frac{q_2}{q_1}=q_2.$$
#
# 2. **Momentum flux.** Replacing both density and velocity gives
#
#    $$\rho u^2=q_1\left(\frac{q_2}{q_1}\right)^2=\frac{q_2^2}{q_1}.$$
#
#    Substituting pressure as well, we obtain
#
#    $$\begin{aligned}
#    f_2&=\rho u^2+p\\
#    &=\frac{q_2^2}{q_1}+(\gamma-1)\left(q_3-\frac{q_2^2}{2q_1}\right)\\
#    &=\frac{3-\gamma}{2}\frac{q_2^2}{q_1}+(\gamma-1)q_3.
#    \end{aligned}$$
#
# 3. **Total-energy flux.** Here $\rho E=q_3$, while the velocity multiplying the energy and pressure is $u=q_2/q_1$:
#
#    $$\begin{aligned}
#    f_3&=u(\rho E+p)\\
#    &=\frac{q_2}{q_1}\left[q_3+(\gamma-1)\left(q_3-\frac{q_2^2}{2q_1}\right)\right]\\
#    &=\gamma\frac{q_2q_3}{q_1}-\frac{\gamma-1}{2}\frac{q_2^3}{q_1^2}.
#    \end{aligned}$$
#
# Every flux component is now explicitly a function of $q_1$, $q_2$, and $q_3$. In particular, $p$ and $u$ must not be treated as independent constants when differentiating with respect to these conserved variables. For compactness, we can retain the shorthand $p(\mathbf{q})$ and collect the results as
#
# $$
# \mathbf{f}(\mathbf{q})=
# \begin{pmatrix}
# q_2\\
# \dfrac{q_2^2}{q_1}+p(\mathbf{q})\\[4pt]
# \dfrac{q_2}{q_1}\bigl[q_3+p(\mathbf{q})\bigr]
# \end{pmatrix}.
# $$
#
# When taking each partial derivative, we hold the other conserved variables fixed. We first differentiate pressure and velocity to obtain
#
# $$
# \begin{aligned}
# \frac{\partial p}{\partial q_1}&=\frac{\gamma-1}{2}u^2,&\frac{\partial p}{\partial q_2}&=-(\gamma-1)u,&\frac{\partial p}{\partial q_3}&=\gamma-1,\\
# \frac{\partial u}{\partial q_1}&=-\frac{u}{\rho},&\frac{\partial u}{\partial q_2}&=\frac{1}{\rho},&\frac{\partial u}{\partial q_3}&=0.
# \end{aligned}
# $$
#
# The first flux is $f_1=q_2$, giving the first Jacobian row $(0,1,0)$. For the momentum flux, differentiating $f_2=q_2^2/q_1+p$ gives
#
# $$\left(\frac{\partial f_2}{\partial q_1},\frac{\partial f_2}{\partial q_2},\frac{\partial f_2}{\partial q_3}\right)
# =\left(\frac{\gamma-3}{2}u^2,(3-\gamma)u,\gamma-1\right).$$
#
# For the energy flux $f_3=u(q_3+p)$, the product rule gives
#
# $$\frac{\partial f_3}{\partial q_j}=(q_3+p)\frac{\partial u}{\partial q_j}
# +u\left(\delta_{3j}+\frac{\partial p}{\partial q_j}\right),$$
#
# where $\delta_{3j}$ is $1$ for $j=3$ and $0$ otherwise. For convenience, we introduce the specific total enthalpy $H=E+p/\rho=(q_3+p)/\rho$. We then substitute the pressure and velocity derivatives obtained above to obtain
#
# $$
# \boxed{\mathbf{J}(\mathbf{q})=\frac{\partial\mathbf{f}}{\partial\mathbf{q}}
# =\begin{pmatrix}
# 0&1&0\\[3pt]
# \dfrac{\gamma-3}{2}u^2&(3-\gamma)u&\gamma-1\\[6pt]
# u\left(\dfrac{\gamma-1}{2}u^2-H\right)&H-(\gamma-1)u^2&\gamma u
# \end{pmatrix}.}
# $$
#
# **Vector–matrix form.** For a smooth solution, the Euler equations can therefore be written as
#
# $$
# \frac{\partial}{\partial t}\begin{pmatrix}\rho\\\rho u\\\rho E\end{pmatrix}
# +\mathbf{J}(\mathbf{q})\frac{\partial}{\partial x}\begin{pmatrix}\rho\\\rho u\\\rho E\end{pmatrix}
# =\begin{pmatrix}0\\0\\0\end{pmatrix},
# \qquad\text{or}\qquad
# \partial_t \mathbf{q}+\mathbf{J}(\mathbf{q})\partial_x \mathbf{q}=\mathbf{0}.
# $$
#
# The flux Jacobian is a $3\times3$ matrix and depends on the local flow state through $u$ and $H$. The Euler equations are therefore nonlinear conservation laws, whose expanded form is quasilinear. Unlike two independent transport equations, mass, momentum, and energy are coupled through the fluxes and the equation of state.

# %% [markdown]
# #### Classification through characteristic variables
#
# The quasilinear conservation equations
#
# $$\partial_t \mathbf{q}+\mathbf{J}(\mathbf{q})\partial_x \mathbf{q}=\mathbf{0}$$
#
# look almost like the scalar convection equation $\partial_t \varphi+u\partial_x \varphi=0$. The difference is that $\mathbf{q}$ is a vector and the flux Jacobian $\mathbf{J}$ is a matrix rather than a scalar transport velocity. Its off-diagonal entries couple the different conserved quantities.
#
# To classify the system locally, we linearize about a uniform reference state $\overline{\mathbf{q}}$. The resulting constant-coefficient system is
#
# $$\partial_t\mathbf{v}+\mathbf{J}_0\partial_x\mathbf{v}=\mathbf{0},\qquad
# \mathbf{J}_0=\mathbf{J}(\overline{\mathbf{q}}).$$
#
# Here $\mathbf{v}$ describes perturbations of the conserved quantities, not the velocity vector. The detailed derivation is optional and can be expanded below.
#
# ::: {.callout-note collapse="true" title="Additional reading: linearization about a uniform reference state"}
#
# To classify the system locally, we first study small perturbations about a uniform reference state $\overline{\mathbf{q}}$. We write
#
# $$\mathbf{q}(x,t)=\overline{\mathbf{q}}+\varepsilon\mathbf{v}(x,t),\qquad 0<\varepsilon\ll1.$$
#
# Here $\varepsilon$ is a dimensionless parameter measuring the perturbation amplitude. The vector $\mathbf{v}$ describes changes in the conserved quantities and has the same units as $\mathbf{q}$; it is not the velocity vector. The actual perturbation is $\varepsilon\mathbf{v}$. We assume that the perturbation and its derivatives remain bounded as $\varepsilon\to0$.
#
# **1. We differentiate the perturbed state.** Because the reference state is constant in space and time, its derivatives vanish:
#
# $$\partial_t\mathbf{q}=\varepsilon\partial_t\mathbf{v},\qquad
# \partial_x\mathbf{q}=\varepsilon\partial_x\mathbf{v}.$$
#
# **2. We expand the flux Jacobian about the reference state.** For a sufficiently smooth flux, a Taylor expansion gives
#
# $$\mathbf{J}(\overline{\mathbf{q}}+\varepsilon\mathbf{v})
# =\mathbf{J}_0+\varepsilon\mathbf{J}_1[\mathbf{v}]+O(\varepsilon^2),\qquad
# \mathbf{J}_0=\mathbf{J}(\overline{\mathbf{q}}).$$
#
# The matrix $\mathbf{J}_1[\mathbf{v}]$ is the first-order change in the Jacobian caused by the perturbation. Component by component, it is
#
# $$\bigl(\mathbf{J}_1[\mathbf{v}]\bigr)_{ij}
# =\sum_{k=1}^{m}\left.\frac{\partial J_{ij}}{\partial q_k}\right|_{\overline{\mathbf{q}}}v_k,$$
#
# where $m$ is the number of conserved variables. Thus the Jacobian still changes with the state; we have not assumed this dependence away.
#
# **3. We substitute both expansions into the quasilinear equation.** We obtain
#
# $$\varepsilon\partial_t\mathbf{v}
# +\left(\mathbf{J}_0+\varepsilon\mathbf{J}_1[\mathbf{v}]+O(\varepsilon^2)\right)
# \varepsilon\partial_x\mathbf{v}=\mathbf{0},$$
#
# or, after multiplying out,
#
# $$\varepsilon\left(\partial_t\mathbf{v}+\mathbf{J}_0\partial_x\mathbf{v}\right)
# +\varepsilon^2\mathbf{J}_1[\mathbf{v}]\partial_x\mathbf{v}
# +O(\varepsilon^3)=\mathbf{0}.$$
#
# The change in the Jacobian is multiplied by the perturbation gradient, so its contribution is **second order in the perturbation amplitude**. We divide by $\varepsilon$ and retain only the leading terms as $\varepsilon\to0$:
#
# $$\boxed{\partial_t\mathbf{v}+\mathbf{J}_0\partial_x\mathbf{v}=\mathbf{0}.}$$
#
# This equation is linear because $\mathbf{J}_0$ is evaluated at the prescribed reference state and does not depend on $\mathbf{v}$. It has constant coefficients because that reference state is uniform. For the Euler equations, this means evaluating the Jacobian using the reference density, velocity, and pressure. The resulting system describes the leading-order evolution of small disturbances; it does not retain their nonlinear interaction.
#
# Uniformity matters: linearization about a spatially varying reference solution also produces terms involving its gradients. The constant-coefficient system used here is therefore an exact linearization about a uniform state, or a local frozen-coefficient model when we examine a nonuniform flow at a particular state.
#
# :::
#
# **What if the matrix were diagonal?** Then each component would satisfy its own convection equation, without coupling to the other components. We would have independent profiles travelling at their individual transport speeds. This motivates diagonalizing the Jacobian to identify variables in which the system decouples.
#
# ##### Eigenvectors define the new variables
#
# We find the eigenvalues and right eigenvectors of the reference Jacobian from
#
# $$\mathbf{J}_0\mathbf{r}_j=\lambda_j\mathbf{r}_j,\qquad\mathbf{r}_j\neq\mathbf{0}.$$
#
# If there are $m$ linearly independent eigenvectors, we collect them as columns of an invertible matrix $\mathbf{R}$:
#
# $$\mathbf{R}=\begin{pmatrix}\mathbf{r}_1&\cdots&\mathbf{r}_m\end{pmatrix},\qquad
# \boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_m).$$
#
# The eigenvector equations then give the eigendecomposition
#
# $$\mathbf{J}_0\mathbf{R}=\mathbf{R}\boldsymbol{\Lambda},\qquad
# \mathbf{J}_0=\mathbf{R}\boldsymbol{\Lambda}\mathbf{R}^{-1}.$$
#
# We substitute the eigendecomposition into the linearized PDE system:
#
# $$\partial_t \mathbf{v}+\mathbf{R}\boldsymbol{\Lambda}\mathbf{R}^{-1}\partial_x \mathbf{v}=\mathbf{0}.$$
#
# Next, we multiply the entire equation from the left by $\mathbf{R}^{-1}$. Using $\mathbf{R}^{-1}\mathbf{R}=\mathbf{I}$ gives
#
# $$\mathbf{R}^{-1}\partial_t \mathbf{v}+\boldsymbol{\Lambda}\mathbf{R}^{-1}\partial_x \mathbf{v}=\mathbf{0}.$$
#
# Because the eigenvector matrix is constant in this frozen linear model, we can move its inverse inside the derivatives:
#
# $$\frac{\partial}{\partial t}(\mathbf{R}^{-1}\mathbf{v})
# +\boldsymbol{\Lambda}\frac{\partial}{\partial x}(\mathbf{R}^{-1}\mathbf{v})=\mathbf{0}.$$
#
# This suggests introducing the **characteristic variables**
#
# $$\mathbf{w}=\mathbf{R}^{-1}\mathbf{v},\qquad\mathbf{v}=\mathbf{R}\mathbf{w}.$$
#
# Each entry of $\mathbf{w}$ is a linear combination of the original perturbations. In these transformed variables, the system becomes
#
# $$\boxed{\partial_t \mathbf{w}+\boldsymbol{\Lambda}\partial_x \mathbf{w}=\mathbf{0}.}$$
#
# For three equations, this means
#
# $$\begin{aligned}
# \partial_t w_1+\lambda_1\partial_x w_1&=0,\\
# \partial_t w_2+\lambda_2\partial_x w_2&=0,\\
# \partial_t w_3+\lambda_3\partial_x w_3&=0.
# \end{aligned}$$
#
# For real eigenvalues and initial profiles $w_{j,0}(x)$ on an unbounded domain, the solutions are
#
# $$w_j(x,t)=w_{j,0}(x-\lambda_jt).$$
#
# Thus the eigenvalues are the **propagation speeds** of the characteristic variables. The lines $x=x_0+\lambda_jt$ are their characteristics. Reconstructing the original perturbations gives
#
# $$\mathbf{v}(x,t)=\sum_{j=1}^m\mathbf{r}_j\,w_{j,0}(x-\lambda_jt).$$
#
# This exact decoupling applies to the constant-coefficient linearized system. In the nonlinear conservation laws, the eigenvectors generally vary with the state. A position- or time-dependent transformation introduces additional terms and does not automatically produce globally independent convection equations.
#
# ##### Classification of the first-order system
#
# The plane-wave ansatz provides the same interpretation. Substituting $\mathbf{v}=\hat{\mathbf{v}}\exp[\mathrm{i}(kx-\omega t)]$, with real $k\neq0$, gives
#
# $$\left(\mathbf{J}_0-\frac{\omega}{k}\mathbf{I}\right)\hat{\mathbf{v}}=\mathbf{0}.$$
#
# A nontrivial amplitude therefore requires $\det(\mathbf{J}_0-\lambda\mathbf{I})=0$, with $\lambda=\omega/k$. This is precisely the eigenvalue equation for the characteristic speeds.
#
# | Eigenstructure of the real matrix $\mathbf{J}_0$ | Classification or consequence |
# |:---|:---|
# | All eigenvalues are real, with $m$ independent eigenvectors | Hyperbolic; the frozen system decouples into real convection modes. |
# | All eigenvalues are real and distinct | Strictly hyperbolic; distinct eigenvalues guarantee independent eigenvectors. |
# | Eigenvalues repeat, but $m$ independent eigenvectors remain | Still hyperbolic; some modes travel at the same speed. |
# | All eigenvalues are real, but there are fewer than $m$ independent eigenvectors | Parabolic in the first-order-system terminology used here; also described as weakly hyperbolic. Complete diagonalization fails. |
# | No eigenvalue is real | Elliptic as a first-order system in the two variables $(x,t)$; it is not a hyperbolic time-evolution problem. |
# | Real and nonreal eigenvalues both occur | Mixed hyperbolic–elliptic character in this two-variable classification; we cannot represent all modes as real convection waves. |
#
# ::: {.callout-note title="Terminology: parabolic and weakly hyperbolic systems"}
# Some classifications of first-order systems use **parabolic** for a real spectrum with fewer than $m$ independent eigenvectors. We follow that convention here. In discussions of time-evolution problems, this defective real eigenstructure is also described as **weakly hyperbolic**. This terminology does not imply diffusion or the smoothing behavior of the heat equation. Repeated eigenvalues alone are insufficient: if a complete set of independent eigenvectors exists, the system remains hyperbolic.
# :::
#
# Nonreal eigenvalues need not be purely imaginary: they may have both nonzero real and imaginary parts. Since $\mathbf{J}_0$ is real, nonreal eigenvalues occur in complex-conjugate pairs. In particular, a real $3\times3$ matrix always has at least one real eigenvalue, so its three eigenvalues cannot all be nonreal.
#
# **Parabolic behavior is not identified by repeated eigenvalues alone.** A diagonalizable first-order system with repeated real speeds is still hyperbolic. Diffusive parabolic evolution normally involves second spatial derivatives, for example $\partial_t \mathbf{q}=\mathbf{D}\partial_{xx} \mathbf{q}$ with a positive-definite diffusion matrix. Such equations can be rewritten using auxiliary variables, but then they do not generally take the simple form $\partial_t \mathbf{q}+\mathbf{J}_0\partial_x \mathbf{q}=\mathbf{0}$ with an invertible coefficient of every time derivative. The parabolic principal-part classification encountered in the second-order reduction must therefore not be transferred to an arbitrary conservation system merely by counting repeated roots.
#
# ##### Example: three distinct propagation speeds
#
# We illustrate the characteristic variables using the Euler equations linearized about a uniform ideal-gas state. The three eigenvalues of its flux Jacobian are $\lambda_1=\bar u-\bar c$, $\lambda_2=\bar u$, and $\lambda_3=\bar u+\bar c$, where $\bar u$ is the reference velocity and $\bar c=\sqrt{\gamma\bar p/\bar\rho}>0$ is the reference sound speed. They are real and distinct, so the system is **strictly hyperbolic**.
#
# In dimensionless variables, we choose $\bar u=0.4$, $\bar c=1$, and $\gamma=1.4$. The speeds are therefore $-0.6$, $0.4$, and $1.4$. We prescribe the same initial Gaussian profile for each characteristic variable:
#
# $$w_{j,0}(x)=0.1\exp[-(x/0.35)^2],\qquad
# w_j(x,t)=0.1\exp\left[-\left(\frac{x-\lambda_jt}{0.35}\right)^2\right].$$
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Three Euler characteristic profiles travelling independently at distinct real speeds"><source src="media/euler-characteristic-modes-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Three Euler characteristic profiles travelling independently at distinct real speeds"><source src="media/euler-characteristic-modes-dark.mp4" type="video/mp4"></video>
#
# Each panel shows one characteristic variable. The faint dotted curve marks its initial profile, and the vertical dashed line tracks its current center. The first acoustic mode travels to the left, the entropy/contact mode moves with the mean flow, and the second acoustic mode travels to the right more quickly. All profiles retain their shapes and amplitudes. The axes use the same scales.
#
# These are transformed variables, not separate concentrations or independent physical fluids. Together they describe a single coupled Euler perturbation through $\mathbf{v}=\mathbf{R}\mathbf{w}$. Their amplitudes depend on the chosen eigenvector normalization. The animation evaluates the exact solutions of the frozen linearized system; it is not a simulation of finite-amplitude nonlinear Euler waves or shocks.
#
# **From characteristic modes to physical perturbations.** We can reconstruct density, velocity, and pressure changes from the same modes. For the eigenvector normalization used here,
#
# $$\rho'=w_1+w_2+w_3,\qquad
# u'=\frac{\bar c}{\bar\rho}(w_3-w_1),\qquad
# p'=\bar c^2(w_1+w_3).$$
#
# Primes denote departures from the uniform reference state. We additionally choose $\bar\rho=1$, so the reference pressure is $\bar p=\bar\rho\bar c^2/\gamma=1/1.4$. The following animation shows these physical perturbations, normalized by $\bar\rho$, $\bar c$, and $\bar\rho\bar c^2$, respectively:
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Reconstructed density velocity and pressure perturbations with contributions from the three Euler characteristic modes"><source src="media/euler-physical-perturbations-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Reconstructed density velocity and pressure perturbations with contributions from the three Euler characteristic modes"><source src="media/euler-physical-perturbations-dark.mp4" type="video/mp4"></video>
#
# Solid curves show the total physical perturbations; colored dashed curves show the contributions from the characteristic modes, and faint dotted curves mark the initial physical profiles. Density receives contributions from all three modes. Velocity and pressure receive contributions only from the two acoustic modes: the entropy/contact mode changes density without changing pressure or velocity in this linearized model. The acoustic contributions to velocity have opposite signs, while their pressure contributions have the same sign. The vertical scales differ between panels to make each quantity readable.
#
# The physical fields can therefore change shape as their constituent modes separate, even though each characteristic profile travels unchanged. These remain exact solutions of the linearized Euler equations, rather than finite-amplitude solutions of the nonlinear equations.
#
# **Domain of dependence and range of influence.** We can also follow the three characteristic families through a point $P=(x_P,t_P)$ in space–time:
#
# $$x=x_P+\lambda_j(t-t_P),\qquad
# \lambda_j\in\{\bar u-\bar c,\bar u,\bar u+\bar c\}.$$
#
# ![Space-time dependence and influence bounds through P for subsonic and supersonic linearized Euler flow.](media/euler-dependence-light.png){.light-content width=100%}
# ![Space-time dependence and influence bounds through P for subsonic and supersonic linearized Euler flow.](media/euler-dependence-dark.png){.dark-content width=100%}
#
# Time increases upward. The acoustic characteristics bound the blue domain of dependence below $P$ and the orange range of influence above it. The dashed characteristic represents the entropy/contact mode travelling with the mean flow. For the source-free constant-coefficient system, each characteristic variable depends on its own initial footpoint; the shaded regions show the enclosing bounds rather than implying that every interior point contributes to every mode.
#
# - **Subsonic flow, $0<\bar u<\bar c$.** One acoustic mode travels upstream in the fixed coordinate system, while the other acoustic mode and the entropy/contact mode travel downstream.
# - **Supersonic flow, $\bar u>\bar c$.** All three characteristic speeds are positive, so all modes travel downstream. The dependence region lies upstream of $P$, and its influence region lies downstream.
#
# Both systems remain strictly hyperbolic because their three speeds are real and distinct. Unlike the earlier steady potential-flow sketches, these diagrams describe causal propagation in time, with $t$ on the vertical axis rather than a second spatial coordinate.
#
# **Building the dependence diagram in time.** The following animation links the characteristic variables $w_1$, $w_2$, and $w_3$ to the characteristic paths carrying information toward the fixed event $P=(0.4,1)$. We use the same reference state and speeds as above, but broaden the initial Gaussian profiles to $w_{j,0}(x)=0.1\exp[-(x/1.2)^2]$ so that all three contributions arriving at $P$ are visible.
#
# <video class="light-content" controls loop playsinline preload="metadata" width="100%" aria-label="Euler characteristic variables synchronized with the construction of dependence paths leading to a fixed event P"><source src="media/euler-dependence-build-light.mp4" type="video/mp4"></video>
# <video class="dark-content" controls loop playsinline preload="metadata" width="100%" aria-label="Euler characteristic variables synchronized with the construction of dependence paths leading to a fixed event P"><source src="media/euler-dependence-build-dark.mp4" type="video/mp4"></video>
#
# On the left, each panel shows one characteristic variable, with a faint dotted curve marking its initial profile. The moving dot follows the information travelling toward $P$ along the characteristic of the same color on the right. Its height remains constant because each characteristic variable is transported unchanged along its own characteristic. The colored vertical dashed line marks the dot's current position; it does not track the pulse center. The fixed neutral dash-dotted line marks $x_P$, the common destination that all three dots reach at $t=t_P$. The blue shaded interval is the current-time slice between the outer dependence bounds shown on the right; it shrinks to $x_P$. For this constant-coefficient system, each $w_j$ receives information from its own colored dot, rather than from every point in the shaded interval.
#
# On the right, these positions build the three characteristics from their initial footpoints to $P$. The horizontal dotted line marks the current time, and the lightly shaded region lies between the outer dependence bounds. The animation stops at $t=1$, when the paths meet at $P$. We can read the value of each $w_j$ at $P$ directly from its initial footpoint:
#
# $$w_j(x_P,t_P)=w_j(x_P-\lambda_j t_P,0).$$

#
#
# ## Well-posedness: initial and boundary conditions
#
# Classifying a PDE also guides how we complete the problem with initial and boundary conditions. A problem is **well posed** if a solution exists, the solution is unique, and the solution depends continuously on the prescribed data: small changes in those data lead to small changes in the solution. These are the three requirements in Hadamard's definition of well-posedness.
#
# We consider a spatial domain $\Omega$ with boundary $\partial\Omega$, and write $\mathbf n$ for its outward unit normal. The following examples illustrate standard choices of data. The precise requirements also depend on the equation, the domain, and the regularity and compatibility of the data; the PDE type alone does not guarantee well-posedness.
#
# ### Elliptic problems: boundary data around the spatial domain
#
# As a representative elliptic equation, we consider the Poisson equation in two spatial dimensions:
#
# $$-\kappa\nabla^2\varphi=f\quad\text{in }\Omega,\qquad \kappa>0.$$
#
# Here $\varphi(x,y)$ is the unknown scalar field, $f(x,y)$ is a prescribed source, and $\kappa$ is a constant coefficient. We prescribe a suitable boundary condition around the entire boundary $\partial\Omega$. There is no time coordinate and therefore no initial condition.
#
# ![An elliptic spatial domain Omega with boundary data prescribed along its entire boundary.](media/well-posed-elliptic-light.png){.light-content width=100%}
# ![An elliptic spatial domain Omega with boundary data prescribed along its entire boundary.](media/well-posed-elliptic-dark.png){.dark-content width=100%}
#
# We can use different boundary-condition types on different portions of the boundary:
#
# | Boundary-condition type | Prescribed relation | Meaning |
# |:--|:--|:--|
# | Dirichlet | $\varphi=g_D$ | We prescribe the value of the field. |
# | Neumann | $\partial_n\varphi=\nabla\varphi\cdot\mathbf n=g_N$ | We prescribe its outward normal derivative. |
# | Robin | $\alpha\varphi+\beta\partial_n\varphi=g_R$ | We prescribe a linear combination of its value and normal derivative. |
#
# The functions $g_D$, $g_N$, and $g_R$ are boundary data, and $\alpha$ and $\beta$ are prescribed coefficients. We choose one appropriate relation at each boundary location, rather than prescribing all three independently. For a standard Robin condition, we can take $\beta>0$ and $\alpha\geq0$.
#
# A pure Neumann problem needs special care. The source and boundary data must satisfy the compatibility condition
#
# $$\int_\Omega f\,\mathrm d\Omega=-\kappa\int_{\partial\Omega}g_N\,\mathrm ds.$$
#
# Even then, adding a constant to $\varphi$ leaves both the equation and the boundary data unchanged. We therefore also fix a reference value, such as the spatial mean, to obtain a unique solution on a connected domain.
#
# ### Parabolic problems: initial data and lateral boundary data
#
# As a representative parabolic equation, we consider diffusion:
#
# $$\frac{\partial\varphi}{\partial t}=\kappa\nabla^2\varphi
# \quad\text{in }\Omega\times(0,T),\qquad \kappa>0.$$
#
# Here $\varphi(x,y,t)$ is the evolving field, $\kappa$ is a constant diffusivity, and $T$ is the final time of interest. We extend the two-dimensional spatial domain, with the same irregular outline as above, upward along the time coordinate. The bottom patch represents the initial state, and the lateral surfaces represent the spatial boundary as time advances.
#
# ![An irregular space-time extrusion for diffusion with initial data on the bottom spatial domain and boundary data on all lateral surfaces.](media/well-posed-parabolic-light.png){.light-content width=100%}
# ![An irregular space-time extrusion for diffusion with initial data on the bottom spatial domain and boundary data on all lateral surfaces.](media/well-posed-parabolic-dark.png){.dark-content width=100%}
#
# We supply the initial condition throughout the spatial domain,
#
# $$\varphi(x,y,0)=\varphi_0(x,y)\quad\text{in }\Omega,$$
#
# together with suitable Dirichlet, Neumann, or Robin boundary data on $\partial\Omega$ for $0<t<T$. For a smooth solution, the initial and boundary data must agree where they meet at $t=0$. We do not independently prescribe the final state at $t=T$; the forward evolution determines it. Unlike the stationary pure Neumann Poisson problem, diffusion with Neumann boundary data has its initial mean fixed by the initial condition.
#
# ### Hyperbolic problems: initial data and incoming characteristics
#
# As a simple hyperbolic example, we consider constant-velocity advection on the same irregular spatial domain:
#
# $$\frac{\partial\varphi}{\partial t}+\mathbf a\cdot\nabla\varphi=0,
# \qquad \mathbf a=(a,0),\quad a>0.$$
#
# Here $\varphi(x,y,t)$ is a transported scalar, and $\mathbf a$ is its prescribed transport velocity. We again prescribe the initial field throughout $\Omega$. However, information now enters through the upstream-facing portions of the curved boundary and leaves through its downstream-facing portions. We prescribe the incoming scalar value on the inflow portions; we do not independently prescribe its outgoing value on the outflow portions. The outward normal varies along the curved boundary, so we determine inflow and outflow locally.
#
# ![A hyperbolic space-time extrusion with initial data on the bottom, a shaded inflow surface with prescribed data, and a shaded outflow surface without independent boundary data.](media/well-posed-hyperbolic-light.png){.light-content width=100%}
# ![A hyperbolic space-time extrusion with initial data on the bottom, a shaded inflow surface with prescribed data, and a shaded outflow surface without independent boundary data.](media/well-posed-hyperbolic-dark.png){.dark-content width=100%}
#
# More generally, we distinguish the inflow boundary $\Gamma_{\mathrm{in}}$, where $\mathbf a\cdot\mathbf n<0$, from the outflow boundary $\Gamma_{\mathrm{out}}$, where $\mathbf a\cdot\mathbf n>0$. The scalar problem receives data of the form
#
# $$\varphi(\mathbf x,0)=\varphi_0(\mathbf x)\quad\text{in }\Omega,
# \qquad
# \varphi(\mathbf x,t)=g_{\mathrm{in}}(\mathbf x,t)
# \quad\text{on }\Gamma_{\mathrm{in}}\times(0,T).$$
#
# For a first-order hyperbolic system, we apply this principle to the **characteristic variables**. At each boundary, we examine the eigenvalues of the flux Jacobian in the outward normal direction. Negative eigenvalues identify incoming modes, for which we supply boundary information; positive eigenvalues identify outgoing modes, which are determined by the interior solution. A zero normal speed needs separate treatment because the corresponding characteristic is tangential to the boundary. The boundary relations may be stated in physical variables, but their number and form must respect the incoming modes. For the second-order acoustic wave equation introduced earlier, we instead prescribe both the initial field and its initial time derivative; the required initial data depend on the time order of the formulation.
#
# This distinction is especially useful for the Euler equations. For rightward subsonic flow in one dimension, $0<u<c$, two characteristic modes enter at the left boundary and one enters at the right boundary. Thus a physical outlet can still require boundary information: it admits an upstream-travelling acoustic mode. For rightward supersonic flow, $u>c$, all three modes enter at the left boundary and leave at the right boundary. We then supply three incoming conditions on the left and no independent conditions on the outgoing modes on the right. Counting incoming modes guides the choice of boundary data; the actual boundary relations must also be compatible and mathematically admissible.
#
# ### Application: boundary conditions for flow past an airfoil
#
# We now apply these ideas to a two-dimensional flow domain containing an airfoil. We distinguish the inlet $I$, the solid walls $W$ (including the airfoil surface), and the outlet $O$. These labels describe the physical boundaries; the governing equations determine which mathematical conditions we impose there. We consider unsteady flow with stationary walls and no backflow at the outlet.
#
# ![Airfoil in a channel with inlet I, solid walls W, and outlet O; the normals point outward from the fluid domain.](media/airfoil-boundaries-light.png){.light-content width=100%}
# ![Airfoil in a channel with inlet I, solid walls W, and outlet O; the normals point outward from the fluid domain.](media/airfoil-boundaries-dark.png){.dark-content width=100%}
#
# #### Incompressible Navier–Stokes equations
#
# Viscous momentum evolution has a parabolic character, coupled to an elliptic pressure constraint enforcing $\nabla\cdot\mathbf u=0$. A representative formulation is:
#
# - Initially, we prescribe a divergence-free velocity field compatible with the boundary data.
# - At $I$, we prescribe the inlet velocity.
# - At $W$, we impose no slip, $\mathbf u=\mathbf0$.
# - At $O$, we use a suitable open-boundary condition, such as a prescribed traction $\boldsymbol\sigma\mathbf n=-p_{\mathrm{out}}\mathbf n$, where $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$ is the total stress tensor.
#
# If we solve a pressure Poisson equation, we still need a suitable pressure boundary relation on every boundary portion. We do not prescribe the pressure value everywhere: on velocity boundaries, consistent normal pressure-gradient relations follow from the momentum equation. The outlet treatment follows the chosen pressure or traction formulation. These relations are coupled to the velocity solution and cannot be selected independently; no slip does not generally imply a zero normal pressure gradient. If the formulation leaves the pressure level undetermined, we also fix a pressure reference.
#
# #### Compressible Euler equations
#
# The time-dependent Euler equations form a hyperbolic system. Initially, we prescribe density, velocity, and pressure, with energy determined by the equation of state. At an open boundary, we supply information for the incoming characteristic modes and obtain outgoing information from the interior solution.
#
# In two spatial dimensions, the four normal characteristic speeds are $u_n-c$, $u_n$, $u_n$, and $u_n+c$, where $u_n=\mathbf u\cdot\mathbf n$ and $c$ is the local sound speed. The repeated speed $u_n$ belongs to two independent modes: an entropy/contact disturbance and a tangential-velocity (shear) disturbance. Both are convected with the fluid. Thus the two-dimensional normal system remains hyperbolic, but is not strictly hyperbolic. We obtain the following counts, which differ from the earlier three-equation, one-dimensional example:
#
# | Open-boundary regime | Number of incoming conditions |
# |:--|--:|
# | Subsonic inlet, $-c<u_n<0$ | 3 |
# | Subsonic outlet, $0<u_n<c$ | 1 |
# | Supersonic inlet, $u_n<-c$ | 4 |
# | Supersonic outlet, $u_n>c$ | 0 |
#
# For subsonic flow, total pressure, total temperature, and flow direction are a common inlet choice; an outlet pressure supplies the single incoming acoustic condition. These physical data are combined with outgoing characteristic information. At $W$, we impose impermeability, $\mathbf u\cdot\mathbf n=0$, while allowing tangential slip. We do not impose viscous no slip or a heat-conduction condition in the Euler model.
#
# #### Compressible Navier–Stokes equations
#
# The full viscous, heat-conducting system has a mixed hyperbolic–parabolic structure. We retain convection and acoustic propagation, while viscosity and heat conduction introduce diffusion. Initially, we prescribe a compatible density, velocity, and thermodynamic state.
#
# At $W$, we impose no slip, $\mathbf u=\mathbf0$, together with one thermal condition: either a wall temperature or a normal heat flux. For example, an adiabatic wall satisfies $\mathbf q_{\mathrm{heat}}\cdot\mathbf n=0$, where $\mathbf q_{\mathrm{heat}}$ is the conductive heat-flux vector. We do not independently prescribe wall density and pressure in addition to these conditions.
#
# At $I$ and $O$, we combine appropriate incoming-flow information with consistent treatment of viscous stresses and heat fluxes. The exact relations depend on the open-boundary formulation. In particular, the Euler characteristic count alone is insufficient for the viscous system: even when all convective modes leave at a supersonic outlet, viscous and thermal boundary relations still require treatment.
#
# These examples connect the scalar prototypes to coupled flow systems. Choosing compatible boundary data is necessary, but a complete well-posedness argument also depends on the equation system, its coefficients, and the regularity of the solution and data.
