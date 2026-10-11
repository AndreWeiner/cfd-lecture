# %% [markdown]
# ---
# title: Introduction to computational fluid dynamics (CFD)
# subtitle: Lecture 1
# jupyter: python3
# ---
#
# ## Learning objectives
#
# After this lecture, you should be able to:
#
# - describe the role of models, discretization, and solvers in a CFD workflow;
# - distinguish conservation laws from their numerical approximations;
# - identify the principal sources of error in a simulation.

# %% [markdown]
# ## What is actually CFD?
#
# ::: {.callout-note title="Definition"}
# **Computational fluid dynamics (CFD) combines physics, mathematics, and computer science to numerically predict, visualize, and analyze fluid motion and its interactions with the environment.**
# :::
#
# <figure class="cfd-animation">
# <video controls loop playsinline preload="metadata" aria-label="Velocity magnitude in the flow past a cylinder at Reynolds number 3900">
# <source src="../videos/plane_U_magnitude_preview.mp4" type="video/mp4">
# Your browser does not support embedded video. <a href="../videos/plane_U_magnitude_preview.mp4">Download the animation</a>.
# </video>
# <figcaption>Flow past a cylinder at Reynolds number 3900.</figcaption>
# </figure>
#
# The animation shows the normalized velocity magnitude in a plane through the flow.
# The symbols in the axes, color bar, and time label have the following meanings:
#
# - $x$ and $y$ are the streamwise and transverse coordinates; $x_c$ and $y_c$ are the coordinates of the cylinder center.
# - $D$ is the cylinder diameter. The axes $(x-x_c)/D$ and $(y-y_c)/D$ show distances from the cylinder center in units of its diameter.
# - $\mathbf{U}$ is the local velocity vector, and $|\mathbf{U}|$ is its magnitude.
# - $U_\infty$ is the free-stream velocity magnitude upstream of the cylinder. The color bar shows $|\mathbf{U}|/U_\infty$; a value of one means that the local speed equals the free-stream speed.
# - $t$ is time. The label $tU_\infty/D$ shows dimensionless time, using the characteristic flow time $D/U_\infty$.
# - The Reynolds number is $\mathrm{Re}=U_\infty D/\nu=3900$, where $\nu$ is the fluid's kinematic viscosity.
#
# ## CFD is an interdisciplinary field
#
# The three disciplines meet in the physical models, numerical methods, and
# computer implementations that make a simulation possible.

# %%
#| echo: false
#| output: false
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from figure_utils import save_theme_variants
from figure_style import DPI, WIDTH_MM, figure_style

figure_directory = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def draw_cfd_venn():
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 110 / 25.4))
        # Equal circles illustrate disciplinary overlap, not quantitative set sizes.
        for center, color in [((-0.55, 0.32), "C0"), ((0.55, 0.32), "C1"), ((0, -0.63), "C2")]:
            ax.add_patch(Circle(center, 1.05, facecolor=color, alpha=0.28, edgecolor="none"))
            ax.add_patch(Circle(center, 1.05, facecolor="none", edgecolor=color, linewidth=1.8))
        for x, y, label in [(-0.95, 0.63, "Physics"), (0.95, 0.63, "Mathematics"), (0, -1.13, "Computer science")]:
            ax.text(x, y, label, ha="center", va="center", fontsize=13)
        ax.text(0, -0.02, "CFD", ha="center", va="center", fontsize=18, fontweight="bold")
        ax.set(xlim=(-1.8, 1.8), ylim=(-1.82, 1.52), aspect="equal")
        ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
        return fig


save_theme_variants(draw_cfd_venn, figure_directory / "media/cfd-venn.png", dpi=DPI)
save_theme_variants(draw_cfd_venn, figure_directory / "media/cfd-venn.svg")

# %% [markdown]
# ![Three overlapping circles for physics, mathematics, and computer science, with CFD in their common intersection.](media/cfd-venn-light.png){.light-content width=100%}
# ![Three overlapping circles for physics, mathematics, and computer science, with CFD in their common intersection.](media/cfd-venn-dark.png){.dark-content width=100%}

# %% [markdown]
# These are selected examples; the field extends far beyond this list.
#
# <table>
# <thead><tr><th>Discipline</th><th>Examples relevant to CFD</th></tr></thead>
# <tbody>
# <tr><td>Physics</td><td><ul>
# <li>Continuum mechanics</li>
# <li>Conservation equations for mass, momentum, and energy</li>
# <li>Material laws describing fluid properties and stresses</li>
# <li>Chemistry and radiation when relevant to the application</li>
# </ul></td></tr>
# <tr><td>Mathematics</td><td><ul>
# <li>Partial differential equations expressing physical laws</li>
# <li>Numerical approximation through finite differences and finite volumes</li>
# <li>Linear algebra, especially the iterative solution of linear systems</li>
# <li>Analysis of consistency, stability, and accuracy</li>
# <li>Statistics for describing and analyzing flows</li>
# </ul></td></tr>
# <tr><td>Computer science</td><td><ul>
# <li>Algorithms that implement and solve mathematical models as efficiently as possible</li>
# <li>High-performance computing and parallel algorithms</li>
# <li>Data structures and memory management for large meshes and flow fields</li>
# </ul></td></tr>
# </tbody>
# </table>
#
# ## Why do we use CFD?
#
# Two major motivations are:
#
# 1. **CFD supports a priori design exploration and optimization.** We can predict how a proposed design behaves before building it, compare alternatives, and improve its performance.
# 2. **CFD provides detailed insight into the flow state.** It provides computed fields throughout the domain, such as velocity, pressure, and temperature. This spatial and temporal detail helps us understand mechanisms and develop simplified models.
#
# ### CFD in everyday life
#
# CFD is used in many applications and processes surrounding us, some of which may not be obvious.
#
# - **Weather forecasting:** predicting atmospheric motion together with heat, moisture, and other physical processes.
# - **Mobility:** predicting aerodynamic forces, drag, and flow separation around cars, trains, or aircraft.
# - **Renewable energy:** investigating coolant flow and heat transport in batteries, as well as wind-turbine blade aerodynamics and wakes that affect neighboring turbines.
# - **Civil engineering:** examining ventilation, indoor air distribution, the transport of airborne contaminants, and urban flows around buildings.
# - **Medicine:** investigating blood flow through vessels or medical devices.
#
# These applications involve different scales and physical effects, but share the use of numerical fluid-flow models.
#
# The following animation illustrates unsteady flow effects that can occur on passenger aircraft. Understanding these effects and, where necessary, mitigating them during the design stage is crucial for aircraft safety and reliable operation.
#
# <figure class="cfd-animation">
# <video controls loop playsinline preload="metadata" aria-label="Spectral POD mode showing shock–boundary-layer interactions around the Airbus XRF1 aircraft">
# <source src="../videos/xrf1_spod_mode_example.mp4" type="video/mp4">
# Your browser does not support embedded video. <a href="../videos/xrf1_spod_mode_example.mp4">Download the animation</a>.
# </video>
# <figcaption>The animation shows a spectral proper orthogonal decomposition (SPOD) mode of the flow around the Airbus XRF1 aircraft, highlighting shock–boundary-layer interactions.</figcaption>
# </figure>
#
# Understanding this visualization in detail is beyond the scope of this introductory lecture. The course **Machine Learning in Fluid Mechanics** explores modal analysis of fluid flows in more detail. The interested reader is referred to [Spinner and Weiner (2026)](https://doi.org/10.1016/j.ast.2026.112661).
#
# ## Main challenges of CFD
#
# Knowing the flow state in almost arbitrary detail and predicting its future behavior sounds great, right? There is a catch, though: **the computational cost required and the time available** usually limit what we can achieve. Resolving more spatial and temporal detail, including more physical processes, and exploring more scenarios can require substantial computing resources. Three closely related challenges shape the choices we make.
#
# ### Modeling errors: cost and model uncertainty
#
# A mathematical model simplifies the real application. For turbulent flows, Reynolds-averaged Navier–Stokes (RANS) models describe averaged motion and model the effects of turbulent fluctuations. They are often relatively inexpensive, but predictions depend on the turbulence model and its suitability for the flow.
#
# Scale-resolving approaches, such as large-eddy simulation (LES), resolve more turbulent motion and can reduce reliance on turbulence models at greater computational cost. They still have modeling and numerical uncertainties; greater cost alone does not guarantee a more accurate prediction.
#
# For more information about RANS, LES, and related methods, the reader is referred to the lecture on **Turbulence Modeling**.
#
# ### Discretization errors: accuracy, cost, and stability
#
# A numerical method for solving a partial differential equation (PDE) can be thought of as approximating the true solution, or function, using many small, finite-sized pieces. The size of these pieces and the way we construct the approximation affect its accuracy and computational cost.
#
# Think of approximating a curved road with short straight segments. This is simple, but the segments cut corners. More flexible curves can follow the road more accurately, yet a careless fit can overshoot or wiggle near sharp bends. Similarly, a more flexible numerical approximation can improve accuracy, but requires care to avoid unwanted oscillations or instability.
#
# ### Uncertainty in the problem description
#
# The actual boundary and initial conditions, geometry, or material properties may not be known precisely. We may also omit a physical effect whose importance we have not recognized—an “unknown unknown.” Even a correctly solved numerical model can then describe the real application poorly.
#
# [NASA's overview of uncertainty and error in CFD](https://www.grc.nasa.gov/www/wind/valid/tutorial/errors.html) discusses how modeling, numerical, and input-related limitations affect a prediction.
#
# ## CFD in five essential steps
#
# The division into five steps is a working definition for this course. It is somewhat arbitrary and could be made more fine-grained. In practice, the steps form a cycle: results may lead us to revise the problem description, model, or numerical choices.
#
# 1. **Problem description and scientific question:** Define the application and the question the simulation should answer. Identify the quantities of interest and the evidence needed to answer the question.
# 2. **Mathematical model:** Define the domain, the physical laws expressed as partial differential equations, and the material laws and assumptions. Specify appropriate initial and boundary conditions.
# 3. **Numerical model:** Discretize the domain and the equations, for example through finite differences or finite volumes. Linearize nonlinear equations where required to obtain systems that can be solved numerically.
# 4. **Iterative solution and time integration:** Solve the resulting algebraic systems, update coupled or nonlinear equations iteratively, and advance in time for an unsteady problem. Monitor whether the solution process meets the required tolerances.
# 5. **Post-processing and assessment:** Examine residuals, visualize and analyze the flow, and check the plausibility and numerical reliability of the results. Validate the prediction against measurements or other relevant reference data to assess whether the model represents the real application adequately. Small residuals alone do not establish physical accuracy.
#
# [NASA's CFD analysis process](https://www.grc.nasa.gov/WWW/wind/valid/tutorial/process.html) provides a more detailed breakdown of these activities.
#

# %% [markdown]
# ## Main equations of fluid dynamics
#
# Before we jump into our first full CFD example, we review the main equations of fluid dynamics.
#
# ### A generic transport equation
#
# Many equations share the following structure for a scalar field $\varphi$:
#
# $$
# \underbrace{\frac{\partial(\rho\varphi)}{\partial t}}_{\text{temporal change}}
# +\underbrace{\nabla\cdot(\rho\mathbf{u}\varphi)}_{\text{convection}}
# =\underbrace{\nabla\cdot(\Gamma\nabla\varphi)}_{\text{diffusion}}
# +\underbrace{S_{\varphi}}_{\text{sources}}.
# $$
#
# - $\varphi$ is the transported quantity per unit mass, and $\rho\varphi$ is its amount per unit volume.
# - $\rho$ is the fluid density, $\mathbf{u}$ is the velocity vector, and $t$ is time.
# - $\Gamma$ is a generic diffusion coefficient whose meaning and units depend on the transported quantity.
# - $S_{\varphi}$ is the net source per unit volume and time; a negative value represents a sink.
# - $\nabla\varphi$ is the spatial gradient, and $\nabla\cdot$ is the divergence operator.
#
# Temporal change describes local accumulation. Convection transports the quantity with the fluid, while diffusion transports it down its gradient. Sources describe production or removal within the fluid. This scalar equation is a useful template; coupled vector equations, such as momentum conservation, require a more general stress description.
#
# ### Conservation of mass
#
# Without mass sources, the local mass conservation equation, or continuity equation, is
#
# $$
# \frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0.
# $$
#
# For a **steady flow**, the fields do not change with time at a fixed position, so
#
# $$
# \nabla\cdot(\rho\mathbf{u})=0.
# $$
#
# Steady flow alone does not imply that density is constant or that $\nabla\cdot\mathbf{u}=0$.
#
# Another common and useful assumption is **incompressibility of the fluid**. To define incompressibility, we introduce the **material derivative**, which follows a moving fluid particle. A fluid particle is a small material parcel of fluid that we track as it moves and deforms. It contains the same fluid matter over time and is large enough to describe using continuum quantities such as density and velocity; it is not an individual molecule. The material derivative is
#
# $$
# \frac{D}{Dt}=\frac{\partial}{\partial t}+\mathbf{u}\cdot\nabla.
# $$
#
# Expanding continuity gives
#
# $$
# \frac{D\rho}{Dt}+\rho\nabla\cdot\mathbf{u}=0.
# $$
#
# An **incompressible flow** preserves the volume of each fluid particle. In the absence of mass sources, its density therefore remains constant along its motion, $D\rho/Dt=0$. For positive density, continuity then gives
#
# $$
# \boxed{\nabla\cdot\mathbf{u}=0.}
# $$
#
# This condition applies to both steady and unsteady incompressible flows. A spatially and temporally constant density is a common additional assumption; incompressibility itself does not require every fluid particle to have the same density.
#
# ### Conservation of momentum
#
# The local momentum equation in conservative vector form is
#
# $$
# \frac{\partial(\rho\mathbf{u})}{\partial t}
# +\nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u})
# =\nabla\cdot\mathbf{T}+\rho\mathbf{b}.
# $$
#
# Here, $\otimes$ denotes the outer product, $\mathbf{T}$ is the Cauchy stress tensor describing surface forces, and $\mathbf{b}$ is body force per unit mass, such as gravitational acceleration. Thus, $\rho\mathbf{b}$ is force per unit volume.
#
# We split the total stress into pressure and viscous contributions:
#
# $$
# \mathbf{T}=-p\mathbf{I}+\boldsymbol{\tau},
# \qquad
# \nabla\cdot\mathbf{T}=-\nabla p+\nabla\cdot\boldsymbol{\tau},
# $$
#
# where $p$ is pressure, $\mathbf{I}$ is the identity tensor, and $\boldsymbol{\tau}$ is the viscous stress tensor. Positive pressure produces compressive stress, which accounts for the minus sign.
#
# ### The material law for Newtonian fluids
#
# For an isotropic Newtonian fluid, viscous stress is linear in the strain-rate tensor
#
# $$
# \mathbf{S}=\frac{1}{2}\left[\nabla\mathbf{u}+(\nabla\mathbf{u})^{\mathsf{T}}\right].
# $$
#
# In three spatial dimensions, the material law is
#
# $$
# \boldsymbol{\tau}
# =2\mu\left(\mathbf{S}-\frac{1}{3}(\nabla\cdot\mathbf{u})\mathbf{I}\right)
# +\zeta(\nabla\cdot\mathbf{u})\mathbf{I}.
# $$
#
# The dynamic shear viscosity $\mu$ governs resistance to shape deformation. The bulk viscosity $\zeta$ governs the viscous response to volume expansion or compression. We retain the bulk-viscosity term rather than imposing Stokes' hypothesis, $\zeta=0$. Equivalently, the coefficient of $(\nabla\cdot\mathbf{u})\mathbf{I}$ is $\zeta-2\mu/3$; this coefficient is sometimes called the second viscosity and should not be confused with the bulk viscosity itself.
#
# ### Incompressible, constant-property momentum equation
#
# For incompressible flow, $\nabla\cdot\mathbf{u}=0$, so the volumetric viscous terms vanish. With constant density and constant dynamic viscosity, momentum conservation becomes
#
# $$
# \boxed{
# \rho\left[\frac{\partial\mathbf{u}}{\partial t}
# +(\mathbf{u}\cdot\nabla)\mathbf{u}\right]
# =-\nabla p+\mu\nabla^2\mathbf{u}+\rho\mathbf{b}.
# }
# $$
#
# The vector Laplacian $\nabla^2\mathbf{u}$ acts on each velocity component. Together with $\nabla\cdot\mathbf{u}=0$, this is the incompressible Navier–Stokes system used in this course. If viscosity varies in space, we must retain $\nabla\cdot\boldsymbol{\tau}$ instead of replacing it with $\mu\nabla^2\mathbf{u}$.
#
# There are also conservation equations for energy and chemical species. We will not examine these equations in this course.
#
# ### Conservative and non-conservative forms
#
# The **conservative form** expresses accumulation and transport as a balance of a conserved density and its flux. The **non-conservative form**, also called the advective form here, expresses change along a fluid particle's motion. For the generic scalar equation, the two forms are
#
# $$
# \begin{aligned}
# \text{Conservative:}\quad
# \frac{\partial(\rho\varphi)}{\partial t}
# +\nabla\cdot(\rho\mathbf{u}\varphi)
# &=\mathrm{RHS},\\[0.5em]
# \text{Non-conservative:}\quad
# \rho\left[\frac{\partial\varphi}{\partial t}
# +\mathbf{u}\cdot\nabla\varphi\right]
# &=\mathrm{RHS}.
# \end{aligned}
# $$
#
# Here, RHS abbreviates the right-hand side, containing diffusion and sources; it is the same in both forms.
#
# The product rule shows the connection:
#
# $$
# \frac{\partial(\rho\varphi)}{\partial t}
# +\nabla\cdot(\rho\mathbf{u}\varphi)
# =\rho\frac{D\varphi}{Dt}
# +\varphi\left[\frac{\partial\rho}{\partial t}
# +\nabla\cdot(\rho\mathbf{u})\right].
# $$
#
# The bracket vanishes by mass conservation. Thus, the two forms are equivalent for smooth fields satisfying continuity; constant density is not required for this equivalence. The same identity converts the conservative momentum equation into $\rho D\mathbf{u}/Dt=\nabla\cdot\mathbf{T}+\rho\mathbf{b}$.
#
# “Non-conservative” does not mean that the underlying physical law violates conservation. However, discretizing the two forms can yield different numerical behavior. The conservative form is particularly useful for balancing fluxes across control-volume boundaries and for treating discontinuities, where the smooth-field product-rule argument is insufficient.
#

# %% [markdown]
# ## Example — flow between a belt and a stationary wall
#
# Now it is finally time to solve our first CFD problem end-to-end. To keep things simple, we will look at the flow between a moving conveyor belt and a stationary wall.
#

# %%
#| echo: false
#| output: false
import importlib.util

channel_script = figure_directory.parent / "figures/channel-sketch/create_sketch.py"
channel_spec = importlib.util.spec_from_file_location("channel_sketch", channel_script)
channel_module = importlib.util.module_from_spec(channel_spec)
channel_spec.loader.exec_module(channel_module)
save_theme_variants(channel_module.draw, figure_directory / "media/channel-sketch.png", dpi=DPI)

# %% [markdown]
# ![Channel with a circulating belt with a rightward-moving lower run, a stationary lower wall, pressure labels, channel dimensions, and a schematic velocity profile.](media/channel-sketch-light.png){.light-content width=100%}
# ![Channel with a circulating belt with a rightward-moving lower run, a stationary lower wall, pressure labels, channel dimensions, and a schematic velocity profile.](media/channel-sketch-dark.png){.dark-content width=100%}
#
# %% [markdown]
# The sketch defines a channel of length $L$ and height $H$, with pressures $p_{\mathrm{in}}$ and $p_{\mathrm{out}}$ at its ends. The lower wall is stationary, while the lower run of the belt moves in the positive $x$ direction and its upper run returns to the left. A downstream fan lowers the outlet pressure below the inlet pressure, effectively sucking fluid through the channel. The pressure difference and the moving belt both drive flow in the positive $x$ direction. Dashed lines mark the inlet and outlet sections, and the velocity varies with the wall-normal coordinate $y$. The velocity curve is schematic. A no-slip solution must satisfy $u(0)=0$ and $u(H)=U_{\mathrm{belt}}>0$ at the lower belt run.
#
# As an idealized application, imagine air being drawn through a narrow gap beneath a conveyor belt in a ventilation or drying process. The belt drags the air along, while the fan provides additional suction. Our model isolates these two flow-driving mechanisms from the more complex geometry and heat transfer of a real installation.
#
# ### Scientific question
#
# **How does the mass flow rate through the channel change when we increase the fan speed?** In this model, we investigate how the mass flow rate changes with the pressure difference imposed by the fan. Translating this relationship into fan speed would additionally require a fan characteristic relating speed, pressure difference, and flow rate.
#
# To determine the mass flow rate for a given pressure difference, we first need the velocity profile in the channel at that same pressure difference. Integrating the local mass flux $\rho u(y)$ across the channel height gives the mass flow rate per unit belt width. This leads to our immediate question: **What does the velocity profile in the channel look like for a given pressure difference?**
#
# This example is useful for learning because suitable assumptions reduce it to a one-dimensional problem with a known analytical solution. We can compare our numerical result against that solution to assess the correctness and accuracy of our numerical method.

#
# ### Mathematical model
#
# The art of computational fluid dynamics, and more broadly of computational methods in engineering, is to reduce a seemingly complex problem to a model that is **as simple as possible, but not simpler**. We retain the effects needed to answer our scientific question and simplify the rest.
#
# For this example, we assume that the conditions in the following table are well fulfilled. In a real application, we would need to check to what extent each assumption is valid. This is usually a matter of degree rather than a binary yes-or-no decision: departures from the assumptions can reduce the accuracy of the approximation, especially when they affect the quantities we want to predict.
#
# ::: {.assumptions-table}
# | Assumption | Mathematical consequence |
# |:-----------|:-------------------------|
# | Inlet and outlet effects are negligible in the region considered (fully developed flow). | $\partial u/\partial x=0$. |
# | The belt is sufficiently wide that sidewall effects are negligible. | $\partial u/\partial z=0$. |
# | The process runs continuously under steady conditions. | $\partial u/\partial t=0$. |
# | The fluid density and dynamic viscosity are constant. | $\rho,\mu=\mathrm{const.}$ |
# | The outlet pressure is lower than the inlet pressure. | $\Delta p=p_{\mathrm{out}}-p_{\mathrm{in}}<0$ |
#
# :::
#
# These assumptions allow us to simplify the incompressible Navier–Stokes equations introduced above. We additionally consider laminar flow of a Newtonian fluid, with no spanwise motion, no body forces, and impermeable no-slip boundaries.
#
# ::: {.callout-note title="Exercise 1 — Deriving the simplified equations"}
# We will work through the derivation of these simplified equations in the [first exercise session](exercise_01.py#task-3-simplifying-the-equations-for-the-lecture-example).
# :::
#
# In the two-dimensional description, mass conservation reduces to
#
# $$
# \frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0
# \quad\Longrightarrow\quad
# \frac{\partial v}{\partial y}=0.
# $$
#
# Since the boundaries are impermeable, $v=0$ there and therefore throughout the channel. The velocity is consequently $\mathbf{u}=u(y)\mathbf{e}_x$.
#
# The remaining momentum equations are
#
# $$
# 0=-\frac{\partial p}{\partial x}+\mu\frac{d^2u}{dy^2},
# \qquad
# \frac{\partial p}{\partial y}=0.
# $$
#
# The pressure therefore depends only on $x$. The streamwise pressure gradient and the second derivative of the velocity profile depend on different coordinates, so each must be constant. With the specified inlet and outlet pressures, the simplified streamwise momentum equation becomes
#
# $$
# \mu\frac{d^2u}{dy^2}=\frac{\Delta p}{L},
# \qquad
# u(0)=0,\qquad u(H)=U_{\mathrm{belt}}.
# $$
#
# This ordinary differential equation, together with its boundary conditions, defines our simplified mathematical model. For this particular case, we can solve it analytically and use that solution to assess the numerical result.


# %% [markdown]
# ### Analytical solution
#
# Integrating the simplified ordinary differential equation twice with respect to $y$ and applying the boundary conditions gives the **Couette–Poiseuille solution**:
#
# $$
# u(y)=U_{\mathrm{belt}}\frac{y}{H}
# -\frac{\Delta p}{2\mu L}\,y(H-y),\qquad 0\leq y\leq H,
# \qquad \Delta p=p_{\mathrm{out}}-p_{\mathrm{in}}.
# $$
#
# The first term is the belt-driven contribution; the second is the
# pressure-driven contribution. With $\Delta p<0$, pressure reinforces the
# positive belt motion. The pressure varies linearly:
#
# $$
# p(x)=p_{\mathrm{in}}+\frac{\Delta p}{L}x.
# $$
#
# The figure uses dimensionless coordinates $\eta=y/H$ and $u/U_{\mathrm{belt}}$,
# with an illustrative pressure parameter
# $\Pi=\Delta p H^2/(2\mu L U_{\mathrm{belt}})=-2$:
#
# $$
# \frac{u}{U_{\mathrm{belt}}}=\eta-\Pi\eta(1-\eta).
# $$
#
# This example has positive flow throughout the channel interior. These curves
# are exact for the stated model and parameter, rather than measured data.

# %%
#| echo: false
#| output: false
eta = np.linspace(0.0, 1.0, 401)
pressure_parameter = -2.0
belt_contribution = eta
pressure_contribution = -pressure_parameter * eta * (1.0-eta)
analytical_velocity = belt_contribution + pressure_contribution
# Verify no slip and the dimensionless momentum equation u'' = 2 Pi.
assert np.allclose(analytical_velocity[[0, -1]], [0.0, 1.0])
assert np.allclose(np.diff(analytical_velocity, n=2)/(eta[1]-eta[0])**2,
                   2*pressure_parameter, atol=1e-9)


def draw_analytical_solution():
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"]/25.4, 105/25.4),
                               layout="constrained")
        ax.plot(analytical_velocity, eta, color="C0", linewidth=2,
                label="Combined analytical solution")
        ax.plot(belt_contribution, eta, color="C1", linestyle="--",
                label="Belt contribution")
        ax.plot(pressure_contribution, eta, color="C2", linestyle="-.",
                label="Pressure contribution")
        ax.axvline(0, color=plt.rcParams["text.color"], linewidth=.8, alpha=.5)
        ax.set(xlabel=r"$u/U_{\mathrm{belt}}$",
               ylabel=r"$y/H$", ylim=(0,1), xlim=(-.05,1.2))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02),
                  frameon=False, fontsize=10)
        return fig


for solution_format in ["png", "svg"]:
    save_theme_variants(draw_analytical_solution,
                        figure_directory / f"media/belt-analytical-solution.{solution_format}", dpi=DPI)

# %% [markdown]
# ![Exact dimensionless Couette–Poiseuille velocity profile for pressure parameter minus two, together with belt and pressure contributions; pressure reinforces the belt motion, with zero velocity at the lower wall and the belt speed at the upper wall.](media/belt-analytical-solution-light.png){.light-content width=100%}
# ![Exact dimensionless Couette–Poiseuille velocity profile for pressure parameter minus two, together with belt and pressure contributions; pressure reinforces the belt motion, with zero velocity at the lower wall and the belt speed at the upper wall.](media/belt-analytical-solution-dark.png){.dark-content width=100%}

# %% [markdown]
# ### Domain discretization or meshing
#
# As discussed above, the problem can be simplified to an ordinary differential equation. Our domain is therefore one-dimensional and extends from $y=0$ at the stationary bottom wall to $y=H$ at the lower run of the conveyor belt.
#

# %%
#| echo: false
#| output: false

def draw_1d_mesh():
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 105 / 25.4))
        ink = plt.rcParams["text.color"]
        # Context: the 1D mesh spans the wall-normal height of the channel.
        ax.plot([-2.5, -1.05], [0, 0], color=ink, linewidth=1.8)
        ax.plot([-2.5, -1.05], [1, 1], color=ink, linewidth=1.8)
        ax.plot([-2.5, -1.05], [1.12, 1.12], color=ink, linewidth=1.4)
        # Wavy cut edges indicate that the channel continues beyond this view.
        for cut_x in [-2.5, -1.05]:
            belt_y = np.linspace(.98, 1.14, 121)
            belt_x = cut_x + .025 * np.sin(2 * np.pi * (belt_y-1.0)/.12)
            ax.plot(belt_x, belt_y, color=ink, linewidth=1.2, solid_capstyle="butt")
            wall_y = np.linspace(-.10, .02, 121)
            wall_x = cut_x + .025 * np.sin(2 * np.pi * (wall_y+.08)/.08)
            ax.plot(wall_x, wall_y, color=ink, linewidth=1.2, solid_capstyle="butt")
        for x in np.linspace(-2.40, -1.15, 12):
            ax.plot([x, x-.07], [0, -.06], color=ink, linewidth=.7)
        mesh_x = -1.65
        ax.plot([mesh_x, mesh_x], [0, 1], color="C0", linewidth=2,
                linestyle="--")
        ax.annotate("", xy=(-2.7, 1), xytext=(-2.7, 0),
                    arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1})
        ax.text(-2.84, .5, r"$H$", va="center", fontsize=13)
        for height in [0, 1]:
            ax.plot([-2.82, -2.58], [height, height], color=ink, linewidth=.8)
        # Guide lines map the two physical walls to the mesh endpoints.
        for height in [0, 1]:
            ax.plot([mesh_x, 0], [height, height], color="C0", alpha=.6,
                    linewidth=1.2, linestyle="--", zorder=0)
        nodes = np.linspace(0, 1, 6)
        ax.plot(np.zeros_like(nodes), nodes, color="C0", linewidth=2)
        ax.scatter(np.zeros_like(nodes), nodes, color="C0", s=38, zorder=3)
        labels = [r"$y_0=0$", r"$y_1$", r"$y_2$", r"$\vdots$", r"$y_{N-2}$", r"$y_{N-1}=H$"]
        for y, label in zip(nodes, labels):
            ax.text(.13, y, label, va="center", fontsize=13)
        ax.annotate("", xy=(-2.35, .36), xytext=(-2.35, 0),
                    arrowprops={"arrowstyle": "->", "color": ink, "lw": 1})
        ax.text(-2.35, .41, r"$y$", ha="center", fontsize=12)
        for height in [nodes[1], nodes[2]]:
            ax.plot([-.51, 0], [height, height], color=ink,
                    linewidth=1, linestyle="--")
        ax.annotate("", xy=(-.43, nodes[2]), xytext=(-.43, nodes[1]),
                    arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1})
        ax.text(-.55, (nodes[1]+nodes[2])/2, r"$\Delta y$",
                ha="right", va="center", fontsize=12)
        ax.set(xlim=(-3.05, 1.45), ylim=(-.3, 1.45))
        ax.axis("off")
        fig.subplots_adjust(left=.04, right=.96, top=.96, bottom=.04)
        return fig


for mesh_format in ["png", "svg"]:
    save_theme_variants(draw_1d_mesh, figure_directory / f"media/mesh-1d.{mesh_format}", dpi=DPI)

# %% [markdown]
# ![A vertical uniform one-dimensional mesh, with nodes numbered from y zero at zero to y capital N minus one at H and spacing delta y.](media/mesh-1d-light.png){.light-content width=100%}
# ![A vertical uniform one-dimensional mesh, with nodes numbered from y zero at zero to y capital N minus one at H and spacing delta y.](media/mesh-1d-dark.png){.dark-content width=100%}

# %% [markdown]
# **Domain discretization**, also called **meshing**, divides this interval into finite-sized segments. We choose segments of equal size and define their endpoints using $N$ nodes, giving $N-1$ segments. The node index is $i=0,1,\ldots,N-1$, with $y_0=0$ and $y_{N-1}=H$.
#
# For this uniform mesh, the spacing and node positions are
#
# $$
# \Delta y=\frac{H}{N-1},\qquad y_i=i\Delta y.
# $$
#
# Our numerical solution will provide a velocity value at every node. We denote this value by $u_i$, which approximates the true solution evaluated at the node position:
#
# $$
# u_i\approx u(y_i).
# $$

# %% [markdown]
# ### Equation discretization
#
# In the previous section, we subdivided the domain into finite segments. We now use these segments to approximate the derivatives in our ordinary differential equation. We need the second derivative $d^2u/dy^2$; we will construct its approximation by first approximating the first derivative on each neighboring segment.
#
# The sketch illustrates one choice: approximate the true solution, which is usually unknown, by piecewise-linear segments joining the nodal values $u_{i-1}$, $u_i$, and $u_{i+1}$. The dashed curve represents a possible smooth solution, not a known reference solution. This is a generic illustration of discretization; it does not represent the belt problem's analytical profile. Simple geometry gives the slope of each straight segment as the change in velocity divided by its length.

# %%
#| echo: false
#| output: false

def draw_discretization():
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 100 / 25.4))
        ink = plt.rcParams["text.color"]
        sample_y = np.array([.9, 1.8, 2.7])
        sample_u = np.array([.65, 1.0, 2.4])
        # A possible smooth behavior, not a known analytical solution.
        illustrative_y = np.linspace(.45, 2.96, 250)
        offset = illustrative_y - sample_y[1]
        illustrative_u = 1.0 + (1.75/1.8)*offset + (1.05/1.62)*offset**2
        ax.plot(illustrative_y, illustrative_u, color="C1", linewidth=1.8,
                linestyle="--", label=r"Unknown exact $u(y)$ (illustrative)")
        ax.plot(sample_y, sample_u, color="C0", linewidth=1.8,
                label="Piecewise-linear approximation")
        # Dotted continuations preserve each outer segment's slope.
        left_y, right_y = .25, 3.04
        left_u = sample_u[0] + (left_y-sample_y[0]) * (sample_u[1]-sample_u[0]) / (sample_y[1]-sample_y[0])
        right_u = sample_u[2] + (right_y-sample_y[2]) * (sample_u[2]-sample_u[1]) / (sample_y[2]-sample_y[1])
        ax.plot([left_y, sample_y[0]], [left_u, sample_u[0]],
                color="C0", linewidth=1.8, linestyle=":")
        ax.plot([sample_y[2], right_y], [sample_u[2], right_u],
                color="C0", linewidth=1.8, linestyle=":")
        for y, u in zip(sample_y, sample_u):
            ax.plot([y, y], [0, u], color=ink, linestyle="--", linewidth=.9, alpha=.65)
            ax.plot([0, y], [u, u], color=ink, linestyle="--", linewidth=.9, alpha=.65)
        ax.scatter(sample_y, sample_u, color="C0", s=45, zorder=4)
        ax.set_xticks([.9, 1.35, 1.8, 2.25, 2.7], [r"$y_{i-1}$", r"$y_{i-1/2}$", r"$y_i$", r"$y_{i+1/2}$", r"$y_{i+1}$"])
        ax.set_yticks(sample_u, [r"$u_{i-1}$", r"$u_i$", r"$u_{i+1}$"])
        ax.set(xlim=(0, 3.4), ylim=(0, 3.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=9, labelsize=13)
        ax.annotate("", xy=(3.35, 0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.2})
        ax.annotate("", xy=(0, 3.0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.2})
        ax.text(3.38, -.06, r"$y$", ha="center", va="top", fontsize=14)
        ax.text(-.06, 3.03, r"$u$", ha="right", va="center", fontsize=14)
        ax.legend(loc="upper left", frameon=False, fontsize=11,
                  handlelength=2.5, borderaxespad=1.2)
        midpoint_y = .5*(sample_y[:-1]+sample_y[1:])
        for midpoint, value in zip(midpoint_y, .5*(sample_u[:-1]+sample_u[1:])):
            ax.plot([midpoint, midpoint], [0, value], color="C2", linestyle=":", linewidth=1.1)
        fig.subplots_adjust(left=.15, right=.95, bottom=.16, top=.91)
        return fig


for discretization_format in ["png", "svg"]:
    save_theme_variants(draw_discretization,
                        figure_directory / f"media/equation-discretization.{discretization_format}", dpi=DPI)

# %% [markdown]
# ![Velocity u versus y, with three neighboring nodes i minus one, i, and i plus one joined by two straight segments with different slopes and dotted outer extensions; dashed guides connect each node to its coordinate and velocity on the axes.](media/equation-discretization-light.png){.light-content width=100%}
# ![Velocity u versus y, with three neighboring nodes i minus one, i, and i plus one joined by two straight segments with different slopes and dotted outer extensions; dashed guides connect each node to its coordinate and velocity on the axes.](media/equation-discretization-dark.png){.dark-content width=100%}

#
# We define the slope of each segment at the segment's midpoint,
# $y_{i-1/2}=(y_{i-1}+y_i)/2$ and $y_{i+1/2}=(y_i+y_{i+1})/2$:
#
# $$
# g_{i-1/2}\equiv\frac{u_i-u_{i-1}}{\Delta y}
# \approx\left(\frac{du}{dy}\right)_{i-1/2},
# \qquad
# g_{i+1/2}\equiv\frac{u_{i+1}-u_i}{\Delta y}
# \approx\left(\frac{du}{dy}\right)_{i+1/2}.
# $$
#
# On our uniform mesh, the distance between the two segment midpoints is again $\Delta y$, since $y_{i+1/2}-y_{i-1/2}=\Delta y$. Dividing the change between the two slope estimates by this distance approximates the second derivative at $y_i$:
#
# $$
# \left(\frac{d^2u}{dy^2}\right)_i
# \approx\frac{g_{i+1/2}-g_{i-1/2}}{\Delta y}
# =\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta y^2}.
# $$
#
# This construction is motivated by the calculus definition of a derivative:
#
# $$
# \frac{du}{dy}(y)=\lim_{h\to0}\frac{u(y+h)-u(y)}{h}.
# $$
#
# A finite-difference approximation retains a finite spacing instead of taking the limit. Applying the same idea to the first derivative gives an approximation of the second derivative. Here we compare slopes on neighboring segments rather than differentiating the piecewise-linear curve twice directly: that curve has zero second derivative inside each segment and a slope jump at the middle node.
#
# For a sufficiently smooth solution, reducing the segment size improves these derivative approximations. With a consistent, stable numerical method and sufficiently small solution errors, refining the mesh can also bring the computed solution closer to the true solution of our mathematical model. We will learn the precise meanings of consistency and stability later in the course.
#
# Substituting the second-derivative approximation into the simplified momentum equation gives an algebraic equation at each interior node $i=1,\ldots,N-2$:
#
# $$
# \mu\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta y^2}=\frac{\Delta p}{L}.
# $$
#
# The boundary values are $u_0=0$ and $u_{N-1}=U_{\mathrm{belt}}$.
#

# %% [markdown]
# ### Constructing the linear system of equations
#
# We now evaluate the discretized momentum equation at every interior node, $i=1,\ldots,N-2$. Each equation relates the velocity at one node to the velocities at its two neighbors, so we must solve the equations together as a linear system.
#
# Multiplying the node equation by $\Delta y^2/\mu$ and then by $-1$ gives
#
# $$
# -u_{i-1}+2u_i-u_{i+1}=f,
# \qquad f=-\frac{\Delta p\,\Delta y^2}{\mu L}.
# $$
#
# Multiplication by $-1$ does not change the solution and is not required for solvability. It makes the coefficient of $u_i$ positive. With the prescribed boundary values in this problem, the resulting matrix is symmetric positive definite, which is useful for methods such as conjugate gradients. We will discuss linear-system solvers later in the course.
#
# The boundary velocities $u_0=0$ and $u_{N-1}=U_{\mathrm{belt}}$ are known. Moving them to the right-hand side leaves $N-2$ unknown interior velocities. For at least two interior nodes, the first and last equations are
#
# $$
# 2u_1-u_2=f+u_0=f,
# \qquad
# -u_{N-3}+2u_{N-2}=f+u_{N-1}=f+U_{\mathrm{belt}}.
# $$
#
# Collecting all interior equations gives $\mathbf{A}\mathbf{u}_{\mathrm{num}}=\mathbf{b}$:
#
# - $\mathbf{u}_{\mathrm{num}}=[u_1,u_2,\ldots,u_{N-2}]^{\mathsf{T}}\in\mathbb{R}^{N-2}$ is the **numerical solution vector**. It is a column vector of size $(N-2)\times1$ containing only the unknown interior velocities; the two prescribed boundary values are excluded.
# - $\mathbf{A}\in\mathbb{R}^{(N-2)\times(N-2)}$ is the **coefficient matrix**, also called the system matrix. Row $i$ represents the equation at interior node $i$, and column $k$ contains the coefficients multiplying $u_k$.
# - $\mathbf{b}\in\mathbb{R}^{N-2}$ is the **right-hand-side vector**, of size $(N-2)\times1$. It contains the pressure-driving term and the known boundary contributions.
#
# The row labels outside the matrix identify the interior-node indices:
#
# $$
# \begin{array}{c}1\\2\\3\\\vdots\\N-2\end{array}\;
# \underbrace{\begin{bmatrix}
# 2 & -1 & 0 & \cdots & 0\\
# -1 & 2 & -1 & \ddots & \vdots\\
# 0 & -1 & 2 & \ddots & 0\\
# \vdots & \ddots & \ddots & \ddots & -1\\
# 0 & \cdots & 0 & -1 & 2
# \end{bmatrix}}_{\mathbf{A}}
# \underbrace{\begin{bmatrix}
# u_1\\u_2\\u_3\\\vdots\\u_{N-2}
# \end{bmatrix}}_{\mathbf{u}_{\mathrm{num}}}
# =
# \underbrace{\begin{bmatrix}
# f\\f\\\vdots\\f\\f+U_{\mathrm{belt}}
# \end{bmatrix}}_{\mathbf{b}}.
# $$
#
# The matrix has nonzero entries only on its main diagonal and the two adjacent diagonals; it is **tridiagonal**. Its size is $(N-2)\times(N-2)$, so refining the mesh increases the number of coupled equations. If there is only one interior node, both boundary contributions enter the same equation: $2u_1=f+u_0+u_{N-1}$.
#
# Solving this system provides the interior velocities. Together with the prescribed boundary values, they form our numerical velocity profile.
#
# ::: {.callout-note title="Exercise 1 — Implementing and assessing the solution"}
# In the [first exercise session](exercise_01.py#step-6-implement-and-assess-the-belt-flow-solution), we will implement the numerical solution, compare it with the analytical solution, visualize the results, and assess the accuracy of the numerical solution. We will also integrate the velocity profile to determine the mass flow rate per unit belt width and investigate its dependence on the pressure difference.
# :::
