# %% [markdown]
# ---
# title: Introduction to computational fluid dynamics
# subtitle: Exercise 1 — theory and code lab
# jupyter: python3
# ---
#
# ## Part 1 — Theory
#
# ### Task 1 — Differential operators in Cartesian coordinates
#
# Consider a scalar field $a(x,y)$ and a vector field
#
# $$
# \mathbf{a}(x,y)=a_x(x,y)\mathbf{e}_x+a_y(x,y)\mathbf{e}_y,
# \qquad
# \nabla=\mathbf{e}_x\frac{\partial}{\partial x}
# +\mathbf{e}_y\frac{\partial}{\partial y}.
# $$
#
# The vectors $\mathbf{e}_x$ and $\mathbf{e}_y$ are the Cartesian unit vectors. The subscripts in $a_x$ and $a_y$ identify vector components, not derivatives. Let $\Gamma$ be a constant scalar coefficient.
#
# Write the following expressions explicitly in Cartesian components, using partial derivatives where appropriate. For the vector gradient, use the convention $(\nabla\mathbf{a})_{ij}=\partial a_i/\partial x_j$, where $(x_1,x_2)=(x,y)$:
#
# 1. The gradient of the scalar field, $\nabla a$.
# 2. The divergence of the vector field, $\nabla\cdot\mathbf{a}$.
# 3. The gradient of the vector field, $\nabla\mathbf{a}$, written as a $2\times2$ matrix.
# 4. The outer product $\mathbf{a}\otimes\mathbf{a}$, written as a $2\times2$ matrix. Use $(\mathbf{a}\otimes\mathbf{a})_{ij}=a_i a_j$.
# 5. The coefficient-weighted diffusion operator in divergence form, $\nabla\cdot(\Gamma\nabla\mathbf{a})$. Apply the divergence to each row of the vector-gradient matrix. For constant $\Gamma$, this equals $\Gamma\nabla^2\mathbf{a}$.
#
# For each expression, state whether the result is a scalar, a vector, or a second-order tensor.
#
# #### Identities and product rules
#
# Prove the following identities by expanding their Cartesian components. Let $\rho(x,y,t)$ and $\varphi(x,y,t)$ be smooth scalar fields, and let $\mathbf{u}(x,y,t)$ be a smooth velocity field.
#
# 1. Show that, for constant $\Gamma$,
#
# $$
# \nabla\cdot(\Gamma\nabla\varphi)=\Gamma\nabla^2\varphi.
# $$
#
# Now allow $\Gamma$ to vary in space and explain which additional term appears.
#
# 2. Establish the spatial product rule
#
# $$
# \nabla\cdot(\rho\mathbf{u})
# =\mathbf{u}\cdot\nabla\rho+\rho\nabla\cdot\mathbf{u}.
# $$
#
# Use it to rewrite continuity as $D\rho/Dt+\rho\nabla\cdot\mathbf{u}=0$, where $D/Dt=\partial/\partial t+\mathbf{u}\cdot\nabla$. Explain why $D\rho/Dt=0$ implies $\nabla\cdot\mathbf{u}=0$ for positive density, without assuming steady flow.
#
# 3. Prove the scalar transport identity
#
# $$
# \frac{\partial(\rho\varphi)}{\partial t}
# +\nabla\cdot(\rho\mathbf{u}\varphi)
# =\rho\frac{D\varphi}{Dt}
# +\varphi\left[\frac{\partial\rho}{\partial t}
# +\nabla\cdot(\rho\mathbf{u})\right].
# $$
#
# Use continuity to obtain the non-conservative form. Then specialize it to incompressible flow with constant density.
#
# 4. Prove the corresponding vector identity for momentum:
#
# $$
# \frac{\partial(\rho\mathbf{u})}{\partial t}
# +\nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u})
# =\rho\left[\frac{\partial\mathbf{u}}{\partial t}
# +(\mathbf{u}\cdot\nabla)\mathbf{u}\right]
# +\mathbf{u}\left[\frac{\partial\rho}{\partial t}
# +\nabla\cdot(\rho\mathbf{u})\right].
# $$
#
# Apply the tensor divergence row by row, as above. Use continuity to show that the conservative momentum equation and its non-conservative form are equivalent for smooth fields. Finally, show that for incompressible flow with constant density,
#
# $$
# \nabla\cdot(\mathbf{u}\otimes\mathbf{u})
# =(\mathbf{u}\cdot\nabla)\mathbf{u}.
# $$
#
# ### Task 2 — From vector notation to Cartesian components
#
# Consider an incompressible Newtonian fluid with constant density $\rho$ and constant dynamic viscosity $\mu$. Its mass and momentum conservation equations are
#
# $$
# \nabla \cdot \mathbf{u} = 0,
# $$
#
# $$
# \rho\left(\frac{\partial \mathbf{u}}{\partial t}
# + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
# = -\nabla p + \mu\nabla^2\mathbf{u} + \rho\mathbf{b}.
# $$
#
# Here, $\mathbf{u}$ is velocity, $p$ is pressure, and $\mathbf{b}$ is body force per unit mass (for example, gravitational acceleration). The operator $\nabla^2$ is the vector Laplacian, applied to each velocity component.
#
# For a two-dimensional Cartesian flow, use
#
# $$
# \mathbf{u}(x,y,t) = u(x,y,t)\mathbf{e}_x + v(x,y,t)\mathbf{e}_y,
# \qquad
# \nabla = \mathbf{e}_x\frac{\partial}{\partial x}
# + \mathbf{e}_y\frac{\partial}{\partial y},
# \qquad
# \mathbf{b} = b_x\mathbf{e}_x + b_y\mathbf{e}_y.
# $$
#
# Assume there is no out-of-plane velocity and no variation in the out-of-plane direction.
#
# 1. Expand the mass conservation equation into derivatives of $u$ and $v$.
# 2. Expand each term of the momentum equation separately: the time derivative, convective acceleration, pressure gradient, viscous term, and body force.
# 3. Write the complete scalar momentum equations in the $x$ and $y$ directions.
# 4. Label the physical meaning of each term and check that every term in the momentum equations has the same units.
#
# Keep the time derivatives and body forces; do not assume steady flow or neglect any terms yet.
#
# ### Task 3 — Simplifying the equations for the lecture example
#
# Refer to the [belt-and-wall problem in Lecture 1](lecture_01.py#example-flow-between-a-belt-and-a-stationary-wall) for the full problem description and the assumptions stated in its mathematical model.
#
# Starting from the two-dimensional mass and momentum equations you obtained in Task 2, apply these assumptions to derive the simplified equations for this problem.
#
# 1. State which derivatives, velocity components, and force terms vanish, and explain which assumption justifies each simplification.
# 2. Use mass conservation and the impermeability of the boundaries to determine the transverse velocity.
# 3. Simplify both momentum components and determine the spatial dependence of pressure and streamwise velocity.
# 4. Use the prescribed inlet and outlet pressures to express the streamwise pressure gradient in terms of $\Delta p$ and $L$.
# 5. State the resulting ordinary differential equation and its boundary conditions. Check the sign of the pressure difference against the flow-driving mechanism described in the lecture.
#
# ## Part 2 — Code lab
#
# ### Step 1 — Install Python
#
# Use **Python 3.10 or newer**. The Windows instructions below use Python 3.14 as an example; on Ubuntu, use the distribution's supported Python version. You will need an internet connection for the downloads and package installation.
#
# #### Windows 11
#
# 1. Download the **Python install manager** from the [official Python website](https://www.python.org/downloads/windows/) and run its installer.
# 2. Open a new **Command Prompt** window (search for `cmd` in the Start menu). The Windows commands on this page use Command Prompt, not PowerShell.
# 3. Install Python 3.14 and check its version:
#
# ```bat
# py install 3.14
# py -3.14 --version
# ```
#
# If `py install` is not recognized, you may have the older Python launcher. Install Python 3.14 using its standard Windows installer from the same official website, then run `py -3.14 --version`.
#
# See the [Python Windows guide](https://docs.python.org/3/using/windows.html) for installation troubleshooting.
#
# #### Ubuntu Linux
#
# On **Ubuntu 26.04 LTS**, open a terminal and install Python with virtual-environment support:
#
# ```bash
# sudo apt update
# sudo apt install python3-full python3-pip
# python3 --version
# ```
#
# Check that the reported version is at least 3.10. These commands also apply to Ubuntu 24.04 LTS. Keep Ubuntu's system Python; install course packages in the virtual environment below. See the [Ubuntu Python setup guide](https://ubuntu.com/developers/docs/howto/python-setup/).
#
# ### Step 2 — Create the course environment
#
# Clone the official course repository using Git, then create a virtual environment called `.venv-cfd` and a folder called `exercises` inside your checkout. The environment isolates the course packages, while `exercises` holds your own scripts and results. Both folders are excluded from Git tracking by the repository's `.gitignore` rules.
#
# Install Git first if it is not already available; on Windows, use the [Git for Windows installer](https://gitforwindows.org/). The commands below clone the official GitHub repository for this course.
#
# #### Windows 11 — Command Prompt
#
# ```bat
# git clone https://github.com/AndreWeiner/cfd-lecture.git cfd-lecture
# cd cfd-lecture
# py -3.14 -m venv .venv-cfd
# .venv-cfd\Scripts\activate.bat
# mkdir exercises
# python --version
# ```
#
# #### Ubuntu Linux — terminal
#
# ```bash
# sudo apt install git
# git clone https://github.com/AndreWeiner/cfd-lecture.git cfd-lecture
# cd cfd-lecture
# python3 -m venv .venv-cfd
# source .venv-cfd/bin/activate
# mkdir -p exercises
# python --version
# ```
#
# Run the remaining installation commands from the repository root with the environment active. Whenever you open a new terminal, return to `cfd-lecture` and repeat the activation command for your operating system. Use `deactivate` to leave the environment.
#
# ### Step 3 — Install NumPy and Matplotlib
#
# The cloned repository already contains `requirements.txt` in its root folder. It lists the two course dependencies:
#
# ```text
# numpy
# matplotlib
# ```
#
# On either operating system, install them into the active environment:
#
# ```bash
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt
# ```
#
# This is the student environment. The separate `requirements-quarto.txt` file is for people rendering the course website.
#
# #### Recommended code editor — VS Code
#
# We recommend **Visual Studio Code (VS Code)** for editing and running your Python scripts.
#
# 1. Download and install VS Code from the [official website](https://code.visualstudio.com/Download). Choose the Windows installer on Windows 11 or the `.deb` package on Ubuntu.
# 2. Install the **Python** extension published by Microsoft from the Extensions view.
# 3. Open your cloned `cfd-lecture` folder using **File → Open Folder**.
# 4. Open the Command Palette with **Ctrl+Shift+P**, run **Python: Select Interpreter**, and select the interpreter in `.venv-cfd`. If it is not listed, choose **Enter interpreter path** and browse to `.venv-cfd\Scripts\python.exe` on Windows or `.venv-cfd/bin/python` on Linux.
#
# Save your own scripts in `exercises`. When running commands in VS Code's integrated terminal, activate `.venv-cfd` as described above. On Windows, select a **Command Prompt** terminal to use the commands shown in this exercise. See the [VS Code Python environment guide](https://code.visualstudio.com/docs/python/environments) for further help.
#
# ### Step 4 — Check your setup
#
# Save the following code as `check_setup.py` in the `exercises` folder using VS Code or another code editor. On Windows, ensure the file is not named `check_setup.py.txt`.
#
# ```python
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
#
# assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"
# print("Python:", sys.version.split()[0])
# print("Interpreter:", sys.executable)
# print("NumPy:", np.__version__)
#
# x = np.linspace(0.0, 1.0, 101)
# plt.plot(x, x**2)
# plt.xlabel("x")
# plt.ylabel("x²")
# plt.tight_layout()
# plt.savefig("setup_check.png")
# plt.show()
# ```
#
# From the repository root, run it with the environment activated:
#
# ```bash
# cd exercises
# python check_setup.py
# ```
#
# The interpreter path should point inside `.venv-cfd`. A plot window should appear, and `setup_check.png` should be saved in your working folder. If no window appears, open the saved image to check the result.
#
# ### Step 5 — Optional: OpenCode with free models or TUD:AI
#
# This step is **optional**. You can complete the code lab using Python, NumPy, and Matplotlib alone.
#
# Before using a coding agent, read [Topic 0 — How to learn with generative AI](lecture_00.qmd) for guidance on using AI tools while developing your own understanding and checking their output.
#
# #### Install OpenCode Desktop
#
# - **Windows 11:** On the [official OpenCode download page](https://opencode.ai/download), select **OpenCode Desktop → Windows (x64)**. Run the downloaded installer and start OpenCode from the Start menu.
# - **Ubuntu Linux:** On the same page, download the **Linux (.deb)** package. Open a terminal in the download folder and install that file with `sudo apt install ./NAME.deb`, replacing `NAME.deb` with its actual filename. Start OpenCode from the applications menu.
#
# Open your `cfd-lecture` repository folder as the project in OpenCode. Keep your own exercise scripts and generated results in `exercises`.
#
# #### Recommended: start with a free OpenCode model
#
# Use one of the free models available in OpenCode to get started. This route does not require access to TUD:AI. As of October 2026, our recommended starting point is **GLM-5.3-Flash Free** (GLM 5.3 Flash), or a newer version of the **GLM Flash** model if offered for free. Choose the entry explicitly marked **Free**; similarly named entries may be paid. See the [OpenCode model list](https://opencode.ai/docs/zen/).
#
# - **Desktop app:** Start a session in your course project. Click the model name in the prompt area to open the model picker, search for `GLM`, and select **GLM-5.3-Flash Free**. If necessary, open the model-management settings from the picker and enable the free model so that it appears in the list. Check the selected model name before sending your first prompt.
# - **Command line:** If you use the [OpenCode CLI](https://opencode.ai/docs/cli/), open a terminal in your course repository and start `opencode`. Enter `/models`, search for `GLM`, select the free entry, and confirm with Enter. You can reopen `/models` whenever you want to switch models. See the [model-selection documentation](https://opencode.ai/docs/models/).
#
# Free offerings and usage limits can change. If the recommended model is missing or unavailable, update OpenCode and check the picker for a newer free GLM Flash version or another model explicitly marked **Free**. You can continue the exercise without a coding agent if no free model is available. The free-model route does not require the `opencode.json` configuration or TU Dresden API key below.
#
# #### Alternative: TUD:AI (if you have access)
#
# Participants with TUD:AI API access can use the TU Dresden service instead. Complete the following key and provider setup only if you choose this alternative.
#
# ##### Create a TU Dresden API key
#
# 1. Open the [TUD:AI LLM service in the TU Dresden Self-Service Portal](https://selfservice.tu-dresden.de/services/scads-llm-api/).
# 2. Sign in with your TU Dresden account and select **Manage API access** (German: **API-Zugang verwalten**).
# 3. Follow the portal's prompts to create a key and accept the service's usage terms.
# 4. Keep the key private. Enter it only in the agent's provider connection; do not put it in exercise scripts, shared screenshots, or a Git repository.
#
# The public service description currently lists **employees** as the eligible group, while the TUD:AI documentation describes access for TU Dresden affiliates more broadly. If API access is unavailable for your student account, use the free-model route above. Ask the lecturer or tutor if you specifically need course access to TUD:AI. See the [TUD:AI documentation](https://llm.scads.ai/docs/) for service details.
#
# ##### Connect OpenCode to TUD:AI
#
# Create `opencode.json` in the repository root. This local configuration is excluded from Git tracking. It sets **GLM 5.3 Flash** as the default and includes GLM 5.3 and Qwen3.8 27B as alternatives for course work.
#
# ```json
# {
#   "$schema": "https://opencode.ai/config.json",
#   "model": "scads/zai-org/GLM-5.3-Flash",
#   "provider": {
#     "scads": {
#       "name": "TUD:AI",
#       "npm": "@ai-sdk/openai-compatible",
#       "options": {
#         "baseURL": "https://llm.scads.ai/v1"
#       },
#       "models": {
#         "zai-org/GLM-5.3-Flash": {
#           "name": "GLM 5.3 Flash",
#           "limit": {
#             "context": 1048576,
#             "output": 32768
#           }
#         },
#         "zai-org/GLM-5.3": {
#           "name": "GLM 5.3",
#           "limit": {
#             "context": 524288,
#             "output": 32768
#           }
#         },
#         "Qwen/Qwen3.8-27B": {
#           "name": "Qwen3.8 27B",
#           "limit": {
#             "context": 262144,
#             "output": 32768
#           }
#         }
#       }
#     }
#   }
# }
# ```
#
# Models can change or become unavailable. Check the [TUD:AI model availability page](https://llm.scads.ai/status/) to see which models are currently available, and update the model identifiers in this configuration if necessary. The default `model` uses the format `provider/model-id`.
#
# Restart the project after saving the configuration. Open the provider connection dialog (`/connect` in the command interface), choose **TUD:AI** (provider ID `scads`), and enter your portal key. If you need to add it as a custom provider, use the same provider ID, `scads`. Use the model picker (`/models` in the command interface) to switch between the configured models.
#
# The configuration contains no API key. Requests use the TUD:AI endpoint. See the [OpenCode custom-provider guide](https://opencode.ai/docs/providers/#custom-provider) for further configuration details.
#
# For a first check, ask the agent to explain `exercises/check_setup.py`. Tell it to use the Python interpreter inside `.venv-cfd` when running course code. Read its suggestions and verify the result yourself.
#
# ### Step 6 — Implement and assess the belt-flow solution
#
# Write a small Python script called `belt_flow.py` in your ignored `exercises` folder. Use NumPy and Matplotlib to implement the [mathematical and numerical models developed in Lecture 1](lecture_01.py#example-flow-between-a-belt-and-a-stationary-wall). You may use a coding agent, following the guidance in Topic 0, but you must understand and check the implementation yourself.
#
# 1. **Define the parameters and mesh.** Choose and document physically consistent values for $H$, $L$, $\rho$, $\mu$, $U_{\mathrm{belt}}$, and $\Delta p<0$, including their units. Choose a total number of nodes $N\geq3$ and construct the uniform mesh from $y=0$ to $y=H$.
# 2. **Assemble the linear system.** Fill the coefficient matrix $\mathbf{A}$ and the right-hand-side vector $\mathbf{b}$ for the $N-2$ interior nodes. Include the contributions from the prescribed boundary velocities. Check the array dimensions and the signs of the coefficients and source term.
# 3. **Solve for the velocity.** Solve $\mathbf{A}\mathbf{u}_{\mathrm{num}}=\mathbf{b}$ using `numpy.linalg.solve`. Combine the interior solution with the boundary values to obtain the velocity at all $N$ nodes. Check the boundary values and the residual $\mathbf{A}\mathbf{u}_{\mathrm{num}}-\mathbf{b}$.
# 4. **Visualize and compare.** Evaluate the analytical solution from the lecture using the same parameters. Plot the numerical nodal values and the analytical profile together, with clearly labeled axes and a legend. Use a dense set of positions for the analytical curve so that its shape is visible between the numerical nodes.
# 5. **Investigate mesh refinement.** Repeat the calculation for an increasing number of nodes, for example $N=5,9,17,33,65$. Keep the physical parameters fixed. At each mesh node, calculate $e_i=u_i-u_{\mathrm{exact}}(y_i)$ and report the maximum absolute nodal error $E_\infty=\max_i|e_i|$. Plot or tabulate this error against $N$ or $\Delta y$ and inspect how it changes as the mesh becomes finer.
# 6. **Explain what you observe.** Does the error behave as you expected? Support your explanation using the analytical profile, the derivative approximation, and the numerical solution process. Distinguish errors at the nodes from differences between the straight segments joining those nodes and the continuous analytical profile. Consider whether floating-point arithmetic or the linear-system solve could influence the error you measure.
#
# Save your script, figures, and a short written explanation in `exercises`. Report the parameter values, meshes, checks, and evidence supporting your conclusions. Do not stop at stating that the code runs or that the curves look similar.
#
# ### Step 7 — From the velocity profile to the mass flow rate
#
# Return to the scientific question from the lecture. The local mass flux is $\rho u(y)$; integrating it across the channel height gives the **mass flow rate per unit belt width**:
#
# $$
# \dot m'=\rho\int_0^H u(y)\,dy.
# $$
#
# Its units are $\mathrm{kg\,s^{-1}\,m^{-1}}$. For a belt of width $W$, neglecting sidewall effects, the total mass flow rate is $\dot m=W\dot m'$.
#
# 1. **Integrate the numerical profile.** Extend `belt_flow.py` to approximate $\dot m'_{\mathrm{num}}$ from the computed nodal velocities using the composite trapezoidal rule. Include both boundary nodes in the integration.
# 2. **Obtain the analytical reference.** Integrate the analytical velocity profile from the lecture to derive $\dot m'_{\mathrm{exact}}$ and evaluate it using the same physical parameters. Check its units and its limiting behavior when the pressure difference is zero.
# 3. **Investigate the integration error.** Repeat the comparison on the meshes used in Step 6. Calculate $E_{\dot m'}=|\dot m'_{\mathrm{num}}-\dot m'_{\mathrm{exact}}|$ and plot or tabulate it against $N$ or $\Delta y$. Explain the observed behavior and compare it with the nodal velocity error. Consider separately the numerical velocity solution and the approximation used to integrate it.
# 4. **Vary the pressure difference.** Keep the geometry, fluid properties, and belt speed fixed. Compute the numerical and analytical mass flow rates for a range of $\Delta p\leq0$, including zero, and plot $\dot m'$ against $\Delta p$. Use a mesh whose integration accuracy you have assessed, and label the axes with quantities and units.
# 5. **Explain the physical trend.** Describe why the mass flow rate changes as it does with $\Delta p$. Relate the trend to the pressure-driven and belt-driven contributions, and explain the flow rate at $\Delta p=0$. State what this tells you about the original scientific question and what additional information would be needed to express the result in terms of fan speed.
#
# Save the additional figures and your explanation in `exercises` alongside the script.
