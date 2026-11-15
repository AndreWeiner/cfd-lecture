"""Reproduce the Lecture 4 media from the repository root.

Run python figures/consistency/create_figures.py notebooks/media.
The output uses LaTeX text, and the animations require FFmpeg.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
from figure_style import WIDTH_MM, figure_style
from figure_utils import save_theme_variants

WIDTH = WIDTH_MM["double"] / 25.4
DARK_STYLE = {"figure.facecolor": "#111111", "axes.facecolor": "#111111"}


def error_sketch():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9), layout="constrained")
    x = np.linspace(0, 6, 301)
    nodes = np.arange(7)
    exact = lambda s: 0.95 + 0.4 * np.sin(0.9 * s - 0.7)
    values = exact(nodes) + np.array([-0.02, 0.04, 0.13, 0.36, 0.15, -0.08, -0.09])
    ax.plot(x, exact(x), color="C0", label=r"Exact profile $\varphi(x,t^n)$")
    ax.plot(nodes, values, "o", color="C1", label=r"Numerical values $\varphi_j^n$")
    ax.plot(3, exact(3), "o", color="C0")
    ax.annotate("", xy=(3, values[3]), xytext=(3, exact(3)),
                arrowprops={"arrowstyle": "<->", "color": "C3", "shrinkA": 4, "shrinkB": 4})
    ax.annotate(r"$e_i^n$", (3, (values[3] + exact(3)) / 2),
                xytext=(9, 0), textcoords="offset points", va="center")
    ax.set(xlabel=r"$x$", ylabel=r"$\varphi$", xlim=(-0.2, 6.2), ylim=(0.35, 1.85),
           xticks=nodes[1:-1], yticks=[])
    ax.set_xticklabels([r"$x_{i-2}$", r"$x_{i-1}$", r"$x_i$", r"$x_{i+1}$", r"$x_{i+2}$"])
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    return fig


def characteristic_sketch():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), layout="constrained")
    for ax, courant in zip(axes, (0.75, 1.25)):
        ax.set(xlim=(-2.35, 0.55), ylim=(-0.34, 1.24),
               xticks=[-2, -1, 0], yticks=[0, 1], title=rf"$C={courant}$")
        ax.set_xticklabels([r"$x_{i-2}$", r"$x_{i-1}$", r"$x_i$"])
        ax.set_yticklabels([r"$t^n$", r"$t^{n+1}$"])
        ax.grid(alpha=0.25)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.plot([-1, 0], [0, 0], color="C0", linewidth=3, marker="o",
                label="Upwind stencil span")
        ax.plot([-courant, 0], [0, 1], "--", color="C1", label="Characteristic")
        ax.plot(-courant, 0, "D", color="C1", markersize=5)
        ax.plot(0, 1, "o", color="C0")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1), frameon=False,
                   fontsize=10, handlelength=2)
    return fig


def upwind_history(initial, courant, stride, frames):
    values = initial.copy()
    history = [values.copy()]
    for _ in range(frames - 1):
        for _ in range(stride):
            values = (1 - courant) * values + courant * np.roll(values, 1)
        history.append(values.copy())
    return np.asarray(history)


def upwind_animation(output):
    nodes = 64
    x = np.arange(nodes) / nodes
    initial = np.sin(2 * np.pi * x) + 1e-4 * (-1.0)**np.arange(nodes)
    frames = 16
    times = np.arange(frames) * 2.4 / nodes
    cases = [(0.8, upwind_history(initial, 0.8, 3, frames)),
             (1.2, upwind_history(initial, 1.2, 2, frames))]
    # Dimensionless period and speed are both one. Close the periodic plot at x=1.
    plot_x = np.r_[x, 1.0]
    reference = np.sin(2 * np.pi * (plot_x[None, :] - times[:, None]))

    for theme, theme_style in (("light", ["default"]),
                               ("dark", ["default", "dark_background", DARK_STYLE])):
        with plt.style.context([*theme_style, figure_style()]):
            fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), layout="constrained",
                                     sharey=True)
            time_label = fig.suptitle(r"$t=0.000$")
            exact_lines, numerical_lines = [], []
            for ax, (courant, _) in zip(axes, cases):
                ax.set(xlim=(0, 1), ylim=(-3.6, 3.6), xlabel=r"$x$", title=rf"$C={courant}$")
                ax.grid(alpha=0.2)
                exact_lines.extend(ax.plot(plot_x, reference[0], "--", color="C0",
                                           label="Exact smooth wave"))
                numerical_lines.extend(ax.plot(plot_x, np.r_[initial, initial[0]], color="C1",
                                               marker=".", markersize=2, label="Upwind"))
            axes[0].set_ylabel(r"$\varphi$")
            axes[0].legend(loc="upper left", frameon=False, fontsize=9)
            fig.canvas.draw()
            fig.set_layout_engine(None)

            def update(frame):
                time_label.set_text(rf"$t={times[frame]:.3f}$")
                for exact, numerical, (_, history) in zip(exact_lines, numerical_lines, cases):
                    values = history[frame]
                    exact.set_ydata(reference[frame])
                    numerical.set_ydata(np.r_[values, values[0]])

            clip = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
            # H.264 with YUV 4:2:0 requires even pixel dimensions. The writer
            # adjusts the figure size to the nearest compatible pixel size.
            writer = animation.FFMpegWriter(fps=4, codec="h264",
                                            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            clip.save(output / f"csc-upwind-{theme}.mp4", writer=writer, dpi=144)
            plt.close(fig)


def complex_plane():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1), layout="constrained")
    a, b = 2.0, 1.25
    neutral = plt.rcParams["text.color"]
    ax.axhline(0, color=neutral, linewidth=0.8)
    ax.axvline(0, color=neutral, linewidth=0.8)
    ax.plot([0, a, a], [b, b, -b], ":", color=neutral, linewidth=1)
    ax.plot([0, a], [0, b], color="C0", marker="o", markevery=[1])
    ax.plot([0, a], [0, -b], "--", color="C1", marker="o", markevery=[1])
    angle = np.linspace(0, np.arctan2(b, a), 40)
    ax.plot(0.85 * np.cos(angle), 0.85 * np.sin(angle), color=neutral)
    ax.text(0.98, 0.23, r"$\alpha$")
    ax.text(1.07, 0.78, r"$r$", color=neutral)
    ax.text(a + 0.12, b, r"$z=a+\mathrm{i}b$", va="center")
    ax.text(a + 0.12, -b, r"$\overline{z}=a-\mathrm{i}b$", va="center")
    ax.text(a, 0.1, r"$a$", ha="center")
    ax.text(-0.12, b, r"$b$", ha="right", va="center")
    ax.text(3.43, 0.11, r"$\mathrm{Re}$", ha="right")
    ax.text(-0.14, 1.55, r"$\mathrm{Im}$", ha="right")
    ax.set(xlim=(-0.5, 3.5), ylim=(-1.55, 1.75), aspect="equal")
    ax.set_axis_off()
    return fig


def periodic_modes():
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.5), layout="constrained",
                             sharex=True, sharey=True)
    nodes = np.linspace(0, 1, 9)
    x = np.linspace(0, 1, 401)
    for ax, (mode, color, title) in zip(axes, [
        (1, "C0", r"Longest nonconstant mode, $\lambda=L$"),
        (4, "C1", r"Nyquist mode, $\lambda=2\Delta x$"),
    ]):
        ax.set(title=title, xlim=(-0.035, 1.035), ylim=(-1.18, 1.24),
               yticks=[-1, 0, 1], ylabel=r"$w_j$")
        ax.axhline(0, color=plt.rcParams["axes.edgecolor"], linewidth=0.8, alpha=0.35)
        ax.plot(x, np.cos(2 * np.pi * mode * x), "--", color=color)
        ax.plot(nodes[:-1], np.cos(2 * np.pi * mode * nodes[:-1]), "o", color=color)
        ax.plot(1, 1, "o", markerfacecolor="none", color=color)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks([0, 0.125, 0.25, 1], [r"$0$", r"$\Delta x$", r"$2\Delta x$", r"$L$"])
    axes[-1].set_xlabel(r"$x$")
    return fig


def modified_wavenumbers():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3), layout="constrained")
    theta = np.linspace(0, np.pi, 401)
    x = theta / np.pi
    neutral = plt.rcParams["text.color"]
    ax.plot(x, x, ":", color=neutral, label=r"Exact $\kappa/\kappa_{\max}$")
    ax.plot(x, np.sin(theta) / np.pi, color="C0",
            label=r"$\mathrm{Re}(\widetilde\kappa_0)=\mathrm{Re}(\widetilde\kappa_-)$")
    ax.plot(x, (1 - np.cos(theta)) / np.pi, "--", color="C1",
            label=r"$-\mathrm{Im}(\widetilde\kappa_-)$")
    ax.plot(x, 2 * np.sin(theta / 2) / np.pi, "-.", color="C2",
            label=r"$\widetilde\kappa_2$")
    ax.set(xlim=(0, 1), ylim=(0, 1.035), xlabel=r"$\kappa/\kappa_{\max}$",
           ylabel=r"Modified wavenumber component $/\,\kappa_{\max}$")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    return fig


def stability_regions():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), layout="constrained",
                             sharex=True, sharey=True)
    c = np.linspace(0, 1, 401)
    for ax, (lower, upper, color, title) in zip(axes, [
        (c**2 / 2, np.full_like(c, 0.5), "C0", "Central convection"),
        (np.zeros_like(c), (1 - c) / 2, "C1", "Upwind convection"),
    ]):
        ax.fill_between(c, lower, upper, color=color, alpha=0.26)
        ax.plot(c, upper, color=color)
        ax.plot(c, lower, color=color)
        ax.plot([0, 0], [lower[0], upper[0]], color=color)
        ax.set(xlim=(-0.035, 1.08), ylim=(-0.024, 0.56), title=title,
               xlabel=r"Courant number $C$", xticks=np.linspace(0, 1, 6))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Diffusion number $D$")
    axes[0].text(0.23, 0.34, r"$C^2\leq 2D\leq 1$")
    axes[1].text(0.13, 0.15, r"$C+2D\leq 1$")
    return fig


def spectral_errors():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), layout="constrained",
                             sharex=True)
    theta = np.linspace(0, np.pi, 401)
    c = 0.6
    transport = -1j * c * np.sin(theta)
    sine_squared = np.sin(theta / 2)**2
    gains = [
        (1 + transport, "C0", "-", "Euler central"),
        (1 + transport - 2 * c * sine_squared, "C1", "--", "Euler upwind"),
        (1 + transport - 2 * c**2 * sine_squared, "C2", "-.", "Lax--Wendroff"),
    ]
    for gain, color, linestyle, label in gains:
        axes[0].plot(theta / np.pi, np.abs(gain) - 1, linestyle, color=color, label=label)
        axes[1].plot(theta / np.pi, np.unwrap(np.angle(gain)) + c * theta,
                     linestyle, color=color)
    axes[0].set_ylabel(r"Amplitude factor error")
    axes[1].set_ylabel(r"Phase error (rad)")
    axes[0].legend(frameon=False, loc="lower left")
    for ax in axes:
        ax.axhline(0, color=plt.rcParams["axes.edgecolor"], linewidth=0.8, alpha=0.5)
        ax.set(xlim=(0, 1), xlabel=r"$\kappa/\kappa_{\max}$")
        ax.grid(alpha=0.2)
    return fig


def lax_wendroff_profiles():
    nodes = 80
    final_time = 0.5
    x = np.arange(nodes + 1) / nodes
    fine_x = np.linspace(0, 1, 801)

    def sine(position):
        return np.sin(8 * np.pi * position)

    def pulse(position):
        wrapped = np.mod(position, 1)
        return ((wrapped >= 0.15625) & (wrapped < 0.40625)).astype(float)

    profiles = []
    for initial in (sine, pulse):
        results = []
        for courant in (0.25, 0.5, 1.0):
            values = initial(x[:-1])
            for _ in range(round(final_time * nodes / courant)):
                left, right = np.roll(values, 1), np.roll(values, -1)
                values = (values - (courant / 2) * (right - left)
                          + (courant**2 / 2) * (right - 2 * values + left))
            results.append(np.r_[values, values[0]])
        profiles.append(results)

    def draw():
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.1), layout="constrained",
                                 sharex=True)
        for ax, (title, results) in zip(axes, zip(("Sine wave", "Rectangular pulse"), profiles)):
            for values, color, style, label in zip(
                results, ("C1", "C2", "C3"), ("-", "-.", "none"),
                (r"$C=0.25$", r"$C=0.5$", r"$C=1$ (nodes)"),
            ):
                is_shift = style == "none"
                ax.plot(x, values, linestyle=style, color=color, label=label,
                        marker="o" if is_shift else None, markerfacecolor="none",
                        markersize=3, markevery=3)
            ax.set(xlim=(0, 1), title=title, ylabel=r"$\varphi$")
            ax.grid(alpha=0.2)

        axes[0].plot(fine_x, sine(fine_x - final_time), "--", color="C0", label="Exact")
        axes[1].plot([0, 0.65625, 0.65625, 0.90625, 0.90625, 1], [0, 0, 1, 1, 0, 0],
                     "--", color="C0")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1],
                       ncol=4, loc="upper center", frameon=False)
        axes[0].set_ylim(-1.18, 1.56)
        axes[1].set_ylim(-0.3, 1.3)
        axes[-1].set_xlabel(r"$x/L$")
        return fig

    return draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    with plt.style.context(["default", figure_style()]):
        smoke = plt.figure()
        smoke.text(0.1, 0.5, r"Exact profile $\varphi(x_i,t^n)$")
        smoke.canvas.draw()
        plt.close(smoke)

    figures = (
        (error_sketch, "error"),
        (characteristic_sketch, "characteristics"),
        (complex_plane, "complex-plane"),
        (periodic_modes, "periodic-modes"),
        (modified_wavenumbers, "modified-wavenumbers"),
        (stability_regions, "stability-regions"),
        (spectral_errors, "spectral-errors"),
        (lax_wendroff_profiles(), "lax-wendroff"),
    )
    for draw, stem in figures:
        def styled_draw():
            with plt.rc_context(figure_style()):
                return draw()

        save_theme_variants(styled_draw, output / f"csc-{stem}.svg", transparent=False)
    upwind_animation(output)


if __name__ == "__main__":
    main()
