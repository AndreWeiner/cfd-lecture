"""Lecture 3 mesh and Taylor figures. Run from any directory with the course Python."""
from pathlib import Path
import sys
import subprocess
from math import factorial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'notebooks'))
from figure_style import figure_style, WIDTH_MM
from figure_utils import save_theme_variants
OUT = ROOT / 'notebooks/media'
OUT.mkdir(parents=True, exist_ok=True)


def mesh():
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 2.5))
        ink = plt.rcParams['text.color']
        ax.plot([0, 6], [1, 1], color='C0', lw=3)
        ax.text(3, 1.18, r'$\Omega=[a,b]$', ha='center')
        nodes = np.arange(7)
        ax.plot(nodes, np.zeros(7), color='C0', lw=1)
        ax.scatter(nodes, np.zeros(7), color='C0', s=28, zorder=3)
        labels = ['$x_0=a$', '$x_1$', r'$\cdots$', '$x_{i-1}$', '$x_i$', r'$\cdots$', '$x_{N-1}=b$']
        for x, label in zip(nodes, labels):
            ax.text(x, -.17, label, ha='center', va='top')
        ax.text(-.35, 0, r'$\Omega_h$', ha='right', va='center')
        ax.annotate('', (4, .48), (3, .48), arrowprops=dict(arrowstyle='<->', color=ink))
        ax.text(3.5, .6, r'$\Delta x_i$', ha='center')
        ax.plot([0, 0], [-.72, -.58], color=ink, lw=1)
        ax.annotate('', (1.5, -.65), (0, -.65), arrowprops=dict(arrowstyle='->', color=ink, shrinkA=0, shrinkB=0))
        ax.text(1.62, -.65, '$x$', ha='left', va='center')
        ax.set(xlim=(-.8, 7), ylim=(-.9, 1.5))
        ax.axis('off')
        fig.subplots_adjust(left=.03, right=.97, top=.97, bottom=.06)
        return fig


def taylor(mode):
    styles = ['default'] + (['dark_background'] if mode == 'dark' else [])
    with plt.style.context(styles), plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 3.6))
        fig.patch.set_facecolor('#111111' if mode == 'dark' else 'white')
        ax.set_facecolor(fig.get_facecolor())
        x = np.linspace(-.5, 2.5, 401)
        ax.plot(x, np.exp(x), color='C0', label=r'$\varphi(x)=e^x$', lw=2)
        line, = ax.plot(x, np.ones_like(x), '--', color='C1', label=r'Taylor polynomial $T_m(x)$', lw=2)
        ax.scatter([1], [np.e], color=plt.rcParams['text.color'], s=22, zorder=4)
        ax.annotate(r'$x_i=1$', (1, np.e), xytext=(12, -27), textcoords='offset points')
        ax.set(xlim=(-.5, 2.5), ylim=(-2, 13), xlabel='$x$', ylabel=r'$\varphi$')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        title = ax.set_title('')
        fig.text(.5, .02, r'$T_m(x)=\sum_{k=0}^{m}\frac{\varphi^{(k)}(1)}{k!}(x-1)^k$', ha='center', fontsize=10)
        fig.subplots_adjust(left=.11, right=.97, bottom=.2, top=.86)
        def update(m):
            line.set_ydata(np.e*sum((x-1)**k/factorial(k) for k in range(int(m)+1)))
            title.set_text('Terms retained: %d; polynomial degree: %d' % (m+1, m))
            return line, title
        frames = np.repeat(np.arange(7), 12)
        animation = FuncAnimation(fig, update, frames=frames, interval=125)
        animation.save(OUT/f'fd-taylor-{mode}.mp4', writer=FFMpegWriter(fps=8, codec='libx264', extra_args=['-pix_fmt','yuv420p']), dpi=150)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT/f"fd-taylor-{mode}.mp4"), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", str(OUT/f"fd-taylor-{mode}.webm")], check=True)
        for m in [0, 2, 6]:
            update(m)
            fig.savefig(OUT/f'fd-taylor-degree-{m}-{mode}.png', dpi=150)
        plt.close(fig)



def first_derivatives():
    """Uniform-grid secants and exact tangent for a dimensionless exponential."""
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 3.6))
        x = np.linspace(.25, 1.75, 400)
        nodes = np.array([.5, 1., 1.5])
        values = np.exp(2*(nodes-1))
        ink = plt.rcParams['text.color']
        ax.plot(x, np.exp(2*(x-1)), color=ink, lw=2, label=r'Function $\varphi(x)=e^{2(x-1)}$')
        ax.plot(x, 1+2*(x-1), color=ink, ls=':', lw=1.5, label=r'Exact tangent at $x_i$')
        # Actual secants: the central chord joins the two outer samples.
        for left, right, color, style, label in [
            (1, 2, 'C1', '--', r'Forward: $D^+\varphi_i$'),
            (0, 1, 'C0', '-.', r'Backward: $D^-\varphi_i$'),
            (0, 2, 'C2', '--', r'Central: $D^0\varphi_i$')]:
            slope = (values[right]-values[left])/(nodes[right]-nodes[left])
            xx = np.linspace(nodes[left]-.08, nodes[right]+.08, 100)
            ax.plot(xx, values[left]+slope*(xx-nodes[left]), color=color, ls=style, lw=2, label=label)
        ax.scatter(nodes, values, color=ink, s=25, zorder=5)
        for xx, yy, label in zip(nodes, values,
                                 [r'$\varphi_{i-1}$', r'$\varphi_i$', r'$\varphi_{i+1}$']):
            ax.plot([.25, xx], [yy, yy], color=ink, alpha=.25, lw=.8)
        ax.set_yticks(values, [r'$\varphi_{i-1}$', r'$\varphi_i$', r'$\varphi_{i+1}$'])
        for xx, yy in zip(nodes, values):
            ax.plot([xx, xx], [-.6, yy], color=ink, alpha=.25, lw=.8)
        ax.set_xticks(nodes, [r'$x_{i-1}$', r'$x_i$', r'$x_{i+1}$'])
        ax.set(xlim=(.25, 1.75), ylim=(-.6, 4.8), xlabel='$x$', ylabel=r'$\varphi$')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        formulas = [
            r'$D^-\varphi_i=\displaystyle\frac{\varphi_i-\varphi_{i-1}}{h}$',
            r'$D^+\varphi_i=\displaystyle\frac{\varphi_{i+1}-\varphi_i}{h}$',
            r'$D^0\varphi_i=\displaystyle\frac{\varphi_{i+1}-\varphi_{i-1}}{2h}$']
        for y, formula in zip([.99, .87, .75], formulas):
            ax.text(.48, y, formula, transform=ax.transAxes,
                    ha='left', va='top', fontsize=12)
        fig.text(.5, .025, r'Uniform mesh: $\Delta x_i=\Delta x_{i+1}=h$; exact samples; dimensionless example', ha='center', fontsize=9)
        fig.subplots_adjust(left=.1, right=.97, bottom=.23, top=.97)
        return fig


def mapping_meshes():
    """Pair uniform computational nodes with their mapped physical nodes."""
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 3.2))
        xi = np.linspace(0, 1, 7)
        physical = np.expm1(2*xi)/np.expm1(2)
        ink = plt.rcParams['text.color']
        labels = ['0', '1', '2', '3', '4', '5', 'N-1']
        for upper, lower in zip(xi, physical):
            ax.annotate('', (lower, .07), (upper, .93),
                        arrowprops=dict(arrowstyle='->', color=ink, alpha=.35, lw=.8,
                                        shrinkA=0, shrinkB=0))
        for y, nodes, coordinate, color in [(1, xi, r'\xi', 'C0'), (0, physical, 'x', 'C1')]:
            ax.plot([0, 1], [y, y], color=color, lw=1.5)
            ax.scatter(nodes, np.full(7, y), color=color, s=24, zorder=4)
            for j in [0, 1, 3, 6]:
                ax.text(nodes[j], y+(.1 if y else -.1),
                        '$'+coordinate+'_{'+labels[j]+'}$',
                        ha='center', va='bottom' if y else 'top', fontsize=10)
        ax.text(.5, 1.65, r'Uniform computational mesh: $\widehat\Omega_{\Delta\xi}$', ha='center')
        ax.text(.5, -.7, r'Nonuniform physical mesh: $\Omega_h$', ha='center')
        ax.text(1.03, .5, r'$x=g(\xi)$', va='center', ha='left')
        ax.annotate('', (xi[4], 1.3), (xi[3], 1.3),
                    arrowprops=dict(arrowstyle='<->', color=ink, shrinkA=0, shrinkB=0))
        ax.text((xi[3]+xi[4])/2, 1.34, r'$\Delta\xi$', ha='center', va='bottom', fontsize=10)
        ax.annotate('', (physical[4], -.3), (physical[3], -.3),
                    arrowprops=dict(arrowstyle='<->', color=ink, shrinkA=0, shrinkB=0))
        ax.text((physical[3]+physical[4])/2, -.34, r'$\Delta x_4$', ha='center', va='top', fontsize=10)
        ax.set(xlim=(-.07, 1.27), ylim=(-.9, 1.9))
        ax.axis('off')
        fig.subplots_adjust(left=.04, right=.97, bottom=.05, top=.95)
        return fig


def time_integration():
    """Quadrature geometry on a dimensionless illustrative RHS trajectory."""
    with plt.rc_context(figure_style()):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM['double']/25.4, 5.4), sharex=True, sharey=True)
        t = np.linspace(0, 1, 401)
        rate = lambda q: .7+.3*q+q*q
        mean = .7+.15+1/3
        ink = plt.rcParams['text.color']
        panels = [('Forward Euler', [0], r'$\overline f\approx f^n$'),
                  ('Backward Euler', [1], r'$\overline f\approx f^{n+1}$'),
                  ('Crank--Nicolson', [0, 1], r'$\overline f\approx(f^n+f^{n+1})/2$'),
                  ('Midpoint rule', [.5], r'$\overline f\approx f(t^{n+1/2})$')]
        for ax, (name, samples, formula) in zip(axes.flat, panels):
            approximation = np.interp(t, [0, 1], [rate(0), rate(1)]) if len(samples)==2 else np.full_like(t, rate(samples[0]))
            ax.fill_between(t, 0, approximation, color='C1', alpha=.18)
            ax.plot(t, approximation, color='C1', ls='--', lw=1.5)
            ax.plot(t, rate(t), color='C0', lw=1.8)
            ax.axhline(mean, color=ink, ls=':', lw=1)
            ax.scatter(samples, [rate(q) for q in samples], color='C1', s=22, zorder=5)
            if samples == [.5]:
                ax.plot([.5,.5], [0,rate(.5)], color='C1', ls=':', lw=.8)
            ax.set_title(name, fontsize=11)
            ax.text(.04,.92,formula,transform=ax.transAxes,va='top',fontsize=10)
            ax.set(xlim=(0,1),ylim=(0,2.6))
            ax.set_xticks([0,.5,1], [r'$t^n$',r'$t^{n+1/2}$',r'$t^{n+1}$'])
            ax.set_yticks([])
            ax.spines[['top','right']].set_visible(False)
        for ax in axes[:,0]: ax.set_ylabel(r'$f(\varphi(t))$')
        for ax in axes[1,:]: ax.set_xlabel('$t$')
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],color='C0',label='Exact illustrative rate'),
                            Line2D([],[],color='C1',ls='--',label='Quadrature approximation'),
                            Line2D([],[],color=ink,ls=':',label='Exact interval mean')],
                   loc='lower center',ncol=3,fontsize=8,frameon=False)
        fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.93,hspace=.38,wspace=.15)
        return fig


def multistep_methods():
    """AB2 rate extrapolation and BDF2 endpoint differentiation."""
    with plt.rc_context(figure_style()):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM['double']/25.4, 3.7))
        ink = plt.rcParams['text.color']
        t = np.linspace(-1, 1, 301)
        rate = lambda z: 1+.4*z+.3*z*z
        extrapolation = 1+.1*t
        ax = axes[0]
        ax.plot(t, rate(t), color='C0', label='Illustrative exact rate')
        ax.plot(t, extrapolation, '--', color='C1', label='Linear extrapolation')
        future = np.linspace(0,1,100)
        ax.fill_between(future, 0, 1+.1*future, color='C1', alpha=.15)
        ax.scatter([-1,0], [rate(-1),rate(0)], color=ink, s=22, zorder=5)
        ax.axvline(0,color=ink,lw=.7,ls=':')
        ax.set_title('Adams--Bashforth (AB2)',fontsize=11)
        ax.set(ylabel=r'$f(\varphi(t))$',ylim=(0,2.4))
        ax.text(.04,.96,r'$\overline f\approx\frac32 f^n-\frac12 f^{n-1}$',
                transform=ax.transAxes,va='top',fontsize=10)
        ax.legend(loc='lower left',frameon=False,fontsize=8)
        ax = axes[1]
        # y'=y, illustrative known states y(-1)=exp(-1), y(0)=1.
        # Solve BDF2 at dt=1: (3*ynew-4+exp(-1))/2=ynew.
        new = 4-np.exp(-1)
        coeff = np.polyfit([-1,0,1],[np.exp(-1),1,new],2)
        poly = np.polyval(coeff,t)
        ax.plot(t,poly,color='C0',label='Quadratic through states')
        tangent_t = np.linspace(.5,1.12,80)
        ax.plot(tangent_t,new+new*(tangent_t-1),'--',color='C1',label='Endpoint tangent')
        ax.scatter([-1,0],[np.exp(-1),1],color=ink,s=22,zorder=5)
        ax.scatter([1],[new],facecolors='none',edgecolors='C1',s=48,zorder=6)
        ax.annotate(r'$\varphi^{n+1}$: solve for this state', (1,new),
                    xytext=(.1,4.5),textcoords='data',fontsize=8,
                    arrowprops=dict(arrowstyle='->',color=ink,lw=.8))
        ax.text(.04,.96,r'$p\prime(t^{n+1})=f(\varphi^{n+1})$',transform=ax.transAxes,va='top',fontsize=10)
        ax.set_title('Backward differentiation (BDF2)',fontsize=11)
        ax.set(ylabel=r'$\varphi$',ylim=(0,5.5))
        ax.legend(loc='upper left',bbox_to_anchor=(0,.84),frameon=False,fontsize=8)
        for ax in axes:
            ax.set_xlim(-1.1,1.15)
            ax.set_xticks([-1,0,1],[r'$t^{n-1}$',r'$t^n$',r'$t^{n+1}$'])
            ax.set_yticks([])
            ax.set_xlabel('$t$')
            ax.spines[['top','right']].set_visible(False)
        fig.text(.5,.03,'Known states: filled markers; new BDF2 state: open marker. Constant time steps.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.2,wspace=.25)
        return fig


def convection_diffusion_problem():
    """Initial condition and periodic domain for the unit-interval model."""
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.4))
        ink=plt.rcParams['text.color']
        x=np.linspace(0,1,500)
        ax.plot(x,np.sin(2*np.pi*x),color='C0',lw=2,label=r'$\varphi(x,0)=\sin(2\pi x)$')
        ax.axhline(0,color=ink,lw=.8)
        ax.plot([0,0],[-1.15,1.15],':',color=ink,lw=.8)
        ax.plot([1,1],[-1.15,1.15],':',color=ink,lw=.8)
        ax.scatter([0,1],[0,0],color='C0',s=25,zorder=4)
        ax.annotate('',(.72,1.3),(.35,1.3),arrowprops=dict(arrowstyle='->',color=ink))
        ax.text(.535,1.4,r'$v>0$',ha='center')
        ax.annotate('',(1,-1.38),(0,-1.38),arrowprops=dict(arrowstyle='<->',color='C1'))
        ax.text(.5,-1.48,'Periodic boundaries: right end connects to left end',ha='center',va='top',fontsize=10)
        ax.set(xlim=(-.04,1.04),ylim=(-1.8,1.8),xlabel='$x$',ylabel=r'$\varphi$')
        ax.set_xticks([0,.5,1],['$0$','$1/2$','$1$'])
        ax.set_yticks([-1,0,1])
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left',frameon=False,fontsize=10)
        fig.subplots_adjust(left=.1,right=.97,bottom=.2,top=.96)
        return fig

if __name__ == '__main__':
    for ext in ['svg', 'png']:
        save_theme_variants(mesh, OUT/f'fd-mesh.{ext}', dpi=300)
        save_theme_variants(first_derivatives, OUT/f'fd-first-derivatives.{ext}', dpi=300)
        save_theme_variants(mapping_meshes, OUT/f'fd-mapping.{ext}', dpi=300)
        save_theme_variants(time_integration, OUT/f'fd-time-integration.{ext}', dpi=300)
        save_theme_variants(multistep_methods, OUT/f'fd-multistep.{ext}', dpi=300)
        save_theme_variants(convection_diffusion_problem, OUT/f'fd-convection-diffusion.{ext}', dpi=300)
    for mode in ['light', 'dark']:
        taylor(mode)
