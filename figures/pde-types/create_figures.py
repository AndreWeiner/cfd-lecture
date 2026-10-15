"""Exact, dimensionless PDE prototypes; no numerical solver is used.
Run with the course's .venv-quarto/bin/python from the repository root.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'notebooks'))
from figure_style import figure_style, WIDTH_MM
from figure_utils import save_theme_variants
OUT = ROOT / 'notebooks/media'
OUT.mkdir(exist_ok=True)


def convection(x, t):
    return np.exp(-((x - t - 0.6) / 0.18)**2)


def diffusion(x, t):
    return 0.5 + 0.4*np.cos(2*np.pi*(x-0.5))*np.exp(-0.08*(2*np.pi)**2*t)


def animate(kind, mode):
    if kind == "diffusion":
        return animate_diffusion(mode)
    with plt.style.context(['default'] + (['dark_background'] if mode == 'dark' else [])), plt.rc_context(figure_style()):
        fig = plt.figure(figsize=(WIDTH_MM['double']/25.4, 4.8))
        fig.patch.set_facecolor('#111111' if mode == 'dark' else 'white')
        ax = fig.add_subplot(projection='3d')
        ax.set_facecolor(fig.get_facecolor())
        adv = kind == 'convection'
        xmax, end = (3., 1.5) if adv else (1., 2.)
        x = np.linspace(0, xmax, 240)
        fn = convection if adv else diffusion
        ax.set(xlim=(0,xmax), ylim=(0,end), zlim=(0,1.1), xlabel=r'$x$', ylabel=r'$t$', zlabel=r'$\varphi$')
        ax.view_init(elev=25, azim=-65)
        ax.set_box_aspect((1.4,1.1,0.8))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
        ax.grid(False)
        history=[]
        for t in np.linspace(0,end,9):
            line,=ax.plot(x, np.full_like(x,t), fn(x,t), color='C0', alpha=.28, lw=1)
            history.append((t,line))
        active,=ax.plot([],[],[],color='C0',lw=2.5,label=r'$\varphi(x,t)$')
        chars=[]
        if adv:
            for start in [.42,.6,.78]:
                line,=ax.plot([],[],[],color='C1',ls='--',lw=1.4)
                chars.append((start,line))
            ax.plot([],[],[],color='C1',ls='--',label=r'$x=x_0+ut$')
        else:
            ax.plot(x,np.full_like(x,end),np.full_like(x,.5),color='C1',ls='--',label=r'$\overline{\varphi}=0.5$')
        ax.legend(loc='upper left',bbox_to_anchor=(0,1.06),frameon=False)
        fig.subplots_adjust(left=.02,right=.9,bottom=.08,top=.9)
        def update(t):
            active.set_data_3d(x,np.full_like(x,t),fn(x,t))
            for ti,line in history:
                line.set_visible(ti<=t+1e-10)
            for start,line in chars:
                tt=np.linspace(0,t,80)
                line.set_data_3d(start+tt,tt,np.full_like(tt,convection(start,0)))
            return []
        times=np.r_[np.linspace(0,end,56),np.full(12,end)]
        anim=FuncAnimation(fig,update,frames=times,interval=125)
        anim.save(OUT/f'pde-{kind}-{mode}.mp4',writer=FFMpegWriter(fps=8,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(end)
        fig.savefig(OUT/f'pde-{kind}-{mode}.png',dpi=150)
        fig.savefig(OUT/f'pde-{kind}-{mode}.svg',transparent=True)
        plt.close(fig)


def animate_diffusion(mode):
    with plt.style.context(['default'] + (['dark_background'] if mode == 'dark' else [])), plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['double']/25.4, 4.2))
        fig.patch.set_facecolor('#111111' if mode == 'dark' else 'white')
        ax.set_facecolor(fig.get_facecolor())
        x = np.linspace(0, 1, 240)
        ax.plot(x, diffusion(x, 0), color=plt.rcParams['text.color'], alpha=.3,
                ls=':', label=r'$\varphi(x,0)$')
        ax.axhline(.5, color='C1', ls='--', label=r'$\overline{\varphi}=0.5$')
        active, = ax.plot(x, diffusion(x, 0), color='C0', lw=2.5,
                          label=r'$\varphi(x,t)$')
        time_label = ax.text(.98, .95, '', transform=ax.transAxes, ha='right', va='top')
        ax.set(xlim=(0,1), ylim=(0,1), xlabel=r'$x$', ylabel=r'$\varphi$')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        def update(t):
            active.set_ydata(diffusion(x,t))
            time_label.set_text(r'$t=%.2f$' % t)
            return active,time_label
        times=np.r_[np.linspace(0,2,56),np.full(12,2.)]
        anim=FuncAnimation(fig,update,frames=times,interval=125)
        anim.save(OUT/f'pde-diffusion-{mode}.mp4', writer=FFMpegWriter(
            fps=8,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(.25)
        fig.savefig(OUT/f'pde-diffusion-{mode}.png',dpi=150)
        fig.savefig(OUT/f'pde-diffusion-{mode}.svg',transparent=True)
        plt.close(fig)


def equilibrium():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.8))
        y=np.linspace(0,1,250)
        ax.plot(4*y*(1-y),y,color='C0',label=r'$u(y)/u_{\max}$')
        ax.axhline(0,color=plt.rcParams['text.color'],lw=2)
        ax.axhline(1,color=plt.rcParams['text.color'],lw=2)
        for yy in np.linspace(.1,.9,9):
            ax.annotate('',xy=(4*yy*(1-yy),yy),xytext=(0,yy),arrowprops=dict(arrowstyle='->',color='C0',lw=1))
        ax.set(xlim=(0,1.12),ylim=(-.07,1.07),xlabel=r'$u/u_{\max}$',ylabel=r'$y/H$')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        return fig

if __name__ == '__main__':
    # Exact checks independent of figure construction.
    assert np.isclose(convection(.6,0),convection(1.6,1))
    assert np.isclose(np.trapezoid(diffusion(np.linspace(0,1,1001),0),np.linspace(0,1,1001)),.5)
    for kind in ['convection','diffusion']:
        for mode in ['light','dark']:
            animate(kind,mode)
    for ext in ['png','svg']:
        save_theme_variants(equilibrium,OUT/f'pde-equilibrium.{ext}',dpi=300)
