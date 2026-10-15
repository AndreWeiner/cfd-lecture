"""Exact Fourier-mode comparison for dimensionless diffusion plus reaction."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style, WIDTH_MM
OUT=ROOT/'notebooks/media'
OUT.mkdir(exist_ok=True)
ALPHA, REACTION, EPSILON = .1, .5, .5


def profile(x,t,k,*,diffuse=True,react=True):
    rate=(REACTION if react else 0)-(ALPHA*k*k if diffuse else 0)
    return EPSILON*np.exp(rate*t)*np.cos(k*x)


def draw_animation(mode):
    with plt.style.context(['default']+(['dark_background'] if mode=='dark' else [])),plt.rc_context(figure_style()):
        fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,4.0),sharey=True)
        fig.patch.set_facecolor('#111111' if mode=='dark' else 'white')
        x=np.linspace(0,2*np.pi,500)
        lines=[]
        for ax,k in zip(axes,[1,4]):
            ax.set_facecolor(fig.get_facecolor())
            ax.set(xlim=(0,2*np.pi),ylim=(-1.5,1.5),xlabel=r'$x$')
            ax.set_xticks([0,np.pi,2*np.pi],['$0$',r'$\pi$',r'$2\pi$'])
            ax.spines[['top','right']].set_visible(False)
            ax.axhline(0,color=plt.rcParams['text.color'],alpha=.2,lw=.8)
            ax.set_title(r'$k=%d,\quad\lambda-\alpha k^2=%+.1f$'%(k,REACTION-ALPHA*k*k),fontsize=11)
            full,=ax.plot(x,profile(x,0,k),color='C0',lw=2,label='Diffusion + reaction')
            reaction,=ax.plot(x,profile(x,0,k),color=plt.rcParams['text.color'],alpha=.45,ls='--',lw=1.2,label='Reaction only')
            diffusion,=ax.plot(x,profile(x,0,k),color='C1',ls=':',lw=1.5,label='Diffusion only')
            lines.append((k,full,reaction,diffusion))
        axes[0].set_ylabel(r'$\varphi$')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9)
        clock=fig.text(.5,.08,'',ha='center')
        fig.subplots_adjust(left=.1,right=.98,bottom=.24,top=.76,wspace=.15)
        def update(t):
            for k,full,reaction,diffusion in lines:
                full.set_ydata(profile(x,t,k))
                reaction.set_ydata(profile(x,t,k,diffuse=False))
                diffusion.set_ydata(profile(x,t,k,react=False))
            clock.set_text(r'$t=%.2f,\quad\alpha=0.1,\quad\lambda=0.5$'%t)
        animation=FuncAnimation(fig,update,frames=np.r_[np.linspace(0,2,56),np.full(12,2.)],interval=125)
        animation.save(OUT/f'pde-reaction-diffusion-{mode}.mp4',writer=FFMpegWriter(fps=8,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(1)
        fig.savefig(OUT/f'pde-reaction-diffusion-{mode}.png',dpi=150)
        fig.savefig(OUT/f'pde-reaction-diffusion-{mode}.svg',transparent=True)
        plt.close(fig)

if __name__=='__main__':
    assert REACTION-ALPHA>0 and REACTION-ALPHA*4**2<0
    x=np.linspace(0,2*np.pi,50)
    for k in [1,4]:
        t=.4;step=1e-5
        dt=(profile(x,t+step,k)-profile(x,t-step,k))/(2*step)
        rhs=(REACTION-ALPHA*k*k)*profile(x,t,k)
        assert np.allclose(dt,rhs,atol=1e-9)
    for mode in ['light','dark']:
        draw_animation(mode)
