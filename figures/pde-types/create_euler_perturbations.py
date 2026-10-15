"""Physical perturbations reconstructed from the existing linear Euler modes."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
sys.path.insert(0,str(Path(__file__).resolve().parent))
from create_hyperbolic_modes import ROOT,OUT,SPEEDS,profile
from figure_style import figure_style,WIDTH_MM

# rho_bar=c_bar=1. Rows: rho'/rho_bar, u'/c_bar, p'/(rho_bar*c_bar^2).
TRANSFORM=np.array([[1,1,1],[-1,0,1],[1,0,1]])

def fields(x,t):
    w=np.array([profile(x,t,s) for s in SPEEDS])
    contributions=TRANSFORM[:,:,None]*w[None,:,:]
    return contributions.sum(axis=1),contributions


def animate(mode):
    with plt.style.context(['default']+(['dark_background'] if mode=='dark' else [])),plt.rc_context(figure_style()):
        fig,axes=plt.subplots(3,1,figsize=(WIDTH_MM['double']/25.4,5.6),sharex=True)
        fig.patch.set_facecolor('#111111' if mode=='dark' else 'white')
        x=np.linspace(-2.5,4,650)
        initial,_=fields(x,0);artists=[]
        labels=[r"$\rho'/\bar\rho$",r"$u'/\bar c$",r"$p'/(\bar\rho\bar c^2)$"]
        for j,ax in enumerate(axes):
            ax.set_facecolor(fig.get_facecolor())
            ax.plot(x,initial[j],color=plt.rcParams['text.color'],alpha=.25,ls=':',lw=1)
            total,=ax.plot(x,initial[j],color=plt.rcParams['text.color'],lw=2,label='Total')
            parts=[]
            for k in range(3):
                line,=ax.plot(x,np.zeros_like(x),color=f'C{k}',ls='--',lw=1.1,label=[r'$\lambda_1=-0.6$',r'$\lambda_2=0.4$',r'$\lambda_3=1.4$'][k])
                line.set_visible(TRANSFORM[j,k]!=0)
                parts.append(line)
            ax.axhline(0,color=plt.rcParams['text.color'],alpha=.15,lw=.7)
            ax.set(xlim=(-2.5,4),ylim=[(-.01,.33),(-.12,.12),(-.01,.23)][j],ylabel=labels[j])
            ax.spines[['top','right']].set_visible(False)
            artists.append((total,parts))
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.55,.96))
        clock=fig.text(.55,.99,'',ha='center',va='top')
        axes[-1].set_xlabel('$x$')
        fig.subplots_adjust(left=.19,right=.98,top=.87,bottom=.11,hspace=.34)
        def update(t):
            total,parts=fields(x,t)
            for j,(line,components) in enumerate(artists):
                line.set_ydata(total[j])
                for k,component in enumerate(components):
                    component.set_ydata(parts[j,k])
            clock.set_text(r'$t=%.2f,\quad\bar\rho=1,\quad\bar u=0.4,\quad\bar c=1$'%t)
        anim=FuncAnimation(fig,update,frames=np.r_[np.linspace(0,2,64),np.full(12,2.)],interval=100)
        anim.save(OUT/f'euler-physical-perturbations-{mode}.mp4',writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(1.2)
        fig.savefig(OUT/f'euler-physical-perturbations-{mode}.png',dpi=150)
        fig.savefig(OUT/f'euler-physical-perturbations-{mode}.svg',transparent=True)
        plt.close(fig)

if __name__=='__main__':
    x=np.linspace(-2,3,100);t=.4;h=1e-5
    dt=(fields(x,t+h)[0]-fields(x,t-h)[0])/(2*h)
    dx=(fields(x+h,t)[0]-fields(x-h,t)[0])/(2*h)
    # Primitive linearized Euler equations with rho_bar=c_bar=1.
    assert np.max(np.abs(dt[0]+.4*dx[0]+dx[1]))<1e-8
    assert np.max(np.abs(dt[1]+.4*dx[1]+dx[2]))<1e-8
    assert np.max(np.abs(dt[2]+.4*dx[2]+dx[1]))<1e-8
    for mode in ['light','dark']:
        animate(mode)
