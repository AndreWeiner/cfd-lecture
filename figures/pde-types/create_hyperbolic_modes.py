"""Exact characteristic transport for a frozen ideal-gas Euler state."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
OUT=ROOT/'notebooks/media'
SPEEDS=np.array([-.6,.4,1.4])

def profile(x,t,speed):
    return .1*np.exp(-((x-speed*t)/.35)**2)


def animate(mode):
    OUT.mkdir(exist_ok=True)
    with plt.style.context(['default']+(['dark_background'] if mode=='dark' else [])),plt.rc_context(figure_style()):
        fig,axes=plt.subplots(3,1,figsize=(WIDTH_MM['double']/25.4,5.2),sharex=True,sharey=True)
        fig.patch.set_facecolor('#111111' if mode=='dark' else 'white')
        x=np.linspace(-2.5,4,650)
        lines=[]
        for j,(ax,speed) in enumerate(zip(axes,SPEEDS)):
            ax.set_facecolor(fig.get_facecolor())
            ax.plot(x,profile(x,0,speed),color=plt.rcParams['text.color'],ls=':',alpha=.35,lw=1)
            line,=ax.plot(x,profile(x,0,speed),color=f'C{j}',lw=2)
            center=ax.axvline(0,color=f'C{j}',ls='--',lw=.8,alpha=.6)
            ax.set(xlim=(-2.5,4),ylim=(-.005,.125),ylabel=r'$w_%d$'%(j+1))
            ax.set_yticks([0,.1]);ax.spines[['top','right']].set_visible(False)
            ax.text(.02,.8,r'$\lambda_%d=%+.1f$'%(j+1,speed),transform=ax.transAxes,color=f'C{j}')
            ax.text(.98,.8,['Left-going acoustic mode','Convected entropy/contact mode','Right-going acoustic mode'][j],transform=ax.transAxes,ha='right',fontsize=9)
            lines.append((speed,line,center))
        axes[-1].set_xlabel('$x$')
        clock=fig.text(.5,.96,'',ha='center')
        fig.subplots_adjust(left=.12,right=.98,top=.89,bottom=.12,hspace=.32)
        def update(t):
            for speed,line,center in lines:
                line.set_ydata(profile(x,t,speed));center.set_xdata([speed*t,speed*t])
            clock.set_text(r'$t=%.2f,\quad\bar u=0.4,\quad\bar c=1$'%t)
        anim=FuncAnimation(fig,update,frames=np.r_[np.linspace(0,2,64),np.full(12,2.)],interval=100)
        anim.save(OUT/f'euler-characteristic-modes-{mode}.mp4',writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(1.2)
        fig.savefig(OUT/f'euler-characteristic-modes-{mode}.png',dpi=150)
        fig.savefig(OUT/f'euler-characteristic-modes-{mode}.svg',transparent=True)
        plt.close(fig)

if __name__=='__main__':
    g,u,c=1.4,.4,1.
    H=c*c/(g-1)+.5*u*u
    J=np.array([[0,1,0],[(g-3)*u*u/2,(3-g)*u,g-1],[u*((g-1)*u*u/2-H),H-(g-1)*u*u,g*u]])
    R=np.array([[1,1,1],[u-c,u,u+c],[H-u*c,.5*u*u,H+u*c]])
    assert np.allclose(J@R,R@np.diag(SPEEDS))
    assert np.linalg.matrix_rank(R)==3
    x=np.linspace(-1,1,100);t=.3;h=1e-5
    for speed in SPEEDS:
        dt=(profile(x,t+h,speed)-profile(x,t-h,speed))/(2*h)
        dx=(profile(x+h,t,speed)-profile(x-h,t,speed))/(2*h)
        assert np.allclose(dt+speed*dx,0,atol=1e-8)
    for mode in ['light','dark']:
        animate(mode)
