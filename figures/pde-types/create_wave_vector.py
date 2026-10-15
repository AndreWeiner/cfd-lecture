"""Wavefronts and their normal wave vector in the x-y plane."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
from figure_utils import save_theme_variants

def draw():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.8))
        n=np.array([.8,.6]);tangent=np.array([-.6,.8]);q=np.linspace(-1,1,100)
        for c in np.linspace(-.8,.8,5):
            pts=c*n+q[:,None]*tangent
            ax.plot(pts[:,0],pts[:,1],color='C0',alpha=.6,lw=1.3)
        ax.annotate('',xy=n,xytext=(0,0),arrowprops=dict(arrowstyle='->',color='C3',lw=2))
        ax.text(.83,.65,r'$\mathbf{n}=(n_x,n_y)$',color='C3')
        ax.plot([0,.8,.8],[0,0,.6],ls='--',color=plt.rcParams['text.color'],alpha=.6,lw=1)
        ax.text(.4,-.1,r'$n_x$',ha='center')
        ax.text(.88,.3,r'$n_y$')
        ax.text(-1.2,.9,r'$n_x x+n_y y=\mathrm{const.}$',color='C0')
        ax.set(xlim=(-1.35,1.65),ylim=(-1.2,1.2),xlabel='$x$',ylabel='$y$',aspect='equal')
        ax.spines[['top','right']].set_visible(False)
        color=plt.rcParams['text.color']
        ax.plot(1,0,marker='>',color=color,transform=ax.transAxes,clip_on=False,markersize=7)
        ax.plot(0,1,marker='^',color=color,transform=ax.transAxes,clip_on=False,markersize=7)
        ax.set_xticks([]);ax.set_yticks([])
        fig.tight_layout()
        return fig
if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(draw,ROOT/'notebooks/media'/f'pde-wave-vector.{ext}',dpi=300)
