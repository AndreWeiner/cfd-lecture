"""Space-time characteristic bounds for constant-state linearized Euler."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
from figure_utils import save_theme_variants

def draw():
    with plt.rc_context(figure_style()):
        fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,4.4),sharey=True)
        color=plt.rcParams['text.color']
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        tb=np.linspace(0,1,100);tf=np.linspace(1,2,100)
        for ax,u,title in zip(axes,[.4,1.5],['Subsonic','Supersonic']):
            speeds=[u-1,u,u+1]
            ax.fill_betweenx(tb,speeds[2]*(tb-1),speeds[0]*(tb-1),facecolor=to_rgba(colors[0],.18))
            ax.fill_betweenx(tf,speeds[0]*(tf-1),speeds[2]*(tf-1),facecolor=to_rgba(colors[1],.18))
            ax.axvline(0,color=color,lw=.7,alpha=.25)
            for j,speed in enumerate(speeds):
                tt=np.linspace(0,2,180)
                ax.plot(speed*(tt-1),tt,color=color,ls='--' if j==1 else '-',lw=1 if j==1 else 1.4)
                ttip=1.65
                ax.annotate('',xy=(speed*(ttip-1),ttip),xytext=(speed*(ttip-.12-1),ttip-.12),arrowprops=dict(arrowstyle='->',color=color,lw=1))
            ax.plot(0,1,'o',color=color,ms=4)
            ax.annotate('$P$',xy=(0,1),xytext=(-13,5),textcoords='offset points',color=color)
            ax.set(xlim=(-2.8,2.8),ylim=(0,2.08),xlabel='$x$')
            ax.set_title(title+'\n'+r'$\bar u=%.1f,\quad\bar c=1$'%u,fontsize=11)
            ax.text(.5,-.25,r'$\lambda_j=(%+.1f,\ %+.1f,\ %+.1f)$'%tuple(speeds),transform=ax.transAxes,ha='center',fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xticks([-2,0,2]);ax.set_yticks([0,1,2])
        axes[0].set_ylabel('$t$')
        handles=[Patch(facecolor=to_rgba(colors[0],.3),label='Dependence bounds'),Patch(facecolor=to_rgba(colors[1],.3),label='Influence bounds')]
        fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=10)
        fig.subplots_adjust(left=.1,right=.98,top=.76,bottom=.26,wspace=.23)
        return fig

if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(draw,ROOT/'notebooks/media'/f'euler-dependence.{ext}',dpi=300)
