"""Airfoil in a channel: inlet, stationary walls, and outlet."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
from figure_utils import save_theme_variants

def draw():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.0))
        fg=plt.rcParams['text.color'];colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for y in [-.8,.8]:
            ax.plot([0,4.8],[y,y],color=fg,lw=1.3)
            ax.fill_between([0,4.8],y,y+np.sign(y)*.065,color=fg,alpha=.15)
            ax.text(2.4,y+np.sign(y)*.20,'$W$',ha='center',va='center',fontsize=13)
        for x,label in [(0,'I'),(4.8,'O')]:
            ax.plot([x,x],[-.8,.8],color=fg,ls='--',lw=.8)
            ax.text(x+(-.26 if x==0 else .26),-.28,f'${label}$',ha='center',fontsize=13)
            ax.annotate('',xy=(x+.40,.05),xytext=(x-.40,.05),arrowprops=dict(arrowstyle='->',color=colors[2],lw=1.5))
            ax.text(x+.07,.17,r'$\mathbf u$',ha='center',color=colors[2])
            ax.annotate('',xy=(x+(-.52 if x==0 else .52),-.07),xytext=(x,-.07),arrowprops=dict(arrowstyle='->',color=fg,lw=1.1))
            ax.text(x+(-.55 if x==0 else .55),-.01,r'$\mathbf n$',ha='center')
        # Symmetric NACA 0012 outline, as in the earlier airfoil illustration.
        x=np.linspace(0,1,400)
        y=.12/.2*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        xx=1.6+1.65*x; yy=1.65*y
        ax.add_patch(Polygon(np.c_[np.r_[xx,xx[::-1]],np.r_[yy,-yy[::-1]]],facecolor=colors[0],alpha=.20,edgecolor='none'))
        ax.plot(np.r_[xx,xx[::-1]],np.r_[yy,-yy[::-1]],color=colors[0],lw=1.2)
        ax.annotate('$W$',xy=(2.2,-.09),xytext=(2.2,-.40),ha='center',color=colors[0],fontsize=13,arrowprops=dict(arrowstyle='-',color=colors[0],lw=.8))
        ax.text(2.4,.46,r'$\Omega$',ha='center',fontsize=14)
        ax.text(2.4,-1.27,r'$I$: inlet\qquad $W$: wall\qquad $O$: outlet',ha='center',fontsize=10)
        ax.set(xlim=(-.85,5.65),ylim=(-1.50,1.15),aspect='equal');ax.axis('off')
        fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
        return fig

if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(draw,ROOT/'notebooks/media'/f'airfoil-boundaries.{ext}',dpi=300)
