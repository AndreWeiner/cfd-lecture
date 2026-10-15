"""Where data are prescribed: elliptic domain and space-time prisms.
Oblique projection is illustrative; time is a separate vertical coordinate.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
from figure_utils import save_theme_variants

def draw(kind):
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.8 if kind=='elliptic' else 4.5))
        fg=plt.rcParams['text.color'];colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.set_aspect('equal');ax.axis('off')
        def arrow(a,b,color=fg):
            ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.2,color=color))
        if kind=='elliptic':
            theta=np.linspace(0,2*np.pi,241);r=1+.1*np.cos(3*theta)
            x=1.45*r*np.cos(theta);y=.83*r*np.sin(theta)
            ax.fill(x,y,color=colors[0],alpha=.12)
            ax.plot(x,y,color=colors[1],lw=2)
            ax.text(0,0,r'$\Omega$',ha='center',va='center',fontsize=18)
            ax.annotate(r'$\partial\Omega$',xy=(x[32],y[32]),xytext=(1.7,1.0),ha='center',arrowprops=dict(arrowstyle='->',color=fg),fontsize=14)
            arrow((-1.9,-1.2),(-1.1,-1.2));arrow((-1.9,-1.2),(-1.9,-.5))
            ax.text(-1.02,-1.2,'$x$',va='center');ax.text(-1.9,-.43,'$y$',ha='center')
            ax.text(0,-1.55,'Boundary data on the entire boundary',ha='center',fontsize=11)
            ax.set(xlim=(-2.3,2.4),ylim=(-1.8,1.5))
        else:
            def p(x,y,t):return np.array([x+.35*y,.48*y+2.15*t])
            theta=np.linspace(0,2*np.pi,241);r=1+.1*np.cos(3*theta)
            x=1.45*r*np.cos(theta);y=.83*r*np.sin(theta)
            dy=.83*(-.3*np.sin(3*theta)*np.sin(theta)+r*np.cos(theta))
            base=np.array([p(xx,yy,0) for xx,yy in zip(x,y)])
            top=base+np.array([0,2.15])
            # Shade the lateral surface strip by strip; normal sign determines inflow.
            for j in range(len(theta)-1):
                c=colors[1] if kind=='parabolic' or (dy[j]+dy[j+1])<0 else colors[2]
                wall=np.array([base[j],base[j+1],top[j+1],top[j]])
                opacity=.17 if kind=='hyperbolic' and (dy[j]+dy[j+1])<0 else .07
                ax.add_patch(Polygon(wall,facecolor=c,alpha=opacity,edgecolor='none'))
            ax.add_patch(Polygon(base,facecolor=colors[0],alpha=.17,edgecolor='none'))
            ax.plot(*base.T,color=colors[0],lw=1.4)
            ax.plot(*top.T,color=fg,ls='--',lw=.9,alpha=.7)
            # Projected silhouette generators show the height of the extrusion.
            for j in [np.argmin(base[:,0]),np.argmax(base[:,0])]:
                ax.plot(*np.array([base[j],top[j]]).T,color=fg,lw=1)
            ax.text(0,-.02,r'$\Omega:\ \varphi_0$',ha='center',va='center',fontsize=12)
            ax.text(0,2.9,'No final-time data',ha='center',fontsize=10)
            # All three coordinate axes share one spatial origin at t=0.
            origin=np.array([-2.15,-.67])
            arrow(origin,origin+np.array([.78,0]));ax.text(-1.29,-.67,'$x$',va='center')
            arrow(origin,origin+np.array([.32,.43]));ax.text(-1.78,-.17,'$y$',ha='center')
            arrow(origin,origin+np.array([0,2.85]));ax.text(-2.15,2.28,'$t$',ha='center')
            ax.text(-2.26,-.67,'$0$',ha='right',va='center',fontsize=10)
            if kind=='parabolic':
                ax.annotate('Boundary data on\n'+r'$\partial\Omega\times(0,T)$',xy=p(.9,-.6,.6),xytext=(2.40,1.35),ha='center',va='center',fontsize=10,arrowprops=dict(arrowstyle='->',color=fg))
                ax.text(0,-1.03,'Initial data throughout the bottom patch',ha='center',fontsize=10)
            else:
                ax.annotate('Inflow boundary:\nprescribed data',xy=p(-1.1,-.3,.6),xytext=(-.2,1.25),ha='center',va='center',color=fg,fontsize=10,bbox=dict(facecolor=fig.get_facecolor(),edgecolor='none',alpha=.9,boxstyle='round,pad=.25'),arrowprops=dict(arrowstyle='->',color=colors[1]))
                ax.annotate('Outflow:\nno independent\nboundary data',xy=p(1.4,.15,.5),xytext=(2.45,1.4),ha='center',va='center',fontsize=10,arrowprops=dict(arrowstyle='->',color=fg))
                ax.text(0,-1.03,r'Inflow: $\mathbf a\cdot\mathbf n<0$; outflow: $\mathbf a\cdot\mathbf n>0$.',ha='center',fontsize=10)
            ax.set(xlim=(-2.65,3.25),ylim=(-1.3,3.20))
        fig.subplots_adjust(left=.03,right=.97,bottom=.03,top=.97)
        return fig

if __name__=='__main__':
    for kind in ['elliptic','parabolic','hyperbolic']:
        for ext in ['png','svg']:
            save_theme_variants(lambda:draw(kind),ROOT/'notebooks/media'/f'well-posed-{kind}.{ext}',dpi=300)
