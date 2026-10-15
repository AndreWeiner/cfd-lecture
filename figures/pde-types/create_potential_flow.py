"""Qualitative slender-body geometry and PDE dependence diagrams."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
from figure_utils import save_theme_variants


def ellipse_flow(x,y):
    """Exact incompressible potential velocity outside a centered ellipse.

    Joukowski map z=c(w+m/w), complex potential U*c(w+1/w).
    Lengths scaled by chord; U_infinity=1. Ellipse center x=0.5.
    """
    major,minor=.5,.055
    c=(major+minor)/2
    m=(major-minor)/(major+minor)
    z=np.asarray(x)-.5+1j*np.asarray(y)
    root=np.sqrt(z*z-4*c*c*m+0j)
    w1=(z+root)/(2*c);w2=(z-root)/(2*c)
    w=np.where(np.abs(w1)>=np.abs(w2),w1,w2)
    with np.errstate(divide='ignore',invalid='ignore'):
        velocity=(1-1/w**2)/(1-m/w**2)
    return velocity.real,-velocity.imag


def setup():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,4.0))
        color=plt.rcParams['text.color']
        x=np.linspace(-.5,1.5,450);y=np.linspace(-.42,.42,220)
        xx,yy=np.meshgrid(x,y)
        u,v=ellipse_flow(xx,yy)
        inside=((xx-.5)/.5)**2+(yy/.055)**2<=1.003
        u=np.ma.array(u,mask=inside);v=np.ma.array(v,mask=inside)
        ax.streamplot(x,y,u,v,color='C0',density=[1.2,.65],linewidth=.9,
                      arrowsize=.8,minlength=.15,broken_streamlines=False,
                      start_points=np.c_[np.full(16,-.49),np.r_[np.linspace(-.38,-.035,8),np.linspace(.035,.38,8)]])
        theta=np.linspace(0,2*np.pi,400)
        ax.fill(.5+.5*np.cos(theta),.055*np.sin(theta),color=color,alpha=.2,zorder=3)
        ax.plot(.5+.5*np.cos(theta),.055*np.sin(theta),color=color,lw=1.3,zorder=4)
        qx,qy=np.meshgrid(np.linspace(-.15,1.15,7),[-.25,-.13,.13,.25])
        qu,qv=ellipse_flow(qx,qy)
        arrows=ax.quiver(qx,qy,qu-1,qv,color='C3',angles='xy',scale_units='xy',
                         scale=1.5,width=.0025,headwidth=3.2,headlength=4.0,zorder=5)
        ax.quiverkey(arrows,.83,1.08,.2,r'$|\mathbf{u}-U_\infty\mathbf{e}_x|=0.2U_\infty$',
                     labelpos='N',coordinates='axes',color='C3')
        ax.annotate('',xy=(-.05,0),xytext=(-.4,0),
                    arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
        ax.text(-.3,.035,r'$U_\infty$',ha='center',va='bottom',zorder=6,
                bbox=dict(boxstyle='round,pad=0.18',facecolor=plt.rcParams['axes.facecolor'],
                          edgecolor='none',alpha=.9))
        ax.set(xlim=(-.5,1.5),ylim=(-.42,.42),xlabel=r'$x/L$',ylabel=r'$y/L$',aspect='equal')
        ax.spines[['top','right']].set_visible(False)
        ax.plot(1,0,marker='>',color=color,transform=ax.transAxes,clip_on=False,markersize=7)
        ax.plot(0,1,marker='^',color=color,transform=ax.transAxes,clip_on=False,markersize=7)
        fig.subplots_adjust(left=.1,right=.98,bottom=.17,top=.78)
        return fig


def diagram(kind):
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.8))
        color=plt.rcParams['text.color']
        ax.set(xlim=(-1.3,1.3),ylim=(-1,1),xlabel='$x$',ylabel='$y$',aspect='equal')
        if kind=='hyperbolic':
            beta=np.sqrt(3)  # M=2
            xx=np.linspace(-1.2,1.2,200)
            ax.fill_between(xx[xx<=0],-np.abs(xx[xx<=0])/beta,np.abs(xx[xx<=0])/beta,color='C0',alpha=.2)
            ax.fill_between(xx[xx>=0],-xx[xx>=0]/beta,xx[xx>=0]/beta,color='C1',alpha=.2)
            for sign in [-1,1]:
                ax.plot(xx,sign*xx/beta,color=color,ls='--',lw=1)
            ax.text(-.77,0,'Domain of\ndependence',ha='center',va='center',color=color)
            ax.text(.8,.43,'Range of\ninfluence',ha='center',va='center',color=color)
            # Faint body downstream of P; the bounding rays remain in the fluid.
            theta=np.linspace(0,2*np.pi,300)
            bx=.75+.32*np.cos(theta);by=.045*np.sin(theta)
            ax.fill(bx,by,facecolor=plt.rcParams['axes.facecolor'],zorder=4)
            shade='.75' if color=='black' else '.45'
            ax.fill(bx,by,facecolor=shade,zorder=5)
            ax.text(.75,-.16,'Slender body',ha='center',color=color,fontsize=9)

            ax.set_title(r'$M_\infty=2$: downstream marching')
        elif kind=='sonic':
            ax.axvspan(-1.2,0,color='C0',alpha=.2)
            ax.axvspan(0,1.2,color='C1',alpha=.2)
            ax.axvline(0,color=color,ls='--',lw=1.3)
            ax.text(-.65,.25,'Limiting domain\nof dependence',ha='center',color=color)
            ax.text(.65,.25,'Limiting range\nof influence',ha='center',color=color)
            ax.text(0,.85,r'$\mu_M\to90^\circ$',ha='center',color=color)
            ax.set_title(r'$M_\infty\to1^+$: limiting characteristic geometry')
            theta=np.linspace(0,2*np.pi,300)
            ax.fill(.75+.32*np.cos(theta),.045*np.sin(theta),color='.6',zorder=4)
            ax.text(.75,-.16,'Slender body',ha='center',color=color,fontsize=9)
        else:
            ax.add_patch(Rectangle((-1.1,-.8),2.2,1.6,facecolor='C0',alpha=.12,edgecolor=color,lw=1.4))
            for endpoint in [(-1.1,0),(1.1,0),(0,.8),(0,-.8),(-1.1,.8),(1.1,-.8)]:
                ax.annotate('',xy=endpoint,xytext=(0,0),arrowprops=dict(arrowstyle='<->',color=color,lw=.9))
            ax.text(0,.94,'Boundary data around the whole domain',ha='center')
            ax.text(0,.57,'Domain of dependence\n= range of influence',ha='center',color=color,
                    bbox=dict(facecolor=plt.rcParams['axes.facecolor'],edgecolor='none',alpha=.9,pad=3))
            ax.text(0,-.96,'Global spatial coupling; no finite characteristic wedge',ha='center')
            theta=np.linspace(0,2*np.pi,300)
            bx=.75+.32*np.cos(theta);by=.045*np.sin(theta)
            ax.fill(bx,by,facecolor='.6',zorder=5)
            ax.text(.75,-.16,'Slender body',ha='center',color=color,fontsize=9)
            ax.set_title(r'$M_\infty<1$: elliptic boundary-value problem',pad=28)
        ax.plot(0,0,'o',color=color,ms=5);ax.text(.04,-.13,'$P$',color=color)
        ax.spines[['top','right']].set_visible(False);ax.set_xticks([]);ax.set_yticks([])
        fig.tight_layout();return fig

if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(setup,ROOT/'notebooks/media'/f'slender-body.{ext}',dpi=300)
        for kind in ['hyperbolic','sonic','elliptic']:
            save_theme_variants(lambda k=kind:diagram(k),ROOT/'notebooks/media'/f'potential-{kind}.{ext}',dpi=300)
