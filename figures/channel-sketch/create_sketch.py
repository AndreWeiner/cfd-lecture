"""Redraw the user's rotated channel sketch. Profile is schematic, not a solution."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Ellipse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_utils import save_theme_variants
from figure_style import figure_style, DPI, WIDTH_MM
OUT=Path(__file__).resolve().parent


def draw():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,105/25.4))
        ax.set(xlim=(-.85,6.65),ylim=(-1.35,3.35),aspect='equal')
        ax.axis('off')
        ink=plt.rcParams['text.color']
        def arrow(a,b,color=ink,style='->',lw=1.3):
            ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle=style,color=color,lw=lw))
        L,H=4.5,2.0
        ax.plot([0,L],[0,0],color=ink,lw=2)
        ax.plot([0,L],[H,H],color=ink,lw=2)
        for x in np.linspace(.1,L-.1,22):
            ax.plot([x,x-.16],[0,-.18],color=ink,lw=.8)
        ax.text(L/2,-.4,r'$u=0$ (no slip)',ha='center',va='center',fontsize=12)
        # Closed belt: lower run contacts the fluid, upper run returns leftward.
        thickness=.32
        ax.plot([0,L],[H+thickness,H+thickness],color=ink,lw=2)
        for x,theta1,theta2 in [(0,90,270),(L,-90,90)]:
            ax.add_patch(Arc((x,H+thickness/2),thickness,thickness,
                             theta1=theta1,theta2=theta2,color=ink,lw=2))
            ax.add_patch(Circle((x,H+thickness/2),.07,fill=False,color=ink,lw=.9))
        arrow((1.65,H+thickness+.18),(.45,H+thickness+.18),'C0')
        arrow((.45,H-.18),(1.65,H-.18),'C0')
        ax.text(1.05,H-.43,r'$U_{\mathrm{belt}}>0$',ha='center',fontsize=12,color='C0')
        for x,label in [(0,r'$p_{\mathrm{in}}$'),(L,r'$p_{\mathrm{out}}$')]:
            ax.plot([x,x],[0,H],color=ink,lw=.75,linestyle=(0,(4,4)))
            ax.text(x,H+thickness+.24,label,ha='center',fontsize=12)
        # Downstream fan symbol, separate from the channel-height dimension.
        fan_x,fan_y=5.92,1.05
        ax.add_patch(Circle((fan_x,fan_y),.33,fill=False,color=ink,lw=1))
        for degrees in [0,120,240]:
            theta=np.deg2rad(degrees)
            ax.add_patch(Ellipse((fan_x+.15*np.cos(theta),fan_y+.15*np.sin(theta)),
                                 .32,.13,angle=degrees+30,facecolor='C1',edgecolor='C1',alpha=.75))
        ax.add_patch(Circle((fan_x,fan_y),.055,color=ink))
        ax.text(fan_x,.53,'Fan',ha='center',fontsize=11)
        # Schematic streamwise velocity profile following the handwritten curve.
        base=2.25
        yy=np.linspace(0,H,150)
        uu=.9*(1-np.exp(-3*yy/H))/(1-np.exp(-3))
        ax.plot([base,base],[0,H],color='C0',lw=1)
        ax.plot(base+uu,yy,color='C0',lw=1.7)
        for y in [.4,.8,1.2,1.6]:
            u=.9*(1-np.exp(-3*y/H))/(1-np.exp(-3))
            arrow((base,y),(base+u,y),'C0')
        ax.text(3.34,1.03,r'$u(y)$',fontsize=13,color='C0')
        arrow((0,0),(.7,0));arrow((0,0),(0,.7))
        ax.text(.77,.13,r'$x$',fontsize=12)
        ax.text(-.12,.75,r'$y$',fontsize=12)
        ax.add_patch(Circle((0,0),.065,fill=False,color=ink,lw=1))
        ax.plot(0,0,'o',color=ink,ms=2)
        ax.text(-.18,-.17,r'$z$',fontsize=12)
        arrow((0,-.8),(L,-.8),style='<->')
        ax.text(L/2,-1.02,r'$L$',ha='center',fontsize=13)
        for x in [0,L]:ax.plot([x,x],[-.62,-.95],color=ink,lw=.7)
        arrow((L+.42,0),(L+.42,H),style='<->')
        ax.text(L+.58,H/2,r'$H$',va='center',fontsize=13)
        for y in [0,H]:ax.plot([L+.22,L+.55],[y,y],color=ink,lw=.7)
        fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975)
        return fig

if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(draw,OUT/f'channel-sketch.{ext}',dpi=DPI)
