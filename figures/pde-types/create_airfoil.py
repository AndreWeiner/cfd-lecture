"""Schematic transonic airfoil: regions illustrate models, not computed data."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
from matplotlib.transforms import Affine2D
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style, WIDTH_MM
from figure_utils import save_theme_variants


def draw():
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,4.6))
        x=np.linspace(0,1,400)
        y=.12/.2*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        foreground=plt.rcParams['text.color']
        # Supersonic pocket and exaggerated attached boundary-layer thickness.
        m=(x>=.045)&(x<=.63)
        cap=y[m]+.22*np.sin(np.pi*(x[m]-.045)/.86)
        ax.fill_between(x[m],y[m],cap,color='C1',alpha=.25)
        attached=x<.64
        ax.fill_between(x[attached],y[attached],y[attached]+.025*np.sqrt(x[attached]),color='C2',alpha=.7)
        ax.fill_between(x,-y-.025*np.sqrt(x),-y,color='C2',alpha=.7)
        ax.add_patch(Polygon(np.c_[np.r_[x,x[::-1]],np.r_[y,-y[::-1]]],facecolor=plt.rcParams['axes.facecolor'],edgecolor=foreground,lw=1.5,zorder=5))
        ax.plot([.63,.63],[np.interp(.63,x,y),cap[-1]],color='C3',lw=2)
        wake_x=np.linspace(.64,1.45,180)
        wake_y=.055+.045*np.sin(18*(wake_x-.64))*(1-np.exp(-5*(wake_x-.64)))
        ax.plot(wake_x,wake_y,color='C2',lw=1.5)
        # Roll-up and smaller eddies indicate an unsteady, turbulent wake.
        ax.fill_between(wake_x, wake_y-.065*(wake_x-.64),
                        wake_y+.065*(wake_x-.64), color='C2', alpha=.12)
        for k,(cx,cy,size) in enumerate([(0.91,.075,.038),(1.12,.03,.052),(1.34,.085,.06),
                                       (1.05,.105,.018),(1.26,-.015,.024)]):
            theta=np.linspace(0,3.7*np.pi,180)
            radius=np.linspace(size,.004,len(theta))
            sign=1 if k%2 else -1
            sx=cx+radius*np.cos(theta)
            sy=cy+sign*radius*np.sin(theta)
            ax.plot(sx,sy,color='C2',lw=1.2)

        # Positive incidence: nose raised relative to the horizontal free stream.
        rotation=Affine2D().rotate_deg_around(.25,0,-5)
        for artist in [*ax.lines,*ax.collections,*ax.patches,*ax.texts]:
            artist.set_transform(rotation+ax.transData)
        def point(x,y):
            return rotation.transform((x,y))
        # Acoustic fronts radiating approximately normal to the wake.
        source=point(1.18,.045)
        for radius in [.16,.29,.42]:
            for low,high in [(55,125),(235,305)]:
                ax.add_patch(Arc(tuple(source),2*radius,2*radius,
                                 theta1=low,theta2=high,color='C0',ls='--',lw=1.2))
        ax.annotate('',xy=(source[0],source[1]+.45),
                    xytext=(source[0],source[1]+.12),
                    arrowprops=dict(arrowstyle='->',color='C0',lw=1.2))
        ax.annotate('',xy=(source[0],source[1]-.45),
                    xytext=(source[0],source[1]-.12),
                    arrowprops=dict(arrowstyle='->',color='C0',lw=1.2))
        # Qualitative outer-flow streamlines, not a computed potential solution.
        xx=np.linspace(-.22,1.65,260)
        for yy in [.14,.23,.34,-.16,-.25,-.34]:
            offset=(.075 if yy>0 else -.035)*np.exp(-((xx-.38)/.4)**2)
            streamline=yy+offset-.018*(np.tanh(5*(xx-.65))+1)
            ax.plot(xx,streamline,color=foreground,alpha=.23,lw=.9,zorder=0)
            i=45
            ax.annotate('',xy=(xx[i+9],streamline[i+9]),xytext=(xx[i],streamline[i]),
                        arrowprops=dict(arrowstyle='->',lw=.8,color=foreground,alpha=.35))
        ax.annotate(r'$U_\infty$',xy=(-.02,.06),xytext=(-.23,.06),ha='center',arrowprops=dict(arrowstyle='->',color=foreground))
        style=dict(arrowstyle='-',color=foreground,lw=.8)
        ax.annotate('Subsonic outer flow\nElliptic potential-flow model',xy=(.18,.38),xytext=(.2,.72),ha='center',arrowprops=style)
        ax.annotate('Supersonic pocket\nHyperbolic potential-flow model',xy=point(.4,.19),xytext=(.81,.57),ha='center',arrowprops=style)
        ax.annotate('Shock',xy=point(.63,cap[-1]-.025),xytext=(.85,.4),ha='center',arrowprops=style)
        ax.annotate('Attached boundary layer\nParabolic marching model',xy=point(.38,-.075),xytext=(.25,-.47),ha='center',arrowprops=style)
        ax.annotate('Unsteady separation /\nturbulent wake',xy=point(1.12,.03),xytext=(1.49,-.57),ha='center',arrowprops=style)
        ax.text(1.45,.47,'Acoustic propagation\nHyperbolic wave model',ha='center',color='C0')
        for label in ax.texts:
            if label.get_text():
                label.set_bbox(dict(facecolor=plt.rcParams['axes.facecolor'],edgecolor='none',pad=2,alpha=.95))
        ax.set(xlim=(-.6,1.85),ylim=(-.72,.92),aspect='equal')
        ax.axis('off')
        fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
        return fig

if __name__=='__main__':
    for ext in ['png','svg']:
        save_theme_variants(draw,ROOT/'notebooks/media'/f'transonic-airfoil.{ext}',dpi=300)
