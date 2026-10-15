"""Euler characteristic variables synchronized with dependence paths to a fixed event."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM
OUT=ROOT/'notebooks/media'
SPEEDS=np.array([-.6,.4,1.4]);XP=.4;TP=1.

def modes(x,t):
    # Broader profiles make the contributions arriving at P all visible.
    return .1*np.exp(-((np.asarray(x)[None,:]-SPEEDS[:,None]*t)/1.2)**2)


def animate(mode):
    with plt.style.context(['default']+(['dark_background'] if mode=='dark' else [])),plt.rc_context(figure_style()):
        fig=plt.figure(figsize=(WIDTH_MM['double']/25.4,5.6))
        fig.patch.set_facecolor('#111111' if mode=='dark' else 'white')
        gs=fig.add_gridspec(3,2,width_ratios=[1.2,1],left=.12,right=.98,bottom=.13,top=.83,wspace=.42,hspace=.32)
        axes=[fig.add_subplot(gs[j,0]) for j in range(3)]
        diagram=fig.add_subplot(gs[:,1]);color=plt.rcParams['text.color']
        x=np.linspace(-3,4,550);initial=modes(x,0)
        curves=[];markers=[];profile_dots=[]
        for j,ax in enumerate(axes):
            ax.set_facecolor(fig.get_facecolor());ax.plot(x,initial[j],color=color,ls=':',alpha=.25,lw=.9)
            line,=ax.plot(x,initial[j],color=f'C{j}',lw=1.7);curves.append(line)
            ax.set(xlim=(-3,4),ylim=(-.005,.115),ylabel=r'$w_%d$'%(j+1))
            ax.spines[['top','right']].set_visible(False)
            if j<2: ax.tick_params(labelbottom=False)
            marker=ax.axvline(XP-SPEEDS[j]*TP,color=f'C{j}',ls='--',lw=.8,alpha=.7)
            ax.axvline(XP,color=color,ls='-.',lw=.9,alpha=.55,zorder=0)
            ax.text(XP+.10,.94,r'$x_P$',transform=ax.get_xaxis_transform(),ha='left',va='top',fontsize=9,color=color)
            markers.append(marker)
            dot,=ax.plot([],[],'o',color=f'C{j}',ms=5)
            profile_dots.append(dot)
        axes[-1].set_xlabel('$x$')
        axes[0].set_title('Characteristic variables',fontsize=10)
        diagram.set_facecolor(fig.get_facecolor())
        diagram.set(xlim=(-1.4,1.6),ylim=(-.05,1.14),xlabel='$x$',ylabel='$t$')
        diagram.spines[['top','right']].set_visible(False)
        diagram.set_title('Paths carrying information to $P$',fontsize=10)
        diagram.plot(XP,TP,'o',color=color,ms=4);diagram.text(XP+.08,TP+.015,'$P$',color=color)
        paths=[];dots=[]
        for j,speed in enumerate(SPEEDS):
            foot=XP-speed*TP
            diagram.plot(foot,0,'o',color=f'C{j}',ms=3)
            line,=diagram.plot([],[],color=f'C{j}',lw=1.5,label=r'$\lambda_%d=%+.1f$'%(j+1,speed));paths.append(line)
            dot,=diagram.plot([],[],'o',color=f'C{j}',ms=4);dots.append(dot)
        cursor=diagram.axhline(0,color=color,ls=':',lw=.8)
        patch=[None]
        profile_shading=[None]*3
        fig.legend(handles=paths,loc='upper center',bbox_to_anchor=(.56,.94),ncol=3,frameon=False,fontsize=9)
        clock=fig.text(.55,.99,'',ha='center',va='top')
        def update(t):
            values=modes(x,t)
            for j,line in enumerate(curves): line.set_ydata(values[j])
            left=XP+SPEEDS[2]*(t-TP)
            right=XP+SPEEDS[0]*(t-TP)
            for j,ax in enumerate(axes):
                if profile_shading[j] is not None: profile_shading[j].remove()
                profile_shading[j]=ax.axvspan(left,right,color='C0',alpha=.1,zorder=-1)
            tt=np.linspace(0,t,100)
            for k,speed in enumerate(SPEEDS):
                xx=XP+speed*(tt-TP);paths[k].set_data(xx,tt)
                current=XP+speed*(t-TP);dots[k].set_data([current],[t])
                markers[k].set_xdata([current,current])
                profile_dots[k].set_data([current],[modes(np.array([current]),t)[k,0]])
            if patch[0] is not None: patch[0].remove()
            patch[0]=diagram.fill_betweenx(tt,XP+SPEEDS[2]*(tt-TP),XP+SPEEDS[0]*(tt-TP),color='C0',alpha=.1,zorder=0)
            cursor.set_ydata([t,t]);clock.set_text(r'$t=%.2f,\quad P=(0.4,1),\quad\bar u=0.4,\quad\bar c=1$'%t)
        anim=FuncAnimation(fig,update,frames=np.r_[np.linspace(0,TP,64),np.full(15,TP)],interval=100)
        anim.save(OUT/f'euler-dependence-build-{mode}.mp4',writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        update(.7);fig.savefig(OUT/f'euler-dependence-build-{mode}.png',dpi=150)
        fig.savefig(OUT/f'euler-dependence-build-{mode}.svg',transparent=True)
        plt.close(fig)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    for t in [0,.3,1]:
        xx=XP+SPEEDS*(t-TP)
        values=np.array([modes(np.array([xx[j]]),t)[j,0] for j in range(3)])
        assert np.allclose(values,modes(np.array([XP]),TP)[:,0])
    for mode in ['light','dark']: animate(mode)
