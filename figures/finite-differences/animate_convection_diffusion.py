"""Forward Euler comparison; reproduce with the course Python from any directory."""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style, WIDTH_MM
OUT=ROOT/'notebooks/media'
OUT.mkdir(parents=True,exist_ok=True)
N=40; V=1.; GAMMA=.01; DT=.001; STEPS=3000; STRIDE=20
DX=1/N; C=V*DT/DX; D=GAMMA*DT/DX**2
x=np.arange(N)*DX
initial=np.sin(2*np.pi*x)
profiles={}
errors={}
# Check all discrete Fourier modes for the actual Euler amplification factors.
theta=2*np.pi*np.arange(N)/N
for scheme in ['central','upwind']:
    amplification=1-4*D*np.sin(theta/2)**2-1j*C*np.sin(theta)
    if scheme=='upwind': amplification-=C*(1-np.cos(theta))
    assert np.max(np.abs(amplification))<=1+1e-12
    y=initial.copy(); frames=[y.copy()]
    for step in range(1,STEPS+1):
        rhs=GAMMA*(np.roll(y,-1)-2*y+np.roll(y,1))/DX**2
        rhs-=V*(np.roll(y,-1)-np.roll(y,1))/(2*DX) if scheme=='central' else V*(y-np.roll(y,1))/DX
        y=y+DT*rhs
        assert np.all(np.isfinite(y)) and abs(y.mean())<1e-12
        if step%STRIDE==0: frames.append(y.copy())
    # Independent spectral solution of this fully discrete single-mode problem.
    oracle=np.imag(np.exp(2j*np.pi*x)*amplification[1]**STEPS)
    assert np.max(np.abs(y-oracle))<1e-12
    profiles[scheme]=np.array(frames)
    exact=np.exp(-4*np.pi**2*GAMMA*STEPS*DT)*np.sin(2*np.pi*(x-V*STEPS*DT))
    errors[scheme]=float(np.max(np.abs(y-exact)))
times=np.arange(len(profiles['central']))*STRIDE*DT
np.savez(OUT/'fd-euler-comparison-data.npz',x=x,times=times,**profiles)
(OUT/'fd-euler-comparison-results.json').write_text(json.dumps(dict(N=N,v=V,gamma=GAMMA,dt=DT,steps=STEPS,C=C,D=D,final_max_nodal_error=errors,checks=['all Fourier modes stable','discrete mean preserved','independent Fourier evolution agrees within 1e-12']),indent=2)+'\n')
for mode in ['light','dark']:
    with plt.style.context(['default']+(['dark_background'] if mode=='dark' else [])),plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.6))
        fig.patch.set_facecolor('#111111' if mode=='dark' else 'white'); ax.set_facecolor(fig.get_facecolor())
        dense=np.linspace(0,1,500)
        exact_line,=ax.plot(dense,np.sin(2*np.pi*dense),color=plt.rcParams['text.color'],lw=1.8,label='Analytical solution')
        lines={}
        for scheme,color,marker in [('central','C0','o'),('upwind','C1','s')]:
            lines[scheme],=ax.plot(np.r_[x,1],np.r_[initial,initial[0]],color=color,marker=marker,markevery=2,ms=3,lw=1.2,label='Forward Euler + '+scheme+' convection')
        ax.set(xlim=(0,1),ylim=(-1.15,1.15),xlabel='$x$',ylabel=r'$\varphi$')
        ax.spines[['top','right']].set_visible(False)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),frameon=False,fontsize=9,ncol=1,labelspacing=.2,borderaxespad=0)
        title=ax.set_title('',fontsize=10)
        fig.text(.5,.02,r'$N=40,\ v=1,\ \Gamma=0.01,\ \Delta t=0.001,\ C=0.04,\ D=0.016$',ha='center',fontsize=9)
        fig.subplots_adjust(left=.1,right=.98,bottom=.2,top=.78)
        def update(j):
            t=times[j]
            exact_line.set_ydata(np.exp(-4*np.pi**2*GAMMA*t)*np.sin(2*np.pi*(dense-V*t)))
            for scheme,line in lines.items():
                y=profiles[scheme][j];line.set_ydata(np.r_[y,y[0]])
            title.set_text('$t=%.2f$'%t)
        animation=FuncAnimation(fig,update,frames=len(times),interval=100)
        animation.save(OUT/f'fd-euler-comparison-{mode}.mp4',writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-pix_fmt','yuv420p']),dpi=150)
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(OUT/f'fd-euler-comparison-{mode}.mp4'),'-c:v','libvpx-vp9','-b:v','0','-crf','32',str(OUT/f'fd-euler-comparison-{mode}.webm')],check=True)
        for j in [25,len(times)-1]:
            update(j);fig.savefig(OUT/f'fd-euler-comparison-t{j}-{mode}.png',dpi=150)
        plt.close(fig)
print(json.dumps(errors))
