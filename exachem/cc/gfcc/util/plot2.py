import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

au2ev = 27.21139

filename =       sys.argv[1]
savename =       sys.argv[2]
_xmin    = -2
_xmax    = 0
_ymax    =   int(sys.argv[3])
_yinter  =   int(sys.argv[4])
_label   = 'GFCCSD'

mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
plt.rc('font', size=10,weight='bold')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(_xmin,_xmax)
ax.set_ylim(-5,_ymax)
major_yticks = np.arange(0, _ymax+100, _yinter)
ax.set_yticks(major_yticks)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.grid(True,which='major', color='k', linestyle='--')
ax.tick_params(direction='in',length=6, width=2)

ax.set_xlabel(r'$\omega$/eV',fontsize=16,fontweight='bold')
ax.set_ylabel(r'$A(\omega)$/eV$^{-1}$',fontsize=16,fontweight='bold')

f = open(filename, "r")
data1 = np.loadtxt(f)
npts  = data1.shape[0]
ordering = np.argsort(data1[:,0])
freq = data1[ordering,0]
Atot = data1[ordering,1]
A1   = data1[ordering,2]
A2   = data1[ordering,3]
ax.plot(freq,Atot,'k-')
ax.plot(freq,A1,'g-')
ax.plot(freq,A2,'b-')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,fontsize=12,loc="upper left",fancybox=True, shadow=True)
plt.tight_layout()

plt.savefig(savename,format='pdf', bbox_inches='tight')

plt.draw()
plt.show()
