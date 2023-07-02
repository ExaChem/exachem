##python *.py [#occ] [#vir] [#Cholesky] [#iter in GF] [time in ground state CCSD] [time in GFCCSD]

import numpy as np
import sys

O = int(sys.argv[1])
V = int(sys.argv[2])
C = int(sys.argv[3])
niter = int(sys.argv[4])
Time_GS  = float(sys.argv[5])
Time_GF  = float(sys.argv[6])

diis_step = 10
Flops = 0

ndiis = niter/diis_step

O2 = O*O
O3 = O2*O
O4 = O2*O2
V2 = V*V

Flops_GS = O*V + O2*V*C + 2*O2*V2*C +\
           4*O*V*C + 4*O2*V2*C + 2*O2*C + O2*V + O*V*C +\
           2*O2*V*C + 2*O2*V + 2*O*V2*C + 3*O*V*C + 2*O2*V*C +\
           2*O2*C + O2*V*C + O2*C + O*V2 +\
           4*O2*V2*C + 2*O2*V*C + 5*O*V*C + 2*V2*C + 2*O*V2 + 4*O2*C +\
           2*V2*C + 2*O*V2*C + 2*O2*C + 2*O3*C + 2*O4*C + 4*O2*V2*C +\
           2*O2*V*C + 2*O*V*C + 2*O2*V2*C + 2*V2*V2*C + 2*O2*V2 +\
           4*O4*V2 + 13*O3*V2*V + 2*V2 + 2*O2 + 2*O2*V + 6*O2*V2*V +\
           2*O2*V2

print Flops_GS
print 'CD-CCSD Flops/iteration: ',(float(Flops_GS)/1.0e+12)/(Time_GS*60)

Flops_GF = 2*(O2 + O2*V + O3*V) + \
        5*O3*V + 15*O3*V2 + 4*O2*V + 2*O2*V2 + 2*O4*V + 2*O4 +\
        2*O3*V + 5*O4*V + 4*O3 + 2*O3*V + 4*O2*V2

Flops_GF += 3*O + 4*O2*V
Flops_GF += 2*O + 4*O2*V
Flops_GF += O2 + 2*O2*V + O + 2*O2*V + 2*O + 4*O2*V
Flops_GF += 2*O + 4*O2*V
Flops_GF = niter*Flops_GF

Flops_diis = diis_step * (2*O + 5*O2*V)

if ndiis > 0:
  Flops_GF += ndiis*Flops_diis

print float(Flops)/1.0e+12
TFlops = (float(Flops_GF)/1.0e+12)/Time_GF

print Flops,TFlops