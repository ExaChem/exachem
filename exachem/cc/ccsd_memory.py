import sys
import math
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

#python ccsd_memory.py -oa 99 -ob 94 -va 394 -vb 399 -cv 4027 -ctype uhf -diis 5 -nranks 10 -cache 8 -ts 32

def parseargs(argv=None):

    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    #num_cores = psutil.cpu_count(logical=False)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-oa', '--o_alpha',  type=int, required=True, help='#occupied alpha')
    parser.add_argument('-ob', '--o_beta' ,  type=int, required=True, help='#occupied beta')
    parser.add_argument('-va', '--v_alpha',  type=int, required=True, help='#virtual alpha')
    parser.add_argument('-vb', '--v_beta' ,  type=int, required=True, help='#virtual beta')
    parser.add_argument('-cv', '--cv_count', type=int, required=True, help='#cholesky vectors')

    parser.add_argument('-ctype',  '--scf_type',   type=str, default="rhf", help='RHF/UHF')
    parser.add_argument('-diis',   '--diis'    ,   type=int, default=5,  help='#diis')
    parser.add_argument('-nranks', '--nranks'  ,   type=int, default=1,  help='#mpi ranks')
    parser.add_argument('-cache',  '--cache_size', type=int, default=8,  help='(T) cache size')
    parser.add_argument('-ts', '--ccsdt_tilesize', type=int, default=32, help='(T) tilesize')
    
    # Process arguments
    args = parser.parse_args()
    return args

cc_args = parseargs()

o_alpha=cc_args.o_alpha
o_beta =cc_args.o_beta
v_alpha=cc_args.v_alpha
v_beta =cc_args.v_beta
CI     =cc_args.cv_count

ndiis = cc_args.diis
nranks = cc_args.nranks
cache_size = cc_args.cache_size
ccsdt_tilesize = cc_args.ccsdt_tilesize

print("\nArgs passed:")
print(cc_args)
print("")

is_uhf=(cc_args.scf_type == "uhf")
if is_uhf: print("UHF calculation")
else: print("RHF calculation")


V = v_alpha + v_beta
O = o_alpha + o_beta

ccsd_mem = 0

#setup tensors
d_r1 = v_alpha*o_alpha #r1_aa
d_r2 = v_alpha*v_beta*o_alpha*o_beta #r2_abab

r1_vo = v_alpha*o_alpha + v_beta*o_beta
#{"aaaa", "abab", "bbbb"}
r2_vvoo = v_alpha*v_alpha*o_alpha*o_alpha+v_beta*o_beta*v_beta*o_beta+v_alpha*v_beta*o_alpha*o_beta

t1_vo = r1_vo
t2_vvoo = r2_vvoo

#UHF
if is_uhf:
    d_r1 = r1_vo
    #add "abba", "baab", "baba"
    d_r2 = r2_vvoo + v_alpha*v_beta*o_beta*o_alpha + v_beta*v_alpha*o_alpha*o_beta \
            +v_beta*v_alpha*o_beta*o_alpha

d_t1 = d_r1 #t1_aa
d_t2 = d_r2 #t2_abab

d_r1s = ndiis * d_r1
d_r2s = ndiis * d_r2

d_t1s = ndiis * d_r1
d_t2s = ndiis * d_r2


#RHF
r1_aa   = d_r1
r2_abab = d_r2
t1_aa   = d_t1
t2_abab = d_t2
t2_aaaa = v_alpha*v_alpha*o_alpha*o_alpha
i0_temp = v_beta*v_alpha*o_beta*o_alpha
t2_aaaa_temp = t2_aaaa

_a02  = o_alpha*o_alpha*CI #{O, O, CI}, "aa"
_a03  = o_alpha*v_alpha*CI #{O, V, CI}, "aa"

if is_uhf:
    _a02 += o_beta*o_beta*CI
    _a03 += o_beta*v_beta*CI

# {MO,MO}
d_f1 = (o_alpha*(o_alpha+v_alpha)+o_beta*(o_beta+v_beta) \
         + v_alpha*(o_alpha+v_alpha)+v_beta*(o_beta+v_beta))

f1_oo = (o_alpha*o_alpha + o_beta*o_beta)  
f1_ov = (o_alpha*v_alpha + o_beta*v_beta) 
f1_vv = (v_alpha*v_alpha + v_beta*v_beta)
f1_vo = f1_ov

#cv3d {MO,MO,CI}
cv3d = CI*d_f1

#chol3d_oo, chol3d_ov, chol3d_vv
chol3d_oo = CI*f1_oo  #{O, O, CI}, {"aa", "bb"}
chol3d_ov = CI*f1_ov  #{O, V, CI}, {"aa", "bb"}
chol3d_vv = CI*f1_vv  #{V, V, CI}, {"aa", "bb"}
chol3d_vo = chol3d_ov #{V, O, CI}, {"aa", "bb"}

# skipping f1_oo, f1_ov, f1_vv 
ccsd_mem += d_f1 + f1_oo + f1_ov + f1_vv + t1_aa + t2_abab + r1_aa + r2_abab + cv3d \
    + chol3d_oo + chol3d_ov + chol3d_vv +_a02 +_a03

if not is_uhf: ccsd_mem += t2_aaaa + i0_temp +t2_aaaa_temp
else:
    ccsd_mem += r1_vo + r2_vvoo + t1_vo + t2_vvoo + f1_vo + chol3d_vo

# DIIS
ccsd_mem += d_r1s + d_r2s + d_t1s + d_t2s

# Intermediates
_a04  = o_alpha*o_alpha #{O, O},{"aa"}
_a05  = f1_ov #{O, V},{"aa", "bb"}
_a01  = o_alpha*o_alpha*CI #{O, O, CI},{"aa"}
_a06  = v_alpha*o_alpha*CI #{V, O, CI},{"aa"}
_a001 = v_alpha*v_alpha + v_beta*v_beta #{V, V}, {"aa", "bb"}
_a006 = o_alpha*o_alpha + o_beta*o_beta #{O, O}, {"aa", "bb"}
_a004 = v_alpha*v_alpha*o_alpha*o_alpha + v_alpha*v_beta*o_alpha*o_beta #{V, V, O, O}, {"aaaa", "abab"}

_a008 = o_alpha*o_alpha*CI #{O, O, CI}, {"aa"}
_a009 = o_alpha*o_alpha*CI + o_beta*o_beta*CI #{O, O, CI}, {"aa", "bb"}
_a017 = v_alpha*o_alpha*CI + v_beta*o_beta*CI #{V, O, CI}, {"aa", "bb"}
_a021 = v_alpha*v_alpha*CI + v_beta*v_beta*CI #{V, V, CI}, {"aa", "bb"}

_a019 =  o_alpha*o_beta*o_alpha*o_beta #{O, O, O, O}, "_a019", {"abab"}
#{V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}
_a020 =  v_alpha*o_alpha*v_alpha*o_alpha + v_beta*o_alpha*v_beta*o_alpha \
        +v_beta*o_alpha*v_alpha*o_beta + v_beta*o_beta*v_beta*o_beta

if is_uhf:
    _a04 += o_beta*o_beta
    _a01 += o_beta*o_beta*CI 
    _a06 += v_beta*o_beta*CI
    _a004 += v_beta*v_beta*o_beta*o_beta
    
    _a008 += o_beta*o_beta*CI
    _a019 += o_alpha*o_alpha*o_alpha*o_alpha+o_beta*o_beta*o_beta*o_beta
     #abab,abba
    _a020 += v_alpha*o_beta*v_alpha*o_beta +v_alpha*o_beta*v_beta*o_alpha

    #UHF {V, V, O, O}, {"aaaa", "bbbb"}
    i0_t2_tmp = v_alpha*v_alpha*o_alpha*o_alpha + v_beta*v_beta*o_beta*o_beta
    ccsd_mem += i0_t2_tmp

ccsd_mem += _a01 + _a04 + _a05 + _a06 + _a001 + _a004 + _a006 \
            + _a008 + _a009 + _a017 + _a019 +_a020 + _a021

gib=1024*1024*1024.0

ccsd_mem = round(ccsd_mem*8/gib,2) #bytes
print("Total CPU memory required for CCSD calculation: " + str(ccsd_mem) + " GiB")

v4_mem = v_alpha*v_alpha*v_beta*v_beta
if is_uhf: v4_mem += v_alpha*v_alpha*v_alpha*v_alpha+v_beta*v_beta*v_beta*v_beta
v4_mem *= 8.0 #bytes
v4_mem = round(v4_mem/gib,2)

# print(" - Storage required V^4 intermediate: " + str(v4_mem) + " GiB")
# old_mem = v4_mem+ccsd_mem
# print(" - Total CPU memory required if V^4 intermediate is stored: " + str(old_mem) + " GiB")


# (T)
ccsdt_tilesize = ccsdt_tilesize*1.0

d_t1 = r1_vo
#add "abba", "baab", "baba"
d_t2 = r2_vvoo + v_alpha*v_beta*o_beta*o_alpha + v_beta*v_alpha*o_alpha*o_beta \
        +v_beta*v_alpha*o_beta*o_alpha

ccsd_t_mem = d_f1 + d_t1 + d_t2

skip_ccsd=False

if not skip_ccsd:
    ccsd_t_mem += d_t1 + d_t2
    # retiling allocates full GA versions of the t1,t2 tensors.
    ccsd_t_mem += O * V + V * V * O * O    


def ft_mem(i,j,k,l) :
    #"aaaa", "abab", "bbbb", "abba", "baab", "baba"
    b1=1;b2=1;b3=1;b4=1;b5=1;b6=1
    b1*= o_alpha if (i==O) else v_alpha 
    b2*= o_alpha if (i==O) else v_alpha 
    b3*= o_beta  if (i==O) else v_beta 
    b4*= o_alpha if (i==O) else v_alpha 
    b5*= o_beta  if (i==O) else v_beta 
    b6*= o_beta  if (i==O) else v_beta 

    b1*= o_alpha if (j==O) else v_alpha 
    b2*= o_beta  if (j==O) else v_beta  
    b3*= o_beta  if (j==O) else v_beta  
    b4*= o_beta  if (j==O) else v_beta  
    b5*= o_alpha if (j==O) else v_alpha 
    b6*= o_alpha if (j==O) else v_alpha 

    b1*= o_alpha if (k==O) else v_alpha 
    b2*= o_alpha if (k==O) else v_alpha 
    b3*= o_beta  if (k==O) else v_beta
    b4*= o_beta  if (k==O) else v_beta 
    b5*= o_alpha if (k==O) else v_alpha 
    b6*= o_beta  if (k==O) else v_beta 

    b1*= o_alpha if (l==O) else v_alpha 
    b2*= o_beta  if (l==O) else v_beta
    b3*= o_beta  if (l==O) else v_beta 
    b4*= o_alpha if (l==O) else v_alpha 
    b5*= o_beta  if (l==O) else v_beta 
    b6*= o_alpha if (l==O) else v_alpha

    return b1+b2+b3+b4+b5+b6

# V2: "ijab", "ijka", "iabc"
ccsd_t_mem += ft_mem(O,O,V,V)+ft_mem(O,O,O,V)+ft_mem(O,V,V,V)
ccsd_t_mem = (ccsd_t_mem * 8.0) / gib

noa = math.ceil(o_alpha/ccsdt_tilesize)
nob = math.ceil(o_beta/ccsdt_tilesize)
nva = math.ceil(v_alpha/ccsdt_tilesize)
nvb = math.ceil(v_beta/ccsdt_tilesize)
# print (noa,nob,nva,nvb)

# Can be < ccsdt_tilesize for small problems
max_hdim = ccsdt_tilesize
max_pdim = ccsdt_tilesize

max_d1_kernels_pertask = 9 * noa
max_d2_kernels_pertask = 9 * nva
size_T_s1_t1           = 9 * (max_pdim) * (max_hdim)
size_T_s1_v2           = 9 * (max_pdim * max_pdim) * (max_hdim * max_hdim)
size_T_d1_t2 = max_d1_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim)
size_T_d1_v2 = max_d1_kernels_pertask * (max_pdim) * (max_hdim * max_hdim * max_hdim)
size_T_d2_t2 = max_d2_kernels_pertask * (max_pdim * max_pdim) * (max_hdim * max_hdim)
size_T_d2_v2 = max_d2_kernels_pertask * (max_pdim * max_pdim * max_pdim) * (max_hdim)

extra_buf_mem_per_rank = size_T_s1_t1 + size_T_s1_v2 + size_T_d1_t2 + size_T_d1_v2 + size_T_d2_t2 + size_T_d2_v2
extra_buf_mem_per_rank     = extra_buf_mem_per_rank * 8 / gib
total_extra_buf_mem = extra_buf_mem_per_rank * nranks

cache_buf_size = ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * 8 # bytes
cache_mem_per_rank  = (ccsdt_tilesize * ccsdt_tilesize * 8 + cache_buf_size) * cache_size # s1 t1+v2
cache_mem_per_rank += (noa+nob + nva+nvb) * 2 * cache_size * cache_buf_size # d1,d2 t2+v2
cache_mem_per_rank  = cache_mem_per_rank / gib
total_cache_mem = cache_mem_per_rank * nranks # GiB

ccsd_t_mem = round(ccsd_t_mem,2)
total_cache_mem = round(total_cache_mem,2)
total_extra_buf_mem = round(total_extra_buf_mem,2)

total_ccsd_t_mem = ccsd_t_mem + total_extra_buf_mem + total_cache_mem

print("Total CPU memory required for (T) calculation: " + str(total_ccsd_t_mem) + " GiB")
print("-- memory required for the input tensors: " + str(ccsd_t_mem) + " GiB")
print("-- memory required for intermediate buffers: " + str(total_extra_buf_mem) + " GiB")
cache_msg = "-- memory required for caching t1,t2,v2 blocks"
if total_cache_mem > (ccsd_t_mem + total_extra_buf_mem) / 2.0:
    cache_msg += " (set cache_size option to a lower value to reduce this " \
                     "memory requirement further)"
print(cache_msg + ": " + str(total_cache_mem) + " GiB")

T=ccsdt_tilesize
nbf=o_alpha+v_alpha
#Unfused: mem=9*T*T*T*T*(noab+nvab)*16
t_gpu_mem=9*T*T*T*T*nbf*16
# Fused: 9 . (T^2 + T^4 + 2.T^3.2N) . 8 bytes
t_gpu_mem=9*(T*T + T*T*T*T + 2*2*nbf*T*T*T) * 8
t_gpu_mem=str(round(t_gpu_mem/gib,2))

print("(T): memory required on a single gpu = " + t_gpu_mem + " GiB")