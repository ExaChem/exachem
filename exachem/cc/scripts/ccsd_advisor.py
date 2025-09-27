import sys
import math
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

#python ccsd_advisor.py -oa 99 -ob 94 -va 394 -vb 399 -cv 4027 -ppn 4 -ram 512 -gpn 4 -gram 40 -ctype uhf -diis 5 -nnodes_t 10 -cache 8 -ts 40

# nnodes_t: Optionally specify number of nodes for (T) to check if the calculation fits in memory.
# If not specified, minimum number of nodes required for (T) is printed. Note that the (T) memory requirements vary with the number of nodes (nnodes_t), number of processes per node (ppn), the tilesize (ts) and the cache size (cache).

def get_mo_tiles(noa,nob,nva,nvb,ts):
    est_nt = math.ceil(1.0 * noa / ts)
    mo_tiles = []
    for x in range(0, est_nt):
      mo_tiles.append(int(noa / est_nt + (x < (noa % est_nt))))

    # est_nt = math.ceil(1.0 * nob / ts)
    # for x in range(0, est_nt):
    #   mo_tiles.append(int(nob / est_nt + (x < (nob % est_nt))))

    est_nt = math.ceil(1.0 * nva / ts)
    for x in range(0, est_nt):
       mo_tiles.append(int(nva / est_nt + (x < (nva % est_nt))))

    # est_nt = math.ceil(1.0 * nvb / ts)
    # for x in range(0, est_nt): 
    #    mo_tiles.append(int(nvb / est_nt + (x < (nvb % est_nt))))

    return mo_tiles


def parseargs(argv=None):

    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    #num_cores = psutil.cpu_count(logical=False)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-oa',  '--o_alpha',    type=int, required=True, help='#occupied alpha')
    parser.add_argument('-ob',  '--o_beta' ,    type=int, required=True, help='#occupied beta')
    parser.add_argument('-va',  '--v_alpha',    type=int, required=True, help='#virtual alpha')
    parser.add_argument('-vb',  '--v_beta' ,    type=int, required=True, help='#virtual beta')
    parser.add_argument('-cv',  '--cv_count',   type=int, required=True, help='#cholesky vectors')
    parser.add_argument('-ppn', '--ppn',        type=int, required=True, help='mpi ranks per node')
    parser.add_argument('-gpn', '--gpn',        type=int, default =None, help='number of GPUs per node')
    parser.add_argument('-ram', '--cpu_memory', type=int, required=True, help='CPU Memory per node in GiB')
    parser.add_argument('-gram','--gpu_memory', type=int, required=True, help='Memory per GPU in GiB')

    parser.add_argument('-ctype',     '--scf_type',       type=str,  default="rhf", help='RHF/UHF')
    parser.add_argument('-diis',      '--diis'    ,       type=int,  default=5,     help='#diis')
    parser.add_argument('-nnodes_t',  '--nnodes_t',       type=int,  default=None,  help='(T) number of nodes')
    parser.add_argument('-cache',     '--cache_size',     type=int,  default=8,     help='(T) cache size')
    parser.add_argument('-ts',        '--ccsdt_tilesize', type=int,  default=40,    help='(T) tilesize')
    parser.add_argument('-skip_ccsd', '--skip_ccsd',      type=bool, default=False, help='#skip_ccsd')

    # Process arguments
    args = parser.parse_args()

    # Post-process to set default
    if args.gpn is None:
        args.gpn = args.ppn
    
    return args

cc_args = parseargs()

o_alpha=cc_args.o_alpha
o_beta =cc_args.o_beta
v_alpha=cc_args.v_alpha
v_beta =cc_args.v_beta
CI     =cc_args.cv_count
ppn    =cc_args.ppn #ppn=num-gpus per node
gpn    =cc_args.gpn #gpn=ppn if not specified
cpu_mem=cc_args.cpu_memory
gpu_mem=cc_args.gpu_memory #memory per GPU


ndiis = cc_args.diis
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

nbf=o_alpha + v_alpha

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

#(MOxMOxCI + MOxAOxCI)
chol_mem = CI*(d_f1+ 2*nbf*nbf) 

#chol3d_oo, chol3d_ov, chol3d_vv
chol3d_oo = CI*f1_oo  #{O, O, CI}, {"aa", "bb"}
chol3d_ov = CI*f1_ov  #{O, V, CI}, {"aa", "bb"}
chol3d_vv = CI*f1_vv  #{V, V, CI}, {"aa", "bb"}
chol3d_vo = chol3d_ov #{V, O, CI}, {"aa", "bb"}

# skipping f1_oo, f1_ov, f1_vv 
ccsd_mem += d_f1 + f1_oo + f1_ov + f1_vv + t1_aa + t2_abab + r1_aa + r2_abab \
    + chol3d_oo + chol3d_ov + chol3d_vv +_a02 +_a03

if not is_uhf: ccsd_mem += t2_aaaa + i0_temp +t2_aaaa_temp
else:
    ccsd_mem += r1_vo + r2_vvoo + t1_vo + t2_vvoo + f1_vo + chol3d_vo

# DIIS + cv3d // these are not allocated in the tamm standalone test
ccsd_mem += cv3d + d_r1s + d_r2s + d_t1s + d_t2s

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

#Multiply by 2 to account for posix shared mem
ccsd_mem = round(2*ccsd_mem*8/gib,2) #bytes
chol_mem = round(2*chol_mem*8/gib,2) #bytes 

v4_mem = v_alpha*v_alpha*v_beta*v_beta
if is_uhf: v4_mem += v_alpha*v_alpha*v_alpha*v_alpha+v_beta*v_beta*v_beta*v_beta
v4_mem *= 8.0 #bytes
#Multiply by 2 to account for posix shared mem
v4_mem = round(2*v4_mem/gib,2)

# print("---------------------------------------------------------------------------")
# print(" - Storage required V^4 intermediate: " + str(v4_mem) + " GiB")
# old_mem = v4_mem+ccsd_mem
# print(" - Total CPU memory required if V^4 intermediate is stored: " + str(old_mem) + " GiB")
# print("---------------------------------------------------------------------------")

nnodes = math.ceil(ccsd_mem/cpu_mem)

nranks = nnodes*ppn

print("nbf: "    + str(nbf))

print("\nTotal CPU memory required for Cholesky decomp of the 2e integrals: " + str(chol_mem) + " GiB")
print("\nTotal CPU memory required for CCSD calculation: " + str(ccsd_mem) + " GiB\n")


VabOab = v_alpha*o_beta*v_beta*o_alpha
ts_guess=40
ts_max=ts_guess

def get_ts_recommendation(nranks):

    ts_max_    = 180
    ranks_per_gpu = ppn / gpn
    if(gpu_mem <= 8): ts_max_ = 100
    elif(gpu_mem <= 12): ts_max_ = 120
    elif(gpu_mem <= 16): ts_max_ = 130
    elif(gpu_mem <= 24): ts_max_ = 145
    elif(gpu_mem <= 32): ts_max_ = 155
    elif(gpu_mem <= 40): ts_max_ = 160
    elif(gpu_mem <= 48): ts_max_ = 170

    while( (6 * (math.pow(ts_max_, 4) * 8 / (1024 * 1024 * 1024.0)) * ranks_per_gpu) >=
          0.95 * gpu_mem):
      ts_max_ -= 5


    ts_guess_ = 40
    nblocks_  = 10
    tilesizes = list(range(ts_guess, ts_max_+1, 5))

    for ts in tilesizes:
        nblocks = math.ceil(v_alpha/ts) * math.ceil(o_alpha/ts) * math.ceil(v_beta/ts) * math.ceil(o_beta/ts)
        # print ("  --> MO Tiles for tilesize %s, nblocks=%s: %s" %(ts, nblocks, get_mo_tiles(o_alpha,o_beta,v_alpha,v_beta,ts)))
        ts_max_ = ts
        #nblocks <= nranks
        if (nblocks*1.0/nranks) < 0.31 or ts_max_ >= v_alpha+10 or nblocks==1:
            ts_max_ = ts_guess_
            break
        ts_guess_=ts
        nblocks_=nblocks

    return [ts_max_,nblocks_]

[ts_max,nblocks] = get_ts_recommendation(nranks)
print("Min #nodes required = %s, nranks = %s, nblocks = %s, max tilesize = %s" %(nnodes, nranks, nblocks, ts_max))
# print ("  --> MO Tiles = %s" %(get_mo_tiles(o_alpha,o_beta,v_alpha,v_beta,ts_max)))

nodecounts = list(range(nnodes+10, nnodes*10+1, 10))
for nc in nodecounts:
    # print ("-----------------------------------------------------------------------")
    [ts_max,nblocks] = get_ts_recommendation(nc*ppn)
    # if nblocks <= nc*ppn: break
    if (nblocks*1.0/nc*ppn) < 0.31 or ts_max >= v_alpha+10 or nblocks==1: break
    print("For node count = %s, nranks = %s, nblocks = %s, max tilesize = %s" %(nc, nc*ppn, nblocks, ts_max))
    # print ("  --> MO Tiles = %s" %(get_mo_tiles(o_alpha,o_beta,v_alpha,v_beta,ts_max)))


# (T)
ccsdt_tilesize = ccsdt_tilesize

d_t1 = r1_vo
#add "abba", "baab", "baba"
d_t2 = r2_vvoo + v_alpha*v_beta*o_beta*o_alpha + v_beta*v_alpha*o_alpha*o_beta \
        +v_beta*v_alpha*o_beta*o_alpha

ccsd_t_mem = d_f1 + d_t1 + d_t2

skip_ccsd=cc_args.skip_ccsd

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

noa = o_alpha//ccsdt_tilesize + (1 if o_alpha % ccsdt_tilesize!=0 else 0)
nob = o_beta//ccsdt_tilesize + (1 if o_beta % ccsdt_tilesize!=0 else 0)
nva = v_alpha//ccsdt_tilesize + (1 if v_alpha % ccsdt_tilesize!=0 else 0)
nvb = v_beta//ccsdt_tilesize + (1 if v_beta % ccsdt_tilesize!=0 else 0)

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

# base (T) memory required for input tensors
ccsd_t_mem_base = 2*round(ccsd_t_mem,2) # 2 for posix shared mem

# Calculate per-rank memory requirements (independent of nranks_t)
extra_buf_mem_per_rank = size_T_s1_t1 + size_T_s1_v2 + size_T_d1_t2 + size_T_d1_v2 + size_T_d2_t2 + size_T_d2_v2
extra_buf_mem_per_rank = extra_buf_mem_per_rank * 8 / gib

cache_buf_size = ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * ccsdt_tilesize * 8 # bytes
cache_mem_per_rank  = (ccsdt_tilesize * ccsdt_tilesize * 8 + cache_buf_size) * cache_size # s1 t1+v2
cache_mem_per_rank += (noa+nob + nva+nvb) * 2 * cache_size * cache_buf_size # d1,d2 t2+v2
cache_mem_per_rank  = cache_mem_per_rank / gib

def get_min_nodes_req(total_ccsd_t_mem):
    nnodes_req = (total_ccsd_t_mem//cpu_mem) + (1 if total_ccsd_t_mem % cpu_mem > 0 else 0)
    return int(nnodes_req)

def get_ccsd_t_mem(nranks_t):
    total_extra_buf_mem = extra_buf_mem_per_rank * nranks_t
    total_cache_mem = cache_mem_per_rank * nranks_t
    total_ccsd_t_mem = ccsd_t_mem_base + total_extra_buf_mem + total_cache_mem
    return total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem

def check_t_mem_requirement(nnodes_given,nranks_t_new):
    total_ccsd_t_mem,_,_ = get_ccsd_t_mem(nranks_t_new)
    nnodes_required_real = get_min_nodes_req(total_ccsd_t_mem)
    # print("nnodes_required_real, given: ", nnodes_required_real, nnodes_given)
    if nnodes_required_real <= nnodes_given: return True
    return False

# Auto-determine minimum nranks_t
def determine_minimum_nranks_t():
    # Start with minimum estimate based on input tensors only
    nnodes_min = get_min_nodes_req(ccsd_t_mem_base) 
    nranks_t = nnodes_min * ppn
    
    # Update nranks_t until a suitable number of nodes satisfying the memory requirements is found
    max_iterations = 100
    for i in range(max_iterations):
        # Calculate total memory with current nranks_t
        total_ccsd_t_mem,_,_ = get_ccsd_t_mem(nranks_t)

        # Calculate required nodes and ranks
        nnodes_required = get_min_nodes_req(total_ccsd_t_mem)
        nranks_t_new = nnodes_required * ppn
        
        # Check if the collective memory on the nodes are actually sufficient
        if check_t_mem_requirement(nnodes_required,nranks_t_new):
            break
                
        nranks_t = nranks_t_new
    
    return nranks_t

print("\n\n(T) Memory Analysis")
print("-"*50)
print("Given tilesize, cache_size = " + str(ccsdt_tilesize) + ", " + str(cache_size))

T=ccsdt_tilesize
nbf=o_alpha+v_alpha
#Unfused: mem=9*T*T*T*T*(noab+nvab)*16
t_gpu_mem=9*T*T*T*T*nbf*16
# Fused: 9 . (T^2 + T^4 + 2.T^3.2N) . 8 bytes
t_gpu_mem=9*(T*T + T*T*T*T + 2*2*nbf*T*T*T) * 8
t_gpu_mem=str(round(t_gpu_mem/gib,2))

print("- Memory required on a single gpu = " + t_gpu_mem + " GiB\n")

total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem = 0,0,0

def print_ccsd_memory_stats(nnodes_t,total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem):
    total_cache_mem = round(total_cache_mem,2)
    total_extra_buf_mem = round(total_extra_buf_mem,2)
    total_ccsd_t_mem = round(total_ccsd_t_mem,2)

    print(f"\nTotal CPU memory required for (T) calculation on {nnodes_t} nodes: {total_ccsd_t_mem} GiB")
    print(f"-- memory required for the input tensors: {ccsd_t_mem_base} GiB")
    print(f"-- memory required for intermediate buffers: {total_extra_buf_mem} GiB")
    cache_msg = "-- memory required for caching t1,t2,v2 blocks"
    if total_cache_mem > (ccsd_t_mem_base + total_extra_buf_mem) / 2.0:
        cache_msg += " (set cache_size option to a lower value to reduce this " \
                        "memory requirement further)"
    print(f"{cache_msg}: {total_cache_mem} GiB\n")

nnodes_t = cc_args.nnodes_t

if nnodes_t is not None:
    nranks_t = nnodes_t * ppn
    total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem = get_ccsd_t_mem(nranks_t)
    print(f"Given #nodes for (T) = {nnodes_t}")
    print_ccsd_memory_stats(nnodes_t,total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem)
    if nnodes_t < get_min_nodes_req(total_ccsd_t_mem):
        nranks_t = determine_minimum_nranks_t()
        total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem = get_ccsd_t_mem(nranks_t)
        nnodes_min = get_min_nodes_req(total_ccsd_t_mem)
        print(f"\n[WARNING] Minimum #nodes required for (T) using given tilesize {ccsdt_tilesize}, cache_size {cache_size} = {nnodes_min}")
        print_ccsd_memory_stats(nnodes_min,total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem)
else:
    nranks_t = determine_minimum_nranks_t()
    total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem = get_ccsd_t_mem(nranks_t)
    nnodes_min = get_min_nodes_req(total_ccsd_t_mem)
    print(f"Minimum #nodes required for (T) =  {nnodes_min}")
    print_ccsd_memory_stats(nnodes_min,total_ccsd_t_mem,total_extra_buf_mem,total_cache_mem)


