import sys
import math as m
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

#python3 ccsd_scaling.py -oi 146 -vi 1568 -ki 1882 -ti 36 -oj 498 -vj 2120 -kj 4500

def parseargs(argv=None):

    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    #num_cores = psutil.cpu_count(logical=False)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-oi', '--o_i',  type=int, required=True, help='#occupied alpha for i')
    parser.add_argument('-vi', '--v_i',  type=int, required=True, help='#virtual alpha for i')
    parser.add_argument('-ki', '--k_i' , type=int, required=True, help='#nodes for i')
    parser.add_argument('-ti', '--t_i' , type=float, required=True, help='Time for i')
    parser.add_argument('-oj', '--o_j',  type=int, required=True, help='#occupied alpha for j')
    parser.add_argument('-vj', '--v_j',  type=int, required=True, help='#virtual alpha for j')
    parser.add_argument('-kj', '--k_j' , type=int, required=True, help='#nodes for j')

    # Process arguments
    args = parser.parse_args()
    return args

cc_args = parseargs()

o_i=cc_args.o_i
v_i=cc_args.v_i
k_i=cc_args.k_i
t_i=cc_args.t_i

o_j=cc_args.o_j
v_j=cc_args.v_j
k_j=cc_args.k_j

print("\nArgs passed:")
print(cc_args)
print("")
print("Estimating time for O,V=%s,%s on %s nodes.." %(o_j,v_j,k_j))

#Tj = Ti x *(Ki/Kj) x (No^2_j x Nv^4_j / No^2_i x Nv^4_i) - for problems with varying Nocc for CCSD
t_j = t_i * (k_i/k_j) * (m.pow(o_j,2) * m.pow(v_j,4)) / (m.pow(o_i,2) * m.pow(v_i,4))
print("Time per CCSD iteration: " + str(t_j))

t_j = t_j / (60*24*365)
# print(" -- in years: " + str(t_j))

#Tj = Ti x *(Ki/Kj) x (No^4_j x Nv^3_j + No^3_j x Nv^4_j) / (No^4_i x Nv^3_i + No^3_i x Nv^4_i) -- for problems with varying Nocc for (T)
t_j = t_i * (k_i/k_j) * (m.pow(o_j,4) * m.pow(v_j,3) + m.pow(o_j,3) * m.pow(v_j,4)) / (m.pow(o_i,4) * m.pow(v_i,3) + m.pow(o_i,3) * m.pow(v_i,4))

print("Time for (T): " + str(t_j))
t_j = t_j / (60*24*365)
# print(" -- in years: " + str(t_j))


