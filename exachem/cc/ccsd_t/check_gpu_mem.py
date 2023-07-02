import sys

if len(sys.argv) < 3:
    print("Usage: python check_gpu_mem.py tile-size num_basis_funcs")
    sys.exit(1)

T=int(sys.argv[1])
nbf=int(sys.argv[2])

#Unfused: mem=9*T*T*T*T*(noab+nvab)*16
mem=9*T*T*T*T*nbf*16
# Fused: 9 . (T^2 + T^4 + 2.T^3.2N) . 8 bytes
mem=9*(T*T + T*T*T*T + 2*2*nbf*T*T*T) * 8
GB=1024*1024*1024.0
print("Memory required on a single gpu: " + str(mem/GB) + " GiB")