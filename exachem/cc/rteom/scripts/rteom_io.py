import os
import sys

if __name__ == "__main__":

    if len(sys.argv) < 5: 
      print("\nUsage: python rteom_io.py oa ob va vb")
      sys.exit()

    oa = int(sys.argv[1])
    ob = int(sys.argv[2])
    va = int(sys.argv[3])
    vb = int(sys.argv[4])
    N = oa+ob+va+vb
    K = 8*N

    # complex 2 copies - t1_vo  {MO,{V,O},"t1",{"aa","bb"}};
    # complex 2 copies - t2_vvoo{MO,{V,V,O,O},"t2",{"aaaa","abab","bbbb"}};
    #td_f1mo x2, lcaox2
    #chol_v2 x 2

    Gib = (1024*1024*1024.0)
    t1_size = 2*(va*oa + va*ob) * 16 / Gib
    t2_size = 2*(va*va*oa*oa + va*vb*oa*ob + vb*vb*ob*ob) * 16 / Gib
    cholv2_size = 2*8*K*(oa*oa+oa*va+va*oa+va*va+ob*ob+ob*vb+vb*ob+vb*vb)/Gib

    # total = 2*t1_size + 2*t2_size + 2*cholv2_size

    print("t1_size io in GiB = %s" %(t1_size))
    print("t2_size io in GiB = %s" %(t2_size))
    print("cholv2_size io in GiB = %s" %(cholv2_size))
    print("cholv2_size io in TiB = %s" %(cholv2_size/1024.0))





