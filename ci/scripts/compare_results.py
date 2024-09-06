#!/usr/bin/env python

import sys
import os
import json
import math
import ntpath

def isclose(a, b, rel_tol=1e-09, abs_tol=0):
  return abs(a-b) <= rel_tol #max(rel_tol * max(abs(a), abs(b)), abs_tol)

if len(sys.argv) < 3:
    print("\nUsage: python3 compare_results.py reference_results_path current_results_path")
    sys.exit(1)

ref_res_path = os.path.abspath(str(sys.argv[1]))
cur_res_path = os.path.abspath(str(sys.argv[2]))
file_compare = False

upcxx = False
if len(sys.argv) == 4: upcxx = True

#check if above paths exist
if not os.path.exists(ref_res_path): 
    print("ERROR: " + ref_res_path + " does not exist!")
    sys.exit(1)

if not os.path.exists(cur_res_path): 
    print("ERROR: " + cur_res_path + " does not exist!")
    sys.exit(1)

if(os.path.isfile(ref_res_path)):
    file_compare = True
    ref_files = [ntpath.basename(ref_res_path)]
    ref_res_path = ntpath.dirname(ref_res_path)
    if(os.path.isfile(cur_res_path)):
        cur_files = [ntpath.basename(cur_res_path)]
        cur_res_path = ntpath.dirname(cur_res_path)
    else: 
        print("ERROR: " + cur_res_path + " should be the path to a json file!")
        sys.exit(1)
else:
    ref_files = os.listdir(ref_res_path)
    cur_files = os.listdir(cur_res_path)

ref_notreq = ["ubiquitin_dgrtl","uracil.cc-pvdz.ccsd_t.json","ch4.def2-tzvp.ccsd_t.json"]
for rf in ref_notreq:
    if rf not in cur_files:
        ref_files.remove(rf)

def check_results(ref_energy,cur_energy,ccsd_threshold,en_str):
    if (not isclose(ref_energy, cur_energy, ccsd_threshold)):
        errmsg = " ... ERROR: mismatch in " + en_str + "\nreference: " \
        + str(ref_energy) + ", current: " + str(cur_energy)
        print(errmsg)
        return False
    return True

missing_tests=[]
for ref_file in ref_files:
    if ref_file not in cur_files and not file_compare:
        print("WARNING: " + ref_file + " not available in " + cur_res_path)
        missing_tests.append(ref_file)
        #sys.exit(1)
        continue
    
    with open(ref_res_path+"/"+ref_file) as ref_json_file:
        ref_data = json.load(ref_json_file)

    cur_file = ref_file
    if file_compare: cur_file = cur_files[0]

    with open(cur_res_path+"/"+cur_file) as cur_json_file:
        cur_data = json.load(cur_json_file)    

    scf_threshold = ref_data["input"]["SCF"]["conve"]
    ref_scf_energy = ref_data["output"]["SCF"]["final_energy"]
    cur_scf_energy = cur_data["output"]["SCF"]["final_energy"]

    print(str(ref_file) + ": ", end='')

    if not isclose(ref_scf_energy, cur_scf_energy, scf_threshold*10):
        print("ERROR: SCF energy does not match. reference: " + str(ref_scf_energy) + ", current: " + str(cur_scf_energy))
        sys.exit(1)

    ccsd_threshold = ref_data["input"]["CCSD"]["threshold"]
    if "CCSD" in ref_data["output"]:
        #print("Checking CCSD results")
        ref_ccsd_energy = ref_data["output"]["CCSD"]["final_energy"]["correlation"]
        cur_ccsd_energy = cur_data["output"]["CCSD"]["final_energy"]["correlation"]
        rcheck = check_results(ref_ccsd_energy,cur_ccsd_energy,ccsd_threshold,"CCSD correlation energy")
        if not rcheck: sys.exit(1)

    if "DLPNO-CCSD" in ref_data["output"]:
        print("Checking DLPNO-CCSD results", end='')
        ref_dlpno_ccsd_energy = ref_data["output"]["DLPNO-CCSD"]["final_energy"]["correlation"]
        cur_dlpno_ccsd_energy = cur_data["output"]["DLPNO-CCSD"]["final_energy"]["correlation"]
        rcheck = check_results(ref_dlpno_ccsd_energy,cur_dlpno_ccsd_energy,ccsd_threshold,"DLPNO-CCSD correlation energy")
        if not rcheck: sys.exit(1)


    if "CCSD(T)" in ref_data["output"]:
        print("Checking CCSD(T) results", end='')
        ref_pt_data = ref_data["output"]["CCSD(T)"]
        cur_pt_data = cur_data["output"]["CCSD(T)"]
        
        ref_correction = ref_pt_data["[T]Energies"]["correction"]
        cur_correction = cur_pt_data["[T]Energies"]["correction"]

        ref_correlation = ref_pt_data["[T]Energies"]["correlation"]
        cur_correlation = cur_pt_data["[T]Energies"]["correlation"]

        ref_total = ref_pt_data["[T]Energies"]["total"]
        cur_total = cur_pt_data["[T]Energies"]["total"]        

        rcheck = check_results(ref_correction,cur_correction,ccsd_threshold,"[T] Correction Energy")
        rcheck &= check_results(ref_correlation,cur_correlation,ccsd_threshold,"[T] Correlation Energy")
        rcheck &= check_results(ref_total,cur_total,ccsd_threshold,"[T] Total Energy")

        ref_correction = ref_pt_data["(T)Energies"]["correction"]
        cur_correction = cur_pt_data["(T)Energies"]["correction"]

        ref_correlation = ref_pt_data["(T)Energies"]["correlation"]
        cur_correlation = cur_pt_data["(T)Energies"]["correlation"]

        ref_total = ref_pt_data["(T)Energies"]["total"]
        cur_total = cur_pt_data["(T)Energies"]["total"]           

        rcheck &= check_results(ref_correction,cur_correction,ccsd_threshold,"(T) Correction Energy")
        rcheck &= check_results(ref_correlation,cur_correlation,ccsd_threshold,"(T) Correlation Energy")
        rcheck &= check_results(ref_total,cur_total,ccsd_threshold,"(T) Total Energy")

        if not rcheck: sys.exit(1)


    if "CCSD_Lambda" in ref_data["output"]:
        print("Checking CCSD_Lambda results", end='')
        ref_lambda_data = ref_data["output"]["CCSD_Lambda"]["dipole"]
        cur_lambda_data = cur_data["output"]["CCSD_Lambda"]["dipole"]

        rcheck = True
        for i in ['X','Y','Z','Total']:
            ref_correction = ref_lambda_data[i]
            cur_correction = cur_lambda_data[i]
            rcheck &= check_results(ref_correction,cur_correction,ccsd_threshold,i)
        if not rcheck: sys.exit(1)

    if "EOMCCSD" in ref_data["output"]:
        print("Checking EOMCCSD results", end='')
        nroots = ref_data["input"]["EOMCCSD"]["eom_nroots"]
        ref_lambda_data = ref_data["output"]["EOMCCSD"]["iter"]
        cur_lambda_data = cur_data["output"]["EOMCCSD"]["iter"]

        rcheck = True
        for i in range(1,30):
            iter_str = str(i)
            if not iter_str in ref_lambda_data: break
            ref_tvecs = ref_lambda_data[iter_str]["num_trial_vectors"]
            cur_tvecs = cur_lambda_data[iter_str]["num_trial_vectors"]
            rcheck &= check_results(ref_tvecs,cur_tvecs,ccsd_threshold,"num trial vectors")

            for root in range(1,nroots+1):
                rootstr = "root"+str(root)
                ref_iter_data = ref_lambda_data[iter_str][rootstr]
                cur_iter_data = cur_lambda_data[iter_str][rootstr]
                rcheck &= check_results(ref_iter_data["energy"],cur_iter_data["energy"],ccsd_threshold,"iter "+iter_str+": "+ rootstr +" energy")
                rcheck &= check_results(ref_iter_data["residual"],cur_iter_data["residual"],ccsd_threshold,"iter "+iter_str + ": " + rootstr + " residual")
                if not rcheck: sys.exit(1)

    if "RT-EOMCCSD" in ref_data["output"]:
        print("Checking RT-EOMCCSD results", end='')
        pcore = ref_data["input"]["RT-EOMCC"]["pcore"]
        ntimesteps = ref_data["input"]["RT-EOMCC"]["ntimesteps"]
        ref_rteom_data = ref_data["output"]["RT-EOMCCSD"]["dcdt"]["timestep"]
        cur_rteom_data = cur_data["output"]["RT-EOMCCSD"]["dcdt"]["timestep"]

        for i in range(1,ntimesteps+1):
            ts_str = str(i)
            ref_energy = ref_rteom_data[ts_str]["energy"]
            cur_energy = cur_rteom_data[ts_str]["energy"]
            
            rcheck = check_results(ref_energy["real"],cur_energy["real"],ccsd_threshold,"timestep "+ts_str+": energy-real")
            rcheck &= check_results(ref_energy["imag"],cur_energy["imag"],ccsd_threshold,"timestep "+ts_str+": energy-imag")
            rcheck &= check_results(ref_rteom_data[ts_str]["time_au"],cur_rteom_data[ts_str]["time_au"],ccsd_threshold,"timestep"+ts_str+": time_au")

            if not rcheck: sys.exit(1)

    if "DUCC" in ref_data["output"]:
        print("Checking DUCC results", end='')
        ref_ducc_data = ref_data["output"]["DUCC"]["results"]
        cur_ducc_data = cur_data["output"]["DUCC"]["results"]
        
        rcheck = True
        for i in range(1,10):
            ref_correction = abs(ref_ducc_data["X"+str(i)])
            cur_correction = abs(cur_ducc_data["X"+str(i)])
            rcheck &= check_results(ref_correction,cur_correction,ccsd_threshold,"X"+str(i))

        if not rcheck: sys.exit(1)

    if "GFCCSD" in ref_data["output"]:
        print("Checking GFCCSD results", end='')
        ref_gfcc_data = ref_data["output"]["GFCCSD"]["retarded_alpha"]
        cur_gfcc_data = cur_data["output"]["GFCCSD"]["retarded_alpha"]

        ref_nlevels = ref_gfcc_data["nlevels"]
        cur_nlevels = cur_gfcc_data["nlevels"]

        gf_threshold = ref_data["input"]["GFCCSD"]["gf_threshold"]

        rcheck = check_results(ref_nlevels,cur_nlevels,gf_threshold,"levels")
        
        for lvl in range(1,ref_nlevels+1):
            lvl_str = "level"+str(lvl)

            ref_omega_npts = ref_gfcc_data[lvl_str]["omega_npts"]
            cur_omega_npts = cur_gfcc_data[lvl_str]["omega_npts"]

            rcheck &= check_results(ref_omega_npts,cur_omega_npts,gf_threshold,lvl_str+": number of frequency points")

            for ni in range(0,ref_omega_npts):
                ref_w = ref_gfcc_data[lvl_str][str(ni)]["omega"]
                cur_w = cur_gfcc_data[lvl_str][str(ni)]["omega"]
                ref_A = ref_gfcc_data[lvl_str][str(ni)]["A_a"]
                cur_A = cur_gfcc_data[lvl_str][str(ni)]["A_a"]

                # if(isclose(ref_w,0.0)): cur_w = 0.0
                if (not isclose(ref_w, cur_w)) or (not isclose(ref_A, cur_A, gf_threshold)):
                    print("ERROR: " + lvl_str + " omega, A_a mismatch. reference (w, A0): (" + str(ref_w) + "," + str(ref_A) +
                    "), current (w, A0): (" + str(cur_w) + "," + str(cur_A) + ")")
                    rcheck &= False
                    
            if not rcheck: sys.exit(1)
    
    print(" ... OK")

upcxx_skip_tests=["ozone.sto-3g.rt-eomccsd.json","co.cc-pvdz.gfccsd.json","cr2.def2-svp.scf.json"]
if upcxx:
    print(" **** upcxx: skipping rt-eomccsd and gfccsd tests ****")
    for rf in upcxx_skip_tests:
        missing_tests.remove(rf)

if not missing_tests:
    print(" **** ALL TESTS PASS ****")
else: 
    missing_tests = [os.path.splitext(x)[0] for x in missing_tests]
    print(" ************ The following tests failed ************** ")
    print("\n".join(missing_tests))
    sys.exit(1)
