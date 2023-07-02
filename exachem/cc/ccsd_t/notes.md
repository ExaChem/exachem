CCSD_T_Unfused.cpp: unfused (T) code
 -> ccsd_t_unfused_driver.hpp
 -> ccsd_t_singles_unfused.hpp, ccsd_t_doubles_unfused.hpp: Call one of the following 3 (T) kernels
    -> sd_t_total_cpu.cpp (target-centric CPU kernels, set cuda 0 in input file to run these kernels)
    -> sd_t_total_gpu.cu  (TensorGen target-centric GPU kernels, set cuda #gpus in input file to run these kernels)
    -> sd_t_total_nwc.cu  (NWChem src-centric GPU kernels): 
        : To run NWChem GPU kernels, set "use_nwc_gpu_kernels = true" in CCSD_T_Unfused.cpp


 CCSD_T_Fused.cpp: fused (T) code
 -> ccsd_t_fused_driver.hpp 
 -> ccsd_t_all_fused.hpp     (calls fused target-centric TensorGen GPU kernels)
    -> ccsd_t_all_fused_singles.hpp  (singles)
    -> ccsd_t_all_fused_doubles1.hpp (doubles1)
    -> ccsd_t_all_fused_doubles2.hpp (doubles2)
 -> ccsd_t_all_fused_cpu.cpp (fused target-centric CPU kernels, set cuda 0 in input file to run these kernels)
 -> ccsd_t_all_fused_gpu.cu  (fused target-centric TensorGen GPU kernels, set cuda #gpus in input file to run these kernels)
 