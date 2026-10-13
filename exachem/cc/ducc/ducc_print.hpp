/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

namespace exachem::cc::ducc::internal {

// Writes a section marker (e.g. "Begin IJ Block") into the DUCC results file.
inline void ducc_print_blockstr(const std::string& filename, const std::string& val,
                                bool append = false) {
  if(filename.empty()) return;
  std::ofstream tos;
  if(append) tos.open(filename + ".txt", std::ios::app);
  else tos.open(filename + ".txt", std::ios::out);
  if(!tos) std::cerr << "Error opening file " << filename << std::endl;
  tos << val << std::endl;
  tos.close();
}

// Prints the DUCC-transformed integrals.
// For a closed-shell (RHF) reference, each block is restricted to the
// occupied/virtual index ranges given by nactoa/nactva. For an open-shell
// (UHF) reference those ranges no longer apply cleanly to a single spatial
// active space, so every block is printed unconditionally.
template<typename T>
void print_ducc_results(ChemEnv& chem_env, ExecutionContext& ec, const std::string& files_prefix,
                        bool ducc_print, size_t nactoa, size_t nactva, const Tensor<T>& ftij,
                        const Tensor<T>& ftia, const Tensor<T>& ftab, const Tensor<T>& vtijkl,
                        const Tensor<T>& vtijka, const Tensor<T>& vtaijb, const Tensor<T>& vtijab,
                        const Tensor<T>& vtiabc, const Tensor<T>& vtabcd) {
  const bool        is_unrestricted = chem_env.sys_data.is_unrestricted;
  SystemData&       sys_data        = chem_env.sys_data;
  ExecutionContext  ec_dense{ec.pg(), DistributionKind::dense,
                            MemoryManagerKind::ga}; // create ec_dense once
  const std::string results_file = files_prefix + ".ducc.results";

  if(ducc_print) { ducc_print_blockstr(results_file, "Begin IJ Block"); }

  Tensor<T>                                X1       = to_dense_tensor(ec_dense, ftij);
  std::function<bool(std::vector<size_t>)> dp_cond1 = [&](std::vector<size_t> cond) {
    if(is_unrestricted) return true;
    if(cond[0] < nactoa && cond[1] < nactoa && cond[0] <= cond[1]) return true;
    return false;
  };
  print_dense_tensor(X1, dp_cond1, results_file, true);
  if(ducc_print) {
    ducc_print_blockstr(results_file, "End IJ Block", true);
    T first_val                                         = tamm::get_tensor_element(X1, {0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X1"] = first_val;
  }
  Tensor<T>::deallocate(X1);

  if(nactva > 0) {
    if(ducc_print) ducc_print_blockstr(results_file, "Begin IA Block", true);
    Tensor<T>                                X2       = to_dense_tensor(ec_dense, ftia);
    std::function<bool(std::vector<size_t>)> dp_cond2 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactoa && cond[1] < nactva) return true;
      return false;
    };
    print_dense_tensor(X2, dp_cond2, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End IA Block", true);
      T first_val                                         = tamm::get_tensor_element(X2, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X2"] = first_val;
    }
    Tensor<T>::deallocate(X2);

    if(ducc_print) ducc_print_blockstr(results_file, "Begin AB Block", true);
    Tensor<T>                                X3       = to_dense_tensor(ec_dense, ftab);
    std::function<bool(std::vector<size_t>)> dp_cond3 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactva && cond[1] < nactva && cond[0] <= cond[1]) return true;
      return false;
    };
    print_dense_tensor(X3, dp_cond3, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End AB Block", true);
      T first_val                                         = tamm::get_tensor_element(X3, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X3"] = first_val;
    }
    Tensor<T>::deallocate(X3);
  }

  if(ducc_print) ducc_print_blockstr(results_file, "Begin IJKL Block", true);
  Tensor<T>                                X4       = to_dense_tensor(ec_dense, vtijkl);
  std::function<bool(std::vector<size_t>)> dp_cond4 = [&](std::vector<size_t> cond) {
    if(is_unrestricted) return true;
    if(cond[0] < nactoa && cond[2] < nactoa && nactoa <= cond[1] && nactoa <= cond[3]) return true;
    return false;
  };
  print_dense_tensor(X4, dp_cond4, results_file, true);
  if(ducc_print) {
    ducc_print_blockstr(results_file, "End IJKL Block", true);
    T first_val = tamm::get_tensor_element(X4, {0, 0, 0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X4"] = first_val;
  }
  Tensor<T>::deallocate(X4);

  if(nactva > 0) {
    if(ducc_print) ducc_print_blockstr(results_file, "Begin IJAB Block", true);
    Tensor<T>                                X5       = to_dense_tensor(ec_dense, vtijab);
    std::function<bool(std::vector<size_t>)> dp_cond5 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactoa && nactoa <= cond[1] && cond[2] < nactva && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X5, dp_cond5, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End IJAB Block", true);
      T first_val = tamm::get_tensor_element(X5, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X5"] = first_val;
    }
    Tensor<T>::deallocate(X5);

    if(ducc_print) ducc_print_blockstr(results_file, "Begin ABCD Block", true);
    Tensor<T>                                X6       = to_dense_tensor(ec_dense, vtabcd);
    std::function<bool(std::vector<size_t>)> dp_cond6 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactva && cond[2] < nactva && nactva <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X6, dp_cond6, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End ABCD Block", true);
      T first_val = tamm::get_tensor_element(X6, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X6"] = first_val;
    }
    Tensor<T>::deallocate(X6);

    if(ducc_print) ducc_print_blockstr(results_file, "Begin AIJB Block", true);
    Tensor<T>                                X7         = to_dense_tensor(ec_dense, vtaijb);
    std::function<bool(std::vector<size_t>)> dp_cond7_1 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactva && cond[2] < nactoa && nactoa <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_1, results_file, true);

    std::function<bool(std::vector<size_t>)> dp_cond7_2 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactva && nactoa <= cond[1] && nactoa <= cond[2] && cond[3] < nactva)
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_2, results_file, true);
    if(ducc_print) ducc_print_blockstr(results_file, "End AIJB Block", true);
    if(ducc_print) {
      T first_val = tamm::get_tensor_element(X7, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X7"] = first_val;
    }
    Tensor<T>::deallocate(X7);

    if(ducc_print) ducc_print_blockstr(results_file, "Begin IJKA Block", true);
    Tensor<T>                                X8       = to_dense_tensor(ec_dense, vtijka);
    std::function<bool(std::vector<size_t>)> dp_cond8 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactoa && cond[2] < nactoa && nactoa <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X8, dp_cond8, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End IJKA Block", true);
      T first_val = tamm::get_tensor_element(X8, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X8"] = first_val;
    }
    Tensor<T>::deallocate(X8);

    if(ducc_print) ducc_print_blockstr(results_file, "Begin IABC Block", true);
    Tensor<T>                                X9       = to_dense_tensor(ec_dense, vtiabc);
    std::function<bool(std::vector<size_t>)> dp_cond9 = [&](std::vector<size_t> cond) {
      if(is_unrestricted) return true;
      if(cond[0] < nactoa && cond[2] < nactva && nactva <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X9, dp_cond9, results_file, true);
    if(ducc_print) {
      ducc_print_blockstr(results_file, "End IABC Block", true);
      T first_val = tamm::get_tensor_element(X9, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X9"] = first_val;
    }
    Tensor<T>::deallocate(X9);
  }
}

} // namespace exachem::cc::ducc::internal
