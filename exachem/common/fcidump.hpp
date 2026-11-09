/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/cutils.hpp"
#include <fstream>

namespace fcidump {

template<typename T>
void write_2el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> V, int norb, bool is_uhf,
                    int offset = 0);

template<typename T>
void write_1el_ints(std::ofstream& file, SystemData& sys_data, Tensor<T> h, int norb, bool is_uhf);

template<typename T>
void write_fcidump_file(ChemEnv& chem_env, Tensor<T> H_MO, Tensor<T> full_v2,
                        std::vector<int> orbsym, std::string filename);

// One-shot HF energy of the occupied MSOs, evaluated from the one- and two-electron integrals that
// go into the FCIDUMP file: core_energy + sum_i h_ii + 1/2 sum_ij [(ii|jj) - (ij|ji)]
template<typename T>
T one_shot_hf_energy(ChemEnv& chem_env, Tensor<T> H_MO, Tensor<T> full_v2, T core_energy);

// Transform the AO core Hamiltonian to the MO basis of lcao and write an FCIDUMP file to
// filename (default: chem_env.get_files_prefix("", "fci") + ".fcidump")
template<typename T>
void generate_fcidump(ChemEnv& chem_env, ExecutionContext& ec, const Tensor<T>& hcore,
                      const Tensor<T>& lcao, const Tensor<T>& full_v2,
                      ExecutionHW ex_hw = ExecutionHW::CPU, std::string filename = "");

}; // namespace fcidump
