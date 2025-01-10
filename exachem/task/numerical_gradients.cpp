/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/numerical_gradients.hpp"

namespace exachem::task {

double get_task_energy(ExecutionContext& ec, ChemEnv& chem_env) {
  const TaskOptions& task   = chem_env.ioptions.task_options;
  double             energy = 0.0;
  // TODO:Assumes only 1 task is true
  if(task.scf) energy = chem_env.scf_context.hf_energy;
  else if(task.mp2) energy = chem_env.mp2_context.mp2_total_energy;
  else if(task.cc2) energy = chem_env.cc_context.cc2_total_energy;
  else if(task.ccsd) energy = chem_env.cc_context.ccsd_total_energy;
  else if(task.ccsdt) energy = chem_env.cc_context.ccsdt_total_energy;
  else if(task.ccsd_t) energy = chem_env.cc_context.ccsd_pt_total_energy;

  return energy;
}

double compute_energy(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2) {
  // std::cout << "computing energy" << std::endl;
  execute_task(ec, chem_env, ec_arg2);
  return get_task_energy(ec, chem_env);
}

Matrix compute_numerical_gradients(ExecutionContext& ec, ChemEnv& chem_env,
                                   const std::vector<Atom>&   atoms,
                                   const std::vector<ECAtom>& ec_atoms, const std::string ec_arg2) {
  const auto natoms    = chem_env.atoms.size();
  Matrix     gradients = Matrix::Zero(natoms, 3);

  // TODO: use the movecs and amplitudes from the original geometry as guess for the displaced ones.
  // parallelize the outer loop.

  // geometry is converted to bohr by default when parsed.
  // delta is in bohr by default.
  const double delta = 1e-2;
  int          step  = 1;
  for(size_t i = 0; i < natoms; i++) {
    for(int j = 0; j < 3; j++) { // x, y, z directions
      // Displace in the positive direction
      std::vector<Atom>   patoms    = atoms;
      std::vector<ECAtom> pec_atoms = ec_atoms;
      if(j == 0) {
        patoms[i].x += delta;
        pec_atoms[i].atom.x += delta;
      }
      else if(j == 1) {
        patoms[i].y += delta;
        pec_atoms[i].atom.y += delta;
      }
      else if(j == 2) {
        patoms[i].z += delta;
        pec_atoms[i].atom.z += delta;
      }
      chem_env.atoms    = patoms;
      chem_env.ec_atoms = pec_atoms;

      const auto energy_pos = compute_energy(ec, chem_env, ec_arg2);

      // Displace in the negative direction
      patoms    = atoms;
      pec_atoms = ec_atoms;
      if(j == 0) {
        patoms[i].x -= delta;
        pec_atoms[i].atom.x -= delta;
      }
      else if(j == 1) {
        patoms[i].y -= delta;
        pec_atoms[i].atom.y -= delta;
      }
      else if(j == 2) {
        patoms[i].z -= delta;
        pec_atoms[i].atom.z -= delta;
      }
      chem_env.atoms    = patoms;
      chem_env.ec_atoms = pec_atoms;

      const auto energy_neg = compute_energy(ec, chem_env, ec_arg2);

      gradients(i, j) = (energy_pos - energy_neg) / (2.0 * delta);

      if(ec.print()) {
        std::cout << std::endl
                  << "Step " << step++ << std::fixed << std::setprecision(10)
                  << ": positive, negative energies, gradient = " << energy_pos << ", "
                  << energy_neg << ", " << gradients(i, j) << std::endl;
      }

    } // xyz
  }   // natoms

  // Reset chemenv atoms
  chem_env.atoms    = atoms;
  chem_env.ec_atoms = ec_atoms;

  return gradients;
}

Matrix compute_gradients(ExecutionContext& ec, ChemEnv& chem_env, const std::vector<Atom>& atoms,
                         const std::vector<ECAtom>& ec_atoms, const std::string ec_arg2) {
  // Compute reference energy
  chem_env.atoms       = atoms;
  chem_env.ec_atoms    = ec_atoms;
  chem_env.task_energy = exachem::task::compute_energy(ec, chem_env, ec_arg2);

  if(ec.print()) {
    std::cout << std::endl
              << chem_env.task_string << " Reference Energy: " << std::fixed
              << std::setprecision(10) << chem_env.task_energy << std::endl;
  }

  Matrix gradients = compute_numerical_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);

  if(ec.print()) {
    std::cout << std::endl
              << std::setw(40) << chem_env.task_string << " ENERGY GRADIENTS" << std::endl
              << std::endl;
    std::cout << std::setw(9) << "atom" << std::setw(32) << "coordinates" << std::setw(47)
              << "gradients" << std::endl;

    std::cout << std::setw(10) << "" << std::left << std::setw(16) << std::string(9, ' ') + "x"
              << std::setw(16) << std::string(9, ' ') + "y" << std::setw(16)
              << std::string(9, ' ') + "z" << std::setw(16) << std::string(9, ' ') + "x"
              << std::setw(16) << std::string(9, ' ') + "y" << std::setw(16)
              << std::string(9, ' ') + "z" << std::endl;

    for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
      auto ecatom = chem_env.ec_atoms[i];
      std::cout << std::setw(6) << std::left << i + 1 << std::setw(4) << ecatom.esymbol
                << std::right << std::fixed << std::setprecision(10) << std::setw(16)
                << ecatom.atom.x << std::setw(16) << ecatom.atom.y << std::setw(16) << ecatom.atom.z
                << std::setw(16) << gradients(i, 0) << std::setw(16) << gradients(i, 1)
                << std::setw(16) << gradients(i, 2) << std::endl;
    }
  }

  return gradients;
}

} // namespace exachem::task
