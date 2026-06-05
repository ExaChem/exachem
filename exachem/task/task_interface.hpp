/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"

// Minimal interface to the task orchestrator used by lower-level modules
// (e.g. gradients, optimizers) to run a task or evaluate its energy for the
// current geometry.

namespace exachem::task {
// Runs the enabled task (dispatches to the appropriate method driver).
void execute_task(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2);

// Runs the enabled task and returns its energy for the current geometry.
double compute_energy(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2);
} // namespace exachem::task
