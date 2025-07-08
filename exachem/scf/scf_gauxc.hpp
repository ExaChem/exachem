/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#if defined(USE_GAUXC)
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/xc_integrator.hpp>
#include <gauxc/xc_integrator/impl.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#endif

#if defined(USE_GAUXC)
#include "exachem/common/chemenv.hpp"
#include "exachem/common/options/input_options.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "exachem/scf/scf_data.hpp"
#include "exachem/scf/scf_tensors.hpp"

namespace exachem::scf::gauxc {

std::tuple<std::shared_ptr<GauXC::XCIntegrator<Matrix>>, double>
setup_gauxc(ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data);

GauXC::Molecule make_gauxc_molecule(const std::vector<libint2::Atom>& atoms);

GauXC::BasisSet<double> make_gauxc_basis(const libint2::BasisSet& basis, const double basis_tol);

template<typename TensorType>
TensorType compute_xcf(ExecutionContext& ec, ChemEnv& chem_env, exachem::scf::TAMMTensors& ttensors,
                       exachem::scf::EigenTensors&  etensors,
                       GauXC::XCIntegrator<Matrix>& xc_integrator);

template<typename TensorType>
void compute_exx(ExecutionContext& ec, ChemEnv& chem_env, SCFData& scf_data,
                 exachem::scf::TAMMTensors& ttensors, exachem::scf::EigenTensors& etensors,
                 GauXC::XCIntegrator<Matrix>& xc_integrator);

} // namespace exachem::scf::gauxc
#endif
