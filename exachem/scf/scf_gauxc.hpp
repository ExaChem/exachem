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

namespace exachem::scf {

template<typename T>
class SCFGauxc {
public:
  virtual ~SCFGauxc()                      = default;
  SCFGauxc()                               = default;
  SCFGauxc(const SCFGauxc&)                = default;
  SCFGauxc& operator=(const SCFGauxc&)     = default;
  SCFGauxc(SCFGauxc&&) noexcept            = default;
  SCFGauxc& operator=(SCFGauxc&&) noexcept = default;

  virtual std::tuple<std::shared_ptr<GauXC::XCIntegrator<Matrix>>, double>
  setup_gauxc(ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data) const;

  virtual GauXC::Molecule make_gauxc_molecule(const std::vector<libint2::Atom>& atoms) const;

  virtual GauXC::BasisSet<T> make_gauxc_basis(const libint2::BasisSet& basis,
                                              const double             basis_tol) const;

  virtual T compute_xcf(ExecutionContext& ec, const ChemEnv& chem_env,
                        exachem::scf::TAMMTensors<T>& ttensors,
                        exachem::scf::EigenTensors&   etensors,
                        GauXC::XCIntegrator<Matrix>&  xc_integrator) const;

  virtual void compute_exx(ExecutionContext& ec, const ChemEnv& chem_env, const SCFData& scf_data,
                           exachem::scf::TAMMTensors<T>& ttensors,
                           exachem::scf::EigenTensors&   etensors,
                           GauXC::XCIntegrator<Matrix>&  xc_integrator) const;
};

} // namespace exachem::scf
#endif
