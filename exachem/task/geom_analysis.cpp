#include "exachem/task/geom_analysis.hpp"

namespace exachem::task {

void print_geometry(ExecutionContext& ec, ChemEnv& chem_env) {
  if(ec.print()) {
    std::cout << std::endl << std::string(60, '-') << std::endl;
    for(auto ecatom: chem_env.ec_atoms) {
      std::cout << std::setw(3) << std::left << ecatom.esymbol << " " << std::right << std::setw(14)
                << std::fixed << std::setprecision(10) << ecatom.atom.x << " " << std::right
                << std::setw(14) << std::fixed << std::setprecision(10) << ecatom.atom.y << " "
                << std::right << std::setw(14) << std::fixed << std::setprecision(10)
                << ecatom.atom.z << std::endl;
    }
  }
}

void geometry_analysis(ExecutionContext& ec, ChemEnv& chem_env) {
  // TBD
}

} // namespace exachem::task