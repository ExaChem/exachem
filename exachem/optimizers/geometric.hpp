// geometric.hpp
#pragma once

#include <string>
#include <vector>

#include "exachem/geometry/internal_coordinates.hpp"

namespace exachem::optimizers {

void finalize_python();

class GeomeTRIC {
public:
  static void optimize(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                       std::vector<ECAtom>& ec_atoms, const std::string& ec_arg2);

private:
  static Eigen::RowVectorXd current_geometry(const std::vector<Atom>& atoms);
};

} // namespace exachem::optimizers