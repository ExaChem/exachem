// geometric.hpp
#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <exachem/task/geometry_optimizer.hpp>

namespace exachem::geometric {

class GeomeTRICOptimizer {
public:
  static void optimize(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                       std::vector<ECAtom>& ec_atoms, const std::string& ec_arg2);

private:
  static Eigen::RowVectorXd current_geometry(const std::vector<Atom>& atoms);
};

} // namespace exachem::geometric