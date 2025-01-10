/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * This file contains a C++ version of the Internal Coordinates implementation
 * from the PyBerny Molecular structure optimizer (https://github.com/jhrmnn/pyberny),
 * licensed under the Mozilla Public License, v. 2.0.
 * You can obtain a copy of the MPL-2.0 license at http://mozilla.org/MPL/2.0
 *
 * This file is distributed under the Mozilla Public License, v. 2.0.
 *
 */

#include "exachem/task/geometry_analysis.hpp"

namespace exachem::task {

class InternalCoordinate {
public:
  std::string         type;
  int                 i, j, k, l;
  double              value;
  InternalCoordinate* angle0;
  InternalCoordinate* angle1;

  // Constructors
  InternalCoordinate(int _i, int _j, double _value);
  InternalCoordinate(int _i, int _j, int _k, double _value);
  InternalCoordinate(int _i, int _j, int _k, int _l, double _value, InternalCoordinate* _angle0,
                     InternalCoordinate* _angle1);

  // Operator overloads
  bool operator<(const InternalCoordinate other) const;
  bool operator==(const InternalCoordinate& other) const;

  // Member function
  double hessian_component(Eigen::MatrixXd rho);
};

class InternalCoordinates {
public:
  std::vector<InternalCoordinate> coords;

  // Member functions
  InternalCoordinate operator[](int idx);
  int                size();
  void               push_back(InternalCoordinate coord);
};

InternalCoordinates InternalCoords(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::task