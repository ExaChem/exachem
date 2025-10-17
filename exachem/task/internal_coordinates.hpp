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

class InternalCoordinateUtils {
public:
  InternalCoordinateUtils()  = default;
  ~InternalCoordinateUtils() = default;

  static std::vector<double> atom_radii_int;
  static double              angle_eval(Eigen::MatrixXd coords, bool _, int i, int j, int k);
  static std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd>
  get_clusters(const Eigen::MatrixXd& C);
};
class InternalCoordinate {
public:
  std::string         type;
  int                 i, j, k, l;
  double              value;
  InternalCoordinate* angle0 = nullptr;
  InternalCoordinate* angle1 = nullptr;
  GeometryAnalyzer    geom;
  // Constructors
  InternalCoordinate(int _i, int _j, double _value);
  InternalCoordinate(int _i, int _j, int _k, double _value);
  InternalCoordinate(int _i, int _j, int _k, int _l, double _value, InternalCoordinate* _angle0,
                     InternalCoordinate* _angle1);
  // Destructor
  ~InternalCoordinate() = default;
  // Copy and move constructors
  InternalCoordinate(const InternalCoordinate&)     = default;
  InternalCoordinate(InternalCoordinate&&) noexcept = default;

  // Copy and move assignment operators
  InternalCoordinate& operator=(const InternalCoordinate&);
  InternalCoordinate& operator=(InternalCoordinate&&) noexcept = default;

  // Overload of Compare Operator
  bool operator<(const InternalCoordinate) const;
  // Overload of Equality Operator
  bool operator==(const InternalCoordinate&) const;

  // Member function
  double hessian_component(Eigen::MatrixXd rho);
};

class InternalCoordinates {
public:
  std::vector<InternalCoordinate> coords;
  GeometryAnalyzer                geom;
  // Default constructor
  InternalCoordinates() = default;
  // Destructor
  ~InternalCoordinates() = default;
  // Copy and move constructors
  InternalCoordinates(const InternalCoordinates&)     = default;
  InternalCoordinates(InternalCoordinates&&) noexcept = default;

  // Copy and move assignment operators
  InternalCoordinates& operator=(const InternalCoordinates&)     = default;
  InternalCoordinates& operator=(InternalCoordinates&&) noexcept = default;

  // Member functions
  InternalCoordinate operator[](int idx) const;
  int                size() const;
  void               push_back(InternalCoordinate coord);
  void               print(ExecutionContext& ec);
};

InternalCoordinates InternalCoords(ExecutionContext& ec, ChemEnv& chem_env, bool torsions);

} // namespace exachem::task
