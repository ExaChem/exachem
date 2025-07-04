/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/eigen_utils.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

namespace exachem::scf {

typedef int NODE_T;
typedef int EDGE_T;
typedef int VAL_T;

// load data struct
class Load {
public:
  NODE_T loadId;
  NODE_T rank;
  NODE_T s1;
  NODE_T s2;
  VAL_T  nTasks;
};

// list of loads as a class
class DefaultLoads {
public:
  EDGE_T            nLoads;
  NODE_T            maxS1;
  NODE_T            maxS2;
  std::vector<Load> loadList;

  DefaultLoads() {
    maxS1 = 1;
    maxS2 = 1;
  }

  DefaultLoads(EDGE_T nloads) {
    maxS1  = 1;
    maxS2  = 1;
    nLoads = nloads;
    loadList.reserve(nLoads);
  }

  virtual void readLoads(std::vector<NODE_T>& s1_all, std::vector<NODE_T>& s2_all,
                         std::vector<VAL_T> ntasks_all);
  virtual void simpleLoadBal(NODE_T nMachine);
  virtual void
  createTaskMap(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& taskmap);

  virtual ~DefaultLoads() = default;
};

class Loads: public DefaultLoads {
public:
  Loads(): DefaultLoads() {}
  Loads(EDGE_T nloads): DefaultLoads(nloads) {}
  // Optionally override virtual methods here
};
} // namespace exachem::scf
