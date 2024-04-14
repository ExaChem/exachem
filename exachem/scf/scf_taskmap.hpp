/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/eigen_utils.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

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
class Loads {
public:
  EDGE_T            nLoads;
  NODE_T            maxS1;
  NODE_T            maxS2;
  std::vector<Load> loadList;

  Loads() {
    maxS1 = 1;
    maxS2 = 1;
  }

  Loads(EDGE_T nloads) {
    maxS1  = 1;
    maxS2  = 1;
    nLoads = nloads;
    loadList.reserve(nLoads);
  }
  void readLoads(std::vector<NODE_T>& s1_all, std::vector<NODE_T>& s2_all,
                 std::vector<VAL_T> ntasks_all);
  void simpleLoadBal(NODE_T nMachine);
  void createTaskMap(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& taskmap);
};
