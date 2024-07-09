/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_taskmap.hpp"

void exachem::scf::Loads::readLoads(std::vector<NODE_T>& s1_all, std::vector<NODE_T>& s2_all,
                                    std::vector<VAL_T> ntasks_all) {
  EDGE_T nLoads = 0;

  NODE_T rank;
  NODE_T s1;
  NODE_T s2;
  VAL_T  nTasks;

  for(size_t i = 0; i < s1_all.size(); i++) {
    s1     = s1_all[i];
    s2     = s2_all[i];
    nTasks = ntasks_all[i];
    rank   = 0;

    if(s1 > maxS1) maxS1 = s1;
    if(s2 > maxS2) maxS2 = s2;

    if(nTasks > 0) {
      loadList.push_back({nLoads, rank, s1, s2, nTasks});
      nLoads++;
    }
  }
}

void exachem::scf::Loads::simpleLoadBal(NODE_T nMachine) {
  // sort Loads array w.r.t to ntasks
  sort(loadList.begin(), loadList.end(), [](Load a, Load b) { return a.nTasks > b.nTasks; });
  auto cmpbyFirst = [](const std::pair<VAL_T, NODE_T>& T1, const std::pair<VAL_T, NODE_T>& T2) {
    return T1.first > T2.first;
  };
  // create a pq with nMachine size.
  std::vector<std::pair<VAL_T, NODE_T>> pq;
  std::vector<VAL_T>                    cW(nMachine);
  std::vector<NODE_T>                   bV(nMachine);
  std::vector<NODE_T>                   cV(nMachine);

  NODE_T b_prime = nLoads / nMachine;
  // cout<<b_prime<<endl;
  NODE_T remainder = nLoads % nMachine;

  for(NODE_T i = 0; i < nMachine; i++) { bV[i] = b_prime; }
  // settle the remainder
  for(NODE_T i = 0; i < remainder; i++) { bV[i]++; }
  for(NODE_T i = 0; i < nMachine; i++) {
    pq.push_back(std::make_pair(0, i));
    // L.loadList[i].rank = i;
    cW[i] = 0;
    cV[i] = 0;
  }
  std::make_heap(pq.begin(), pq.end(), cmpbyFirst);

  // scan over the loadList array in high to low and place it in the least load availalble queue.
  for(EDGE_T i = 0; i < nLoads; i++) {
    while(1) {
      auto   top = pq.front();
      NODE_T mId = top.second;
      std::pop_heap(pq.begin(), pq.end(), cmpbyFirst);
      pq.pop_back();
      // cout<<"Top machine, mId: "<<top.second<<" total load: "<<cW[mId]<<endl;
      if(cV[mId] < bV[mId]) {
        loadList[i].rank = mId;
        cV[mId]++;
        top.first = top.first + loadList[i].nTasks;
        cW[mId]   = cW[mId] + loadList[i].nTasks;
        if(cV[mId] < bV[mId]) {
          pq.push_back(top);
          std::push_heap(pq.begin(), pq.end(), cmpbyFirst);
        }
        // cout<<"Load "<<i<<": "<<L.loadList[i].nTasks<<" assigned machine "<<mId<<":
        // "<<cW[mId]<<endl;
        break;
      }
    }
  }
}

void exachem::scf::Loads::createTaskMap(
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& taskmap) {
  for(auto i = 0; i < nLoads; i++) {
    auto u        = loadList[i].s1;
    auto v        = loadList[i].s2;
    taskmap(u, v) = loadList[i].rank;
    // taskmap(u,v) = i;
    // std::cout<<u<<" "<<v<<" "<<taskmap(u,v)<<" "<<L.loadList[i].nTasks<<std::endl;
  }
}
