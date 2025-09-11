/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/tamm.hpp"

using namespace tamm;

namespace exachem::cholesky_2e {

template<typename T>
class V2Tensors {
private:
  std::map<std::string, Tensor<T>> tmap;
  std::vector<Tensor<T>>           allocated_tensors;
  std::vector<std::string> allowed_blocks = {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"};
  std::vector<std::string> blocks;

  void                     set_map(const TiledIndexSpace& MO);
  std::vector<std::string> get_tensor_files(const std::string& fprefix);

public:
  Tensor<T> v2ijab; // hhpp
  Tensor<T> v2iajb; // hphp
  Tensor<T> v2ijka; // hhhp
  Tensor<T> v2ijkl; // hhhh
  Tensor<T> v2iabc; // hppp
  Tensor<T> v2abcd; // pppp

  V2Tensors();
  V2Tensors(std::vector<std::string> v2blocks);

  std::vector<std::string> get_blocks();
  void                     deallocate();
  T                        tensor_sizes(const TiledIndexSpace& MO);
  void                     allocate(ExecutionContext& ec, const TiledIndexSpace& MO);
  void                     write_to_disk(const std::string& fprefix);
  void                     read_from_disk(const std::string& fprefix);
  bool                     exist_on_disk(const std::string& fprefix);
};

// ---------------------------------- class V2TensorSetup -------------------------------
// template<typename T>
// class V2TensorSetup {
// public:
//     V2TensorSetup(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw = ExecutionHW::CPU);
//     V2Tensors<T> setup(std::vector<std::string> blocks = {"ijab", "iajb", "ijka", "ijkl", "iabc",
//     "abcd"});

// private:
//     ExecutionContext& ec_;
//     Tensor<T> cholVpr_;
//     ExecutionHW ex_hw_;
// };
// ---------------------------------- class V2TensorSetup -------------------------------

template<typename T>
V2Tensors<T>
setupV2Tensors(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw = ExecutionHW::CPU,
               std::vector<std::string> blocks = {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"});

template<typename T>
Tensor<T> setupV2(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI, Tensor<T> cholVpr,
                  const tamm::Tile chol_count, ExecutionHW hw = ExecutionHW::CPU,
                  bool anti_sym = true);

} // namespace exachem::cholesky_2e

template<typename T>
Tensor<T> exachem::cholesky_2e::setupV2(ExecutionContext& ec, TiledIndexSpace& MO,
                                        TiledIndexSpace& CI, Tensor<T> cholVpr,
                                        const tamm::Tile chol_count, ExecutionHW hw,
                                        bool anti_sym) {
  auto rank = ec.pg().rank();

  TiledIndexSpace N = MO("all");
  auto [cindex]     = CI.labels<1>("all");
  auto [p, q, r, s] = MO.labels<4>("all");

  // Spin here is defined as spin(p)=spin(r) and spin(q)=spin(s) which is not currently not
  // supported by TAMM.
  //  Tensor<T> d_a2{{N,N,N,N},{2,2}};
  // For V2, spin(p)+spin(q) == spin(r)+spin(s)
  Tensor<T> d_v2{{N, N, N, N}, {2, 2}};
  Tensor<T>::allocate(&ec, d_v2);

  auto cc_t1 = std::chrono::high_resolution_clock::now();
  // clang-format off
  Scheduler sch{ec};
  sch(d_v2(p, q, r, s)  = cholVpr(p, r, cindex) * cholVpr(q, s, cindex));
  if(anti_sym)
    sch(d_v2(p, q, r, s) += -1.0 * cholVpr(p, s, cindex) * cholVpr(q, r, cindex));
  sch.execute(hw);
  // clang-format on

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double v2_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time to reconstruct V2: " << std::fixed << std::setprecision(2) << v2_time
              << " secs" << std::endl;

  // Tensor<T>::deallocate(d_a2);
  return d_v2;
}

template<typename T>
std::vector<std::string>
exachem::cholesky_2e::V2Tensors<T>::get_tensor_files(const std::string& fprefix) {
  std::vector<std::string> tensor_files;
  for(auto block: blocks) { tensor_files.push_back(fprefix + ".v2" + block); }
  return tensor_files;
}

template<typename T>
void exachem::cholesky_2e::V2Tensors<T>::set_map(const TiledIndexSpace& MO) {
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  for(auto x: blocks) {
    if(x == "ijab") {
      v2ijab  = Tensor<T>{{h1, h2, p1, p2}, {2, 2}};
      tmap[x] = v2ijab;
    }
    else if(x == "iajb") {
      v2iajb  = Tensor<T>{{h1, p1, h2, p2}, {2, 2}};
      tmap[x] = v2iajb;
    }
    else if(x == "ijka") {
      v2ijka  = Tensor<T>{{h1, h2, h3, p1}, {2, 2}};
      tmap[x] = v2ijka;
    }
    else if(x == "ijkl") {
      v2ijkl  = Tensor<T>{{h1, h2, h3, h4}, {2, 2}};
      tmap[x] = v2ijkl;
    }
    else if(x == "iabc") {
      v2iabc  = Tensor<T>{{h1, p1, p2, p3}, {2, 2}};
      tmap[x] = v2iabc;
    }
    else if(x == "abcd") {
      v2abcd  = Tensor<T>{{p1, p2, p3, p4}, {2, 2}};
      tmap[x] = v2abcd;
    }
  }
}

template<typename T>
exachem::cholesky_2e::V2Tensors<T>::V2Tensors() {
  blocks = allowed_blocks;
}

template<typename T>
exachem::cholesky_2e::V2Tensors<T>::V2Tensors(std::vector<std::string> v2blocks) {
  blocks              = v2blocks;
  std::string err_msg = "Error in V2 tensors declaration";
  for(auto x: blocks) {
    if(std::find(allowed_blocks.begin(), allowed_blocks.end(), x) == allowed_blocks.end()) {
      tamm_terminate(err_msg + ": Invalid block [" + x +
                     "] specified, allowed blocks are [ijab|iajb|ijka|ijkl|iabc|abcd]");
    }
  }
}

template<typename T>
std::vector<std::string> exachem::cholesky_2e::V2Tensors<T>::get_blocks() {
  return blocks;
}

template<typename T>
void exachem::cholesky_2e::V2Tensors<T>::deallocate() {
  ExecutionContext& ec = get_ec(allocated_tensors[0]());
  Scheduler         sch{ec};
  for(auto x: allocated_tensors) sch.deallocate(x);
  sch.execute();
}

template<typename T>
T exachem::cholesky_2e::V2Tensors<T>::tensor_sizes(const TiledIndexSpace& MO) {
  set_map(MO);
  T v2_sizes{};
  for(auto iter = tmap.begin(); iter != tmap.end(); ++iter)
    v2_sizes += (compute_tensor_size(iter->second) * 8) / (1024 * 1024 * 1024.0);
  return v2_sizes;
}

template<typename T>
void exachem::cholesky_2e::V2Tensors<T>::allocate(ExecutionContext& ec, const TiledIndexSpace& MO) {
  set_map(MO);

  for(auto iter = tmap.begin(); iter != tmap.end(); ++iter)
    allocated_tensors.push_back(iter->second);

  Scheduler sch{ec};
  for(auto x: allocated_tensors) sch.allocate(x);
  sch.execute();
}

template<typename T>
void exachem::cholesky_2e::V2Tensors<T>::write_to_disk(const std::string& fprefix) {
  auto tensor_files = get_tensor_files(fprefix);
  // TODO: Assume all on same ec for now
  ExecutionContext& ec = get_ec(allocated_tensors[0]());
  tamm::write_to_disk_group<T>(ec, allocated_tensors, tensor_files);
}

template<typename T>
void exachem::cholesky_2e::V2Tensors<T>::read_from_disk(const std::string& fprefix) {
  auto              tensor_files = get_tensor_files(fprefix);
  ExecutionContext& ec           = get_ec(allocated_tensors[0]());
  tamm::read_from_disk_group<T>(ec, allocated_tensors, tensor_files);
}

template<typename T>
bool exachem::cholesky_2e::V2Tensors<T>::exist_on_disk(const std::string& fprefix) {
  auto tensor_files = get_tensor_files(fprefix);
  bool tfiles_exist = std::all_of(tensor_files.begin(), tensor_files.end(),
                                  [](std::string x) { return std::filesystem::exists(x); });
  return tfiles_exist;
}
