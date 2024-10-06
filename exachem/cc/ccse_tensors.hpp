/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

// clang-format off
#include "exachem/cholesky/two_index_transform.hpp"
#include "exachem/cholesky/cholesky_2e.hpp"
// clang-format on
using namespace tamm;

/**
 * struct for managing CC spin-explicit tensors
 * CCSE_Tensors<T> cctens{MO,{V,O},"tensor_name",{"aa","bb"}};
 * CCSE_Tensors<T> cctens{MO,{V,O,CI},"tensor_name",{"aa","bb"}};
 * CCSE_Tensors<T> cctens{MO,{V,O,V,O},"tensor_name",{"aaaa","baba","baab","bbbb"}};
 */
template<typename T>
class CCSE_Tensors {
  std::map<std::string, Tensor<T>> tmap;
  std::vector<Tensor<T>>           allocated_tensors;

  std::string tname;
  bool        is_mo_3d{}; // true only when all dims of a 3D tensor are MO

  std::vector<std::string> get_tensor_files(const std::string& fprefix) {
    std::vector<std::string> tensor_files;
    for(auto iter = tmap.begin(); iter != tmap.end(); ++iter) {
      auto block = iter->first;
      tensor_files.push_back(fprefix + "." + tname + "_" + block);
    }
    return tensor_files;
  }

public:
  std::vector<std::string> vblocks;

  void deallocate() {
    ExecutionContext& ec = get_ec(allocated_tensors[0]());
    Scheduler         sch{ec};
    for(auto x: allocated_tensors) sch.deallocate(x);
    sch.execute();
  }

  T sum_tensor_sizes() {
    T total_size{};
    for(auto x: allocated_tensors)
      total_size += (compute_tensor_size(x) * 8) / (1024 * 1024 * 1024.0);
    return total_size;
  }

  Tensor<T> operator()(std::string block) {
    if(tmap.find(block) == tmap.end())
      tamm_terminate("Error: tensor [" + tname + "]: block [" + block +
                     "] requested does not exist");
    return tmap[block];
  }

  TiledIndexSpaceVec construct_tis(const TiledIndexSpace& MO, const TiledIndexSpaceVec tis,
                                   const std::vector<int> btype) {
    const auto ndims = tis.size();

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    const TiledIndexSpace o_alpha = MO("occ_alpha");
    const TiledIndexSpace o_beta  = MO("occ_beta");
    const TiledIndexSpace v_alpha = MO("virt_alpha");
    const TiledIndexSpace v_beta  = MO("virt_beta");

    TiledIndexSpaceVec btis;
    for(size_t x = 0; x < ndims; x++) {
      // assuming only 3D tensor has an independent index space
      if(tis[x] == O) btype[x] == 0 ? btis.push_back(o_alpha) : btis.push_back(o_beta);
      else if(tis[x] == V) btype[x] == 0 ? btis.push_back(v_alpha) : btis.push_back(v_beta);
      else if(ndims == 3 && !is_mo_3d) { btis.push_back(tis[x]); }
    }

    return btis;
  }

  void allocate(ExecutionContext& ec) {
    Scheduler sch{ec};
    for(auto x: allocated_tensors) sch.allocate(x);
    sch.execute();
  }

  CCSE_Tensors() {}

  /**
   * @brief Construct a group of spin-explicit tensors to be used as a single tensor
   *
   * @param [in] MO     the MO tiled index space
   * @param [in] tis    the dimensions specified using O,V tiled index spaces
   * @param [in] tname  tensor name as string
   * @param [in] blocks specify the required blocks as strings
   */

  CCSE_Tensors(const TiledIndexSpace& MO, TiledIndexSpaceVec tis, std::string tensor_name,
               std::vector<std::string> blocks) {
    tname   = tensor_name;
    vblocks = blocks;

    const auto  ndims   = tis.size();
    std::string err_msg = "Error in tensor [" + tname + "] declaration";
    if(ndims < 2 || ndims > 4) tamm_terminate(err_msg + ": Only 2,3,4D tensors are allowed");

    is_mo_3d                 = true;
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    for(size_t x = 0; x < tis.size(); x++) {
      if(tis[x] != O && tis[x] != V) {
        if(ndims == 3) is_mo_3d = false; // assuming only 3D tensors have an independent index space
        else tamm_terminate(err_msg + ": Only O,V tiled index spaces can be specified");
      }
    }

    std::vector<std::string> allowed_blocks = {"aa", "bb"};
    if(ndims == 3 && is_mo_3d) allowed_blocks = {"aaa", "baa", "abb", "bbb"};
    else if(ndims == 4) allowed_blocks = {"aaaa", "abab", "bbbb", "abba", "baab", "baba"};

    if(blocks.size() == 0)
      tamm_terminate(err_msg + ": Please specify the tensor blocks to be allocated");

    for(auto x: blocks) {
      if(std::find(allowed_blocks.begin(), allowed_blocks.end(), x) == allowed_blocks.end()) {
        if(ndims == 2 || (ndims == 3 && !is_mo_3d))
          tamm_terminate(err_msg + ": Invalid block [" + x +
                         "] specified, allowed blocks are [aa|bb]");
        else if(ndims == 3 && is_mo_3d)
          tamm_terminate(err_msg + ": Invalid block [" + x +
                         "] specified, allowed blocks are [aaa|baa|abb|bbb]");
        else
          tamm_terminate(err_msg + ": Invalid block [" + x +
                         "] specified, allowed blocks are [aaaa|abab|bbbb|abba|baab|baba]");
      }
    }

    // a=0,b=1
    if(ndims == 2 || (ndims == 3 && !is_mo_3d)) {
      if(std::find(blocks.begin(), blocks.end(), "aa") != blocks.end()) {
        Tensor<T> aa{construct_tis(MO, tis, {0, 0})};
        tmap["aa"] = aa;
        allocated_tensors.push_back(aa);
      }
      if(std::find(blocks.begin(), blocks.end(), "bb") != blocks.end()) {
        Tensor<T> bb{construct_tis(MO, tis, {1, 1})};
        tmap["bb"] = bb;
        allocated_tensors.push_back(bb);
      }
    }
    else if(ndims == 3 && is_mo_3d) {
      if(std::find(blocks.begin(), blocks.end(), "aaa") != blocks.end()) {
        Tensor<T> aaa{construct_tis(MO, tis, {0, 0, 0})};
        tmap["aaa"] = aaa;
        allocated_tensors.push_back(aaa);
      }
      if(std::find(blocks.begin(), blocks.end(), "baa") != blocks.end()) {
        Tensor<T> baa{construct_tis(MO, tis, {1, 0, 0})};
        tmap["baa"] = baa;
        allocated_tensors.push_back(baa);
      }
      if(std::find(blocks.begin(), blocks.end(), "abb") != blocks.end()) {
        Tensor<T> abb{construct_tis(MO, tis, {0, 1, 1})};
        tmap["abb"] = abb;
        allocated_tensors.push_back(abb);
      }
      if(std::find(blocks.begin(), blocks.end(), "bbb") != blocks.end()) {
        Tensor<T> bbb{construct_tis(MO, tis, {1, 1, 1})};
        tmap["bbb"] = bbb;
        allocated_tensors.push_back(bbb);
      }
    }
    else {
      if(std::find(blocks.begin(), blocks.end(), "aaaa") != blocks.end()) {
        Tensor<T> aaaa{construct_tis(MO, tis, {0, 0, 0, 0})};
        tmap["aaaa"] = aaaa;
        allocated_tensors.push_back(aaaa);
      }
      if(std::find(blocks.begin(), blocks.end(), "abab") != blocks.end()) {
        Tensor<T> abab{construct_tis(MO, tis, {0, 1, 0, 1})};
        tmap["abab"] = abab;
        allocated_tensors.push_back(abab);
      }
      if(std::find(blocks.begin(), blocks.end(), "bbbb") != blocks.end()) {
        Tensor<T> bbbb{construct_tis(MO, tis, {1, 1, 1, 1})};
        tmap["bbbb"] = bbbb;
        allocated_tensors.push_back(bbbb);
      }
      if(std::find(blocks.begin(), blocks.end(), "abba") != blocks.end()) {
        Tensor<T> abba{construct_tis(MO, tis, {0, 1, 1, 0})};
        tmap["abba"] = abba;
        allocated_tensors.push_back(abba);
      }
      if(std::find(blocks.begin(), blocks.end(), "baab") != blocks.end()) {
        Tensor<T> baab{construct_tis(MO, tis, {1, 0, 0, 1})};
        tmap["baab"] = baab;
        allocated_tensors.push_back(baab);
      }
      if(std::find(blocks.begin(), blocks.end(), "baba") != blocks.end()) {
        Tensor<T> baba{construct_tis(MO, tis, {1, 0, 1, 0})};
        tmap["baba"] = baba;
        allocated_tensors.push_back(baba);
      }
    }
  }

  void write_to_disk(const std::string& fprefix) {
    auto tensor_files = get_tensor_files(fprefix);
    // TODO: Assume all on same ec for now
    ExecutionContext& ec = get_ec(allocated_tensors[0]());
    tamm::write_to_disk_group<T>(ec, allocated_tensors, tensor_files);
  }

  void read_from_disk(const std::string& fprefix) {
    auto              tensor_files = get_tensor_files(fprefix);
    ExecutionContext& ec           = get_ec(allocated_tensors[0]());
    tamm::read_from_disk_group<T>(ec, allocated_tensors, tensor_files);
  }

  bool exist_on_disk(const std::string& fprefix) {
    auto tensor_files = get_tensor_files(fprefix);
    bool tfiles_exist = std::all_of(tensor_files.begin(), tensor_files.end(),
                                    [](std::string x) { return fs::exists(x); });
    return tfiles_exist;
  }

  // static
  static void alloc_list(Scheduler& sch) {}

  template<typename... Args>
  static void alloc_list(Scheduler& sch, CCSE_Tensors<T>& ccset, Args&... rest) {
    for(auto x: ccset.allocated_tensors) sch.allocate(x);
    alloc_list(sch, rest...);
  }

  template<typename... Args>
  static void allocate_list(Scheduler& sch, CCSE_Tensors<T>& ccset, Args&... rest) {
    alloc_list(sch, ccset, rest...);
  }

  static void dealloc_list(Scheduler& sch) {}

  template<typename... Args>
  static void dealloc_list(Scheduler& sch, CCSE_Tensors<T>& ccset, Args&... rest) {
    for(auto x: ccset.allocated_tensors) sch.deallocate(x);
    dealloc_list(sch, rest...);
  }

  template<typename... Args>
  static void deallocate_list(Scheduler& sch, CCSE_Tensors<T>& ccset, Args&... rest) {
    dealloc_list(sch, ccset, rest...);
  }

  template<typename... Args>
  static auto sum_tensor_sizes_list(Args&... ccsetensor) {
    return (ccsetensor.sum_tensor_sizes() + ...);
  }

  static void copy(Scheduler& sch, CCSE_Tensors<T>& src, CCSE_Tensors<T>& dest,
                   bool update = false) {
    for(auto x: src.vblocks) {
      if(update) sch(dest(x)() += src(x)());
      else sch(dest(x)() = src(x)());
    }
  }

  static void initialize_list(Scheduler& sch, T value) {}

  template<typename... Args>
  static void initialize_list(Scheduler& sch, T value, CCSE_Tensors<T>& ccset, Args&... rest) {
    for(auto x: ccset.vblocks) { sch(ccset(x)() = value); }
    initialize_list(sch, value, rest...);
  }

  template<typename... Args>
  static void initialize(Scheduler& sch, T value, CCSE_Tensors<T>& ccset, Args&... rest) {
    initialize_list(sch, value, ccset, rest...);
  }
};
