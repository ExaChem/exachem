/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cc/ccse_tensors.hpp"
#include "cc/diis.hpp"

// auto lambdar2 = [](const IndexVector& blockid, span<double> buf){
//     if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
//         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
//     }
// };

template<typename T>
class V2Tensors {
  std::map<std::string, Tensor<T>> tmap;
  std::vector<Tensor<T>>           allocated_tensors;
  std::vector<std::string> allowed_blocks = {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"};
  std::vector<std::string> blocks;

  std::vector<std::string> get_tensor_files(const std::string& fprefix) {
    std::vector<std::string> tensor_files;
    for(auto block: blocks) tensor_files.push_back(fprefix + ".v2" + block);
    return tensor_files;
  }

  void set_map(const TiledIndexSpace& MO) {
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

public:
  Tensor<T> v2ijab; // hhpp
  Tensor<T> v2iajb; // hphp
  Tensor<T> v2ijka; // hhhp
  Tensor<T> v2ijkl; // hhhh
  Tensor<T> v2iabc; // hppp
  Tensor<T> v2abcd; // pppp

  V2Tensors() { blocks = allowed_blocks; }

  V2Tensors(std::vector<std::string> v2blocks) {
    blocks              = v2blocks;
    std::string err_msg = "Error in V2 tensors declaration";
    for(auto x: blocks) {
      if(std::find(allowed_blocks.begin(), allowed_blocks.end(), x) == allowed_blocks.end()) {
        tamm_terminate(err_msg + ": Invalid block [" + x +
                       "] specified, allowed blocks are [ijab|iajb|ijka|ijkl|iabc|abcd]");
      }
    }
  }

  std::vector<std::string> get_blocks() { return blocks; }

  void deallocate() {
    ExecutionContext& ec = get_ec(allocated_tensors[0]());
    Scheduler         sch{ec};
    for(auto x: allocated_tensors) sch.deallocate(x);
    sch.execute();
  }

  T tensor_sizes(const TiledIndexSpace& MO) {
    set_map(MO);
    T v2_sizes{};
    for(auto iter = tmap.begin(); iter != tmap.end(); ++iter)
      v2_sizes += sum_tensor_sizes(iter->second);
    return v2_sizes;
  }

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO) {
    set_map(MO);

    for(auto iter = tmap.begin(); iter != tmap.end(); ++iter)
      allocated_tensors.push_back(iter->second);

    Scheduler sch{ec};
    for(auto x: allocated_tensors) sch.allocate(x);
    sch.execute();
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
};

template<typename T>
void setup_full_t1t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& dt1_full,
                     Tensor<T>& dt2_full);

template<typename TensorType>
void update_r2(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

template<typename TensorType>
void init_diagonal(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

// inline std::string ccsd_test(int argc, char* argv[]) {
//   if(argc < 2) {
//     std::cout << "Please provide an input file!" << std::endl;
//     exit(0);
//   }

//   auto          filename = std::string(argv[1]);
//   std::ifstream testinput(filename);
//   if(!testinput) {
//     std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
//     exit(0);
//   }

//   return filename;
// }

void iteration_print(SystemData& sys_data, const ProcGroup& pg, int iter, double residual,
                     double energy, double time, string cmethod = "CCSD");

/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double, double> rest(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1,
                               Tensor<T>& d_r2, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de,
                               Tensor<T>& d_r1_residual, Tensor<T>& d_r2_residual,
                               std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
                               const TAMM_SIZE& nob, bool transpose = false);

template<typename T>
std::pair<double, double>
rest_cs(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
        Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
        Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
        const TAMM_SIZE& nva, bool transpose = false, const bool not_spin_orbital = false);

void print_ccsd_header(const bool do_print);

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
             bool ccsd_restart = false);

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                bool ccsd_restart = false);

template<typename T>
std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, Tensor<T>, Tensor<T>,
           Tensor<T>, Tensor<T>, TiledIndexSpace, TiledIndexSpace, bool>
hartree_fock_driver(ExecutionContext& ec, const string filename, OptionsMap options_map);

void ccsd_stats(ExecutionContext& ec, double hf_energy, double residual, double energy,
                double thresh);

template<typename T>
V2Tensors<T>
setupV2Tensors(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw = ExecutionHW::CPU,
               std::vector<std::string> blocks = {"ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"});

template<typename T>
Tensor<T> setupV2(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI, Tensor<T> cholVpr,
                  const tamm::Tile chol_count, ExecutionHW hw = ExecutionHW::CPU,
                  bool anti_sym = true);

template<typename T>
void cc_print(SystemData& sys_data, Tensor<T> d_t1, Tensor<T> d_t2, std::string files_prefix);

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
cd_svd_driver(SystemData& sys_data, ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& AO,
              Tensor<T> C_AO, Tensor<T> F_AO, Tensor<T> C_beta_AO, Tensor<T> F_beta_AO,
              libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map, bool readv2 = false,
              std::string cholfile = "", bool is_dlpno = false, bool is_mso = true);

void cd_2e_driver(std::string filename, OptionsMap options_map);

void cd_mp2(std::string filename, OptionsMap options_map);
