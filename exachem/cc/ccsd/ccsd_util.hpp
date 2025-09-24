/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
// clang-format off
#include "exachem/cc/ccse_tensors.hpp"
#include "exachem/cc/diis.hpp"
#include "exachem/scf/scf_main.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"
// clang-format on

template<typename T>
void setup_full_t1t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& dt1_full,
                     Tensor<T>& dt2_full);

template<typename TensorType>
void update_r2(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

template<typename TensorType>
void init_diagonal(ExecutionContext& ec, LabeledTensor<TensorType> ltensor);

void iteration_print(ChemEnv& chem_env, const ProcGroup& pg, int iter, double residual,
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
std::pair<double, double> rest3(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1,
                                Tensor<T>& d_r2, Tensor<T>& d_r3, Tensor<T>& d_t1, Tensor<T>& d_t2,
                                Tensor<T>& d_t3, Tensor<T>& de, Tensor<T>& d_r1_residual,
                                Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl,
                                const TAMM_SIZE& noa, const TAMM_SIZE& nob, bool transpose = false);

template<typename T>
std::pair<double, double>
rest_qed(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
         Tensor<T>& d_r1_1p, Tensor<T>& d_r2_1p, Tensor<T>& d_r1_2p, Tensor<T>& d_r2_2p,
         Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_t1_1p, Tensor<T>& d_t2_1p,
         Tensor<T>& d_t1_2p, Tensor<T>& d_t2_2p, Tensor<T>& de, Tensor<T>& d_r1_residual,
         Tensor<T>& d_r2_residual, Tensor<T>& d_r1_1p_residual, Tensor<T>& d_r2_1p_residual,
         Tensor<T>& d_r1_2p_residual, Tensor<T>& d_r2_2p_residual, std::vector<T>& p_evl_sorted,
         T zshiftl, double omega, const TAMM_SIZE& noa, const TAMM_SIZE& nob,
         bool transpose = false);

template<typename T>
std::pair<double, double>
rest_cs(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_r1, Tensor<T>& d_r2,
        Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& de, Tensor<T>& d_r1_residual,
        Tensor<T>& d_r2_residual, std::vector<T>& p_evl_sorted, T zshiftl, const TAMM_SIZE& noa,
        const TAMM_SIZE& nva, bool transpose = false, const bool not_spin_orbital = false);

void print_ccsd_header(const bool do_print, std::string mname = "");

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
             bool ccsd_restart = false);

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
           Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_qed(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                 bool ccsd_restart = false);

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis,
                bool ccsd_restart = false);

void ccsd_stats(ExecutionContext& ec, double hf_energy, double residual, double energy,
                double thresh, std::string task_str = "CCSD");

template<typename T>
void cc_print(ChemEnv& chem_env, Tensor<T> d_t1, Tensor<T> d_t2, std::string files_prefix);

template<typename T>
struct V2Tensors_SE {
  CCSE_Tensors<T> v2ijab;
  CCSE_Tensors<T> v2iajb;
  CCSE_Tensors<T> v2ijka;
  CCSE_Tensors<T> v2ijkl;
  CCSE_Tensors<T> v2iabc;
  CCSE_Tensors<T> v2abcd;

  void allocate(Scheduler& sch, const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    v2ijab                   = CCSE_Tensors<T>{MO, {O, O, V, V}, "ijab", {"aaaa", "abab", "bbbb"}};
    v2iajb                   = CCSE_Tensors<T>{MO, {O, V, O, V}, "iajb", {"aaaa", "abab", "bbbb"}};
    v2ijka                   = CCSE_Tensors<T>{MO, {O, O, O, V}, "ijka", {"aaaa", "abab", "bbbb"}};
    v2ijkl                   = CCSE_Tensors<T>{MO, {O, O, O, O}, "ijkl", {"aaaa", "abab", "bbbb"}};
    v2iabc                   = CCSE_Tensors<T>{MO, {O, V, V, V}, "iabc", {"aaaa", "abab", "bbbb"}};
    v2abcd                   = CCSE_Tensors<T>{MO, {V, V, V, V}, "abcd", {"aaaa", "abab", "bbbb"}};
    CCSE_Tensors<T>::allocate_list(sch, v2ijab, v2iajb, v2ijka, v2ijkl, v2iabc, v2abcd);
  }

  void init(Scheduler& sch, const TiledIndexSpace& MO,
            exachem::cholesky_2e::V2Tensors<T>& v2tensors) {
    auto [p1_va, p2_va, p3_va, p4_va] = MO.labels<4>("virt_alpha");
    auto [p1_vb, p2_vb, p3_vb, p4_vb] = MO.labels<4>("virt_beta");
    auto [h1_oa, h2_oa, h3_oa, h4_oa] = MO.labels<4>("occ_alpha");
    auto [h1_ob, h2_ob, h3_ob, h4_ob] = MO.labels<4>("occ_beta");

    // clang-format off
    sch( v2ijab("aaaa")(h1_oa,h2_oa,p1_va,p2_va)  =  1.0 * v2tensors.v2ijab(h1_oa,h2_oa,p1_va,p2_va) )
       ( v2ijab("abab")(h1_oa,h2_ob,p1_va,p2_vb)  =  1.0 * v2tensors.v2ijab(h1_oa,h2_ob,p1_va,p2_vb) )
       ( v2ijab("bbbb")(h1_ob,h2_ob,p1_vb,p2_vb)  =  1.0 * v2tensors.v2ijab(h1_ob,h2_ob,p1_vb,p2_vb) );

    sch( v2iajb("aaaa")(h1_oa,p1_va,h2_oa,p2_va)  =  1.0 * v2tensors.v2iajb(h1_oa,p1_va,h2_oa,p2_va) )
       ( v2iajb("abab")(h1_oa,p1_vb,h2_oa,p2_vb)  =  1.0 * v2tensors.v2iajb(h1_oa,p1_vb,h2_oa,p2_vb) )
       ( v2iajb("bbbb")(h1_ob,p1_vb,h2_ob,p2_vb)  =  1.0 * v2tensors.v2iajb(h1_ob,p1_vb,h2_ob,p2_vb) );

    sch( v2ijka("aaaa")(h1_oa,h2_oa,h3_oa,p1_va)  =  1.0 * v2tensors.v2ijka(h1_oa,h2_oa,h3_oa,p1_va) )
       ( v2ijka("abab")(h1_oa,h2_ob,h3_oa,p1_vb)  =  1.0 * v2tensors.v2ijka(h1_oa,h2_ob,h3_oa,p1_vb) )
       ( v2ijka("bbbb")(h1_ob,h2_ob,h3_ob,p1_vb)  =  1.0 * v2tensors.v2ijka(h1_ob,h2_ob,h3_ob,p1_vb) );

    sch( v2ijkl("aaaa")(h1_oa,h2_oa,h3_oa,h4_oa)  =  1.0 * v2tensors.v2ijkl(h1_oa,h2_oa,h3_oa,h4_oa) )
       ( v2ijkl("abab")(h1_oa,h2_ob,h3_oa,h4_ob)  =  1.0 * v2tensors.v2ijkl(h1_oa,h2_ob,h3_oa,h4_ob) )
       ( v2ijkl("bbbb")(h1_ob,h2_ob,h3_ob,h4_ob)  =  1.0 * v2tensors.v2ijkl(h1_ob,h2_ob,h3_ob,h4_ob) );

    sch( v2iabc("aaaa")(h1_oa,p2_va,p3_va,p4_va)  =  1.0 * v2tensors.v2iabc(h1_oa,p2_va,p3_va,p4_va) )
       ( v2iabc("abab")(h1_oa,p2_vb,p3_va,p4_vb)  =  1.0 * v2tensors.v2iabc(h1_oa,p2_vb,p3_va,p4_vb) )
       ( v2iabc("bbbb")(h1_ob,p2_vb,p3_vb,p4_vb)  =  1.0 * v2tensors.v2iabc(h1_ob,p2_vb,p3_vb,p4_vb) );

    sch( v2abcd("aaaa")(p1_va,p2_va,p3_va,p4_va)  =  1.0 * v2tensors.v2abcd(p1_va,p2_va,p3_va,p4_va) )
       ( v2abcd("abab")(p1_va,p2_vb,p3_va,p4_vb)  =  1.0 * v2tensors.v2abcd(p1_va,p2_vb,p3_va,p4_vb) )
       ( v2abcd("bbbb")(p1_vb,p2_vb,p3_vb,p4_vb)  =  1.0 * v2tensors.v2abcd(p1_vb,p2_vb,p3_vb,p4_vb) );

    // clang-format on

    sch.execute();
  }

  void deallocate(Scheduler& sch) {
    CCSE_Tensors<T>::deallocate_list(sch, v2ijab, v2iajb, v2ijka, v2ijkl, v2iabc, v2abcd);
  }
};
