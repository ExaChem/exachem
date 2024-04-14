/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_ccsd_cs_ann.hpp"

using CCEType = double;
CCSE_Tensors<CCEType> _a021;
Tensor<CCEType>       a22_abab, a22_aaaa, a22_bbbb;
TiledIndexSpace       o_alpha, v_alpha, o_beta, v_beta;

Tensor<CCEType>       _a01V, _a02V, _a007V;
CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009, _a017,
  _a019, _a020; //_a022

Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only

template<typename T>
void ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               const Tensor<T>& t1_aa, const Tensor<T>& t2_abab, const Tensor<T>& t2_aaaa,
               std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
    (t2_aaaa_temp() = t2_aaaa(),
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa),
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa),
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)")

    (_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (_a02("aa")(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a02( aa )(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a03("aa")(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a03( aa )(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de()  =  2.0 * _a01V() * _a01V(),
    "de()  =  2.0 * _a01V() * _a01V()")
    (de() += -1.0 * _a02("aa")(h1_oa, h2_oa, cind) * _a02("aa")(h2_oa, h1_oa, cind),
    "de() += -1.0 * _a02( aa )(h1_oa, h2_oa, cind) * _a02( aa )(h2_oa, h1_oa, cind)")
    (de() +=  1.0 * _a03("aa")(h1_oa, p1_va, cind) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "de() +=  1.0 * _a03( aa )(h1_oa, p1_va, cind) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    ;
  // clang-format on
}

template<typename T>
void ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");
  auto [p2]   = MO.labels<1>("virt");
  auto [h1]   = MO.labels<1>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov( aa )(h1_oa, p2_va)")
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01("aa")(h1_oa, h1_oa, cind))
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h1_oa, p1_va, cind) * _a01( aa )(h2_oa, h1_oa, cind)")      // o2vm
    (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va),
    "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06("aa")(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a06( aa )(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov( aa )(h2_oa, p2_va, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa),
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a02V(cind)")                      // ovm
    (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h1_oa, p2_va),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h1_oa, p2_va)")
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
    (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (_a06("aa")(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    "_a06( aa )(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind)")                                      // ovm
    (_a06("aa")(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a06( aa )(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h1_oa, cind)")                   // o2vm
    (_a01("aa")(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a01( aa )(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h2_oa, h1_oa, cind)")                                    // o2m
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h2_oa, h1_oa, cind) * _a06("aa")(p2_va, h2_oa, cind),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h2_oa, h1_oa, cind) * _a06( aa )(p2_va, h2_oa, cind)")        // o2vm
    // (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo("aa")(h2_oa, h1_oa), // MOVED ABOVE
    // "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo( aa )(h2_oa, h1_oa)")                        // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv("aa")(p2_va, p1_va),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv( aa )(p2_va, p1_va)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& t2_aaaa,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind]   = CI.labels<1>("all");
  auto [p3, p4] = MO.labels<2>("virt");
  auto [h1, h2] = MO.labels<2>("occ");

  auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
  auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];
  auto hw        = sch.ec().exhw();
  // auto rank      = sch.ec().pg().rank();

  //_a022("abab")(p1_va,p2_vb,p2_va,p1_vb) = _a021("aa")(p1_va,p2_va,cind) *
  //_a021("bb")(p2_vb,p1_vb,cind)
  Tensor<T>        a22_abab_tmp{v_alpha, v_beta, v_alpha, v_beta};
  LabeledTensor<T> lhs_  = a22_abab_tmp(p1_va, p2_vb, p2_va, p1_vb);
  LabeledTensor<T> rhs1_ = _a021("aa")(p1_va, p2_va, cind);
  LabeledTensor<T> rhs2_ = _a021("bb")(p2_vb, p1_vb, cind);

  // mult op constructor
  auto lhs_lbls  = lhs_.labels();
  auto rhs1_lbls = rhs1_.labels();
  auto rhs2_lbls = rhs2_.labels();

  IntLabelVec lhs_int_labels_;
  IntLabelVec rhs1_int_labels_;
  IntLabelVec rhs2_int_labels_;

  auto labels{lhs_lbls};
  labels.insert(labels.end(), rhs1_lbls.begin(), rhs1_lbls.end());
  labels.insert(labels.end(), rhs2_lbls.begin(), rhs2_lbls.end());

  internal::update_labels(labels);

  lhs_lbls  = IndexLabelVec(labels.begin(), labels.begin() + lhs_.labels().size());
  rhs1_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size(),
                            labels.begin() + lhs_.labels().size() + rhs1_.labels().size());
  rhs2_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size() + rhs1_.labels().size(),
                            labels.begin() + lhs_.labels().size() + rhs1_.labels().size() +
                              rhs2_.labels().size());
  lhs_.set_labels(lhs_lbls);
  rhs1_.set_labels(rhs1_lbls);
  rhs2_.set_labels(rhs2_lbls);

  // fillin_int_labels
  std::map<TileLabelElement, int> primary_labels_map;
  int                             cnt = -1;
  for(const auto& lbl: lhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: rhs1_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: rhs2_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: lhs_.labels()) {
    lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  for(const auto& lbl: rhs1_.labels()) {
    rhs1_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  for(const auto& lbl: rhs2_.labels()) {
    rhs2_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  // todo: validate

  using TensorElType1 = T;
  using TensorElType2 = T;
  using TensorElType3 = T;

  // determine set of all labels for do_work
  IndexLabelVec all_labels{lhs_.labels()};
  all_labels.insert(all_labels.end(), rhs1_.labels().begin(), rhs1_.labels().end());
  all_labels.insert(all_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());
  // LabelLoopNest loop_nest{all_labels};

  // execute-bufacc
  IndexLabelVec lhs_labels{lhs_.labels()};
  IndexLabelVec rhs1_labels{rhs1_.labels()};
  IndexLabelVec rhs2_labels{rhs2_.labels()};
  IndexLabelVec all_rhs_labels{rhs1_.labels()};
  all_rhs_labels.insert(all_rhs_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());

  // compute the reduction labels
  std::sort(lhs_labels.begin(), lhs_labels.end());
  auto unique_labels = internal::unique_entries_by_primary_label(all_rhs_labels);
  std::sort(unique_labels.begin(), unique_labels.end());
  IndexLabelVec reduction_labels; //{reduction.begin(), reduction.end()};
  std::set_difference(unique_labels.begin(), unique_labels.end(), lhs_labels.begin(),
                      lhs_labels.end(), std::back_inserter(reduction_labels));

  std::vector<int> rhs1_map_output;
  std::vector<int> rhs2_map_output;
  std::vector<int> rhs1_map_reduction;
  std::vector<int> rhs2_map_reduction;
  // const auto&      lhs_lbls = lhs_.labels();
  for(auto& lbl: rhs1_labels) {
    auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
    if(it_out != lhs_lbls.end()) rhs1_map_output.push_back(it_out - lhs_lbls.begin());
    else rhs1_map_output.push_back(-1);

    // auto it_red = std::find(reduction.begin(), reduction.end(), lbl);
    auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
    if(it_red != reduction_labels.end())
      rhs1_map_reduction.push_back(it_red - reduction_labels.begin());
    else rhs1_map_reduction.push_back(-1);
  }

  for(auto& lbl: rhs2_labels) {
    auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
    if(it_out != lhs_lbls.end()) rhs2_map_output.push_back(it_out - lhs_lbls.begin());
    else rhs2_map_output.push_back(-1);

    auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
    if(it_red != reduction_labels.end())
      rhs2_map_reduction.push_back(it_red - reduction_labels.begin());
    else rhs2_map_reduction.push_back(-1);
  }

  auto ctensor = lhs_.tensor();
  auto atensor = rhs1_.tensor();
  auto btensor = rhs2_.tensor();
  // for(auto itval=loop_nest.begin(); itval!=loop_nest.end(); ++itval) {}

  auto compute_v4_term = [=](const IndexVector& cblkid, span<T> cbuf) {
    auto& memHostPool = tamm::RMMMemoryManager::getInstance().getHostMemoryPool();

    // compute blockids from the loop indices. itval is the loop index
    // execute_bufacc(ec, hw);
    LabelLoopNest lhs_loop_nest{lhs_.labels()};
    IndexVector   translated_ablockid, translated_bblockid, translated_cblockid;
    auto          it    = lhs_loop_nest.begin();
    auto          itval = *it;
    for(; it != lhs_loop_nest.end(); ++it) {
      itval = *it;
      // auto        it   = ivec.begin();
      IndexVector c_block_id{itval};
      translated_cblockid = internal::translate_blockid(c_block_id, lhs_);
      if(translated_cblockid == cblkid) break;
    }

    // execute
    // const auto& ldist = lhs_.tensor().distribution();
    // for(const auto& lblockid: lhs_loop_nest) {
    //   const auto translated_lblockid = internal::translate_blockid(lblockid, lhs_);
    //   if(lhs_.tensor().is_non_zero(translated_lblockid) &&
    //       std::get<0>(ldist.locate(translated_lblockid)) == rank) {
    //     lambda(lblockid);
    //   }

    const size_t csize = ctensor.block_size(translated_cblockid);
    std::memset(cbuf.data(), 0, csize * sizeof(TensorElType1));
    const auto& cdims = ctensor.block_dims(translated_cblockid);

    SizeVec cdims_sz;
    for(const auto v: cdims) { cdims_sz.push_back(v); }

    AddBuf<TensorElType1, TensorElType2, TensorElType3>* ab{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    TensorElType2* th_a{nullptr};
    TensorElType3* th_b{nullptr};
    auto&          thandle = GPUStreamPool::getInstance().getStream();

    ab =
      new AddBuf<TensorElType1, TensorElType2, TensorElType3>{th_a, th_b, {}, translated_cblockid};
#else
    gpuStream_t thandle{};
    ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{ctensor, {}, translated_cblockid};
#endif

    // LabelLoopNest inner_loop{reduction_lbls};
    LabelLoopNest inner_loop{reduction_labels};

    // int loop_counter = 0;

    TensorElType1* cbuf_dev_ptr{nullptr};
    TensorElType1* cbuf_tmp_dev_ptr{nullptr};
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    auto& memDevicePool = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

    if(hw == ExecutionHW::GPU) {
      cbuf_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));
      cbuf_tmp_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));

      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
    }
#endif

    for(const auto& inner_it_val: inner_loop) { // k

      IndexVector a_block_id(rhs1_.labels().size());

      for(size_t i = 0; i < rhs1_map_output.size(); i++) {
        if(rhs1_map_output[i] != -1) { a_block_id[i] = itval[rhs1_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs1_map_reduction.size(); i++) {
        if(rhs1_map_reduction[i] != -1) { a_block_id[i] = inner_it_val[rhs1_map_reduction[i]]; }
      }

      const auto translated_ablockid = internal::translate_blockid(a_block_id, rhs1_);
      if(!atensor.is_non_zero(translated_ablockid)) continue;

      IndexVector b_block_id(rhs2_.labels().size());

      for(size_t i = 0; i < rhs2_map_output.size(); i++) {
        if(rhs2_map_output[i] != -1) { b_block_id[i] = itval[rhs2_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs2_map_reduction.size(); i++) {
        if(rhs2_map_reduction[i] != -1) { b_block_id[i] = inner_it_val[rhs2_map_reduction[i]]; }
      }

      const auto translated_bblockid = internal::translate_blockid(b_block_id, rhs2_);
      if(!btensor.is_non_zero(translated_bblockid)) continue;

      // compute block size and allocate buffers for abuf and bbuf
      const size_t asize = atensor.block_size(translated_ablockid);
      const size_t bsize = btensor.block_size(translated_bblockid);

      TensorElType2* abuf{nullptr};
      TensorElType3* bbuf{nullptr};
      abuf = static_cast<TensorElType2*>(memHostPool.allocate(asize * sizeof(TensorElType2)));
      bbuf = static_cast<TensorElType3*>(memHostPool.allocate(bsize * sizeof(TensorElType3)));

      atensor.get(translated_ablockid, {abuf, asize});
      btensor.get(translated_bblockid, {bbuf, bsize});

      const auto& adims = atensor.block_dims(translated_ablockid);
      const auto& bdims = btensor.block_dims(translated_bblockid);

      // changed cscale from 0 to 1 to aggregate on cbuf
      T cscale{1};

      SizeVec adims_sz, bdims_sz;
      for(const auto v: adims) { adims_sz.push_back(v); }
      for(const auto v: bdims) { bdims_sz.push_back(v); }

      // A*B
      {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        TensorElType2* abuf_dev{nullptr};
        TensorElType3* bbuf_dev{nullptr};
        if(hw == ExecutionHW::GPU) {
          abuf_dev =
            static_cast<TensorElType2*>(memDevicePool.allocate(asize * sizeof(TensorElType2)));
          bbuf_dev =
            static_cast<TensorElType3*>(memDevicePool.allocate(bsize * sizeof(TensorElType3)));

          gpuMemcpyAsync<TensorElType2>(abuf_dev, abuf, asize, gpuMemcpyHostToDevice, thandle);
          gpuMemcpyAsync<TensorElType3>(bbuf_dev, bbuf, bsize, gpuMemcpyHostToDevice, thandle);
        }
#endif

        kernels::block_multiply<T, TensorElType1, TensorElType2, TensorElType3>(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
          abuf_dev, bbuf_dev,
#endif
          thandle, 1.0, abuf, adims_sz, rhs1_int_labels_, bbuf, bdims_sz, rhs2_int_labels_, cscale,
          cbuf.data(), cdims_sz, lhs_int_labels_, hw, false, cbuf_dev_ptr, cbuf_tmp_dev_ptr);

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
        if(hw == ExecutionHW::GPU) {
          memDevicePool.deallocate(abuf_dev, asize * sizeof(TensorElType2));
          memDevicePool.deallocate(bbuf_dev, bsize * sizeof(TensorElType3));
        }
#endif
      } // A * B

      memHostPool.deallocate(abuf, asize * sizeof(TensorElType2));
      memHostPool.deallocate(bbuf, bsize * sizeof(TensorElType3));
    } // end of reduction loop

    // add the computed update to the tensor
    {
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
      // copy to host
      if(hw == ExecutionHW::GPU) {
        TensorElType1* cbuf_tmp{nullptr};
        cbuf_tmp = static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
        std::memset(cbuf_tmp, 0, csize * sizeof(TensorElType1));
        gpuMemcpyAsync<TensorElType1>(cbuf_tmp, cbuf_dev_ptr, csize, gpuMemcpyDeviceToHost,
                                      thandle);
        // cbuf+=cbuf_tmp
        gpuStreamSynchronize(thandle);
        blas::axpy(csize, TensorElType1{1}, cbuf_tmp, 1, cbuf.data(), 1);

        // free cbuf_dev_ptr
        memDevicePool.deallocate(static_cast<void*>(cbuf_dev_ptr), csize * sizeof(TensorElType1));
        memDevicePool.deallocate(static_cast<void*>(cbuf_tmp_dev_ptr),
                                 csize * sizeof(TensorElType1));

        memHostPool.deallocate(cbuf_tmp, csize * sizeof(TensorElType1));
      }
#endif
      // ctensor.add(translated_cblockid, cbuf);
      // for (size_t i=0;i<csize;i++) dbuf[i] = cbuf[i];
    }

    delete ab;
  };

  a22_abab = Tensor<T>{{v_alpha, v_beta, v_alpha, v_beta}, compute_v4_term};

  // clang-format off
  sch
    (_a017("aa")(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)               = -1.0  * chol3d_ov("aa")(h2_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind),
    "_a006( aa )(h2_oa, h1_oa)               = -1.0  * chol3d_ov( aa )(h2_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
    (_a007V(cind)                            =  2.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    "_a007V(cind)                            =  2.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa)")
    (_a009("aa")(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009( aa )(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a021("aa")(p2_va, p1_va, cind)         = -0.5  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    "_a021( aa )(p2_va, p1_va, cind)         = -0.5  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa)")
    (_a021("aa")(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a021( aa )(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv( aa )(p2_va, p1_va, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021("aa")(p1_va, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021( aa )(p1_va, p2_va, cind)")
    (_a008("aa")(h2_oa, h1_oa, cind)         =  1.0  * _a009("aa")(h2_oa, h1_oa, cind),
    "_a008( aa )(h2_oa, h1_oa, cind)         =  1.0  * _a009( aa )(h2_oa, h1_oa, cind)")
    (_a009("aa")(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a009( aa )(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo( aa )(h2_oa, h1_oa, cind)")
    .exact_copy(_a009("bb")(h2_ob,h1_ob,cind),_a009("aa")(h2_ob,h1_ob,cind))
    .exact_copy(_a021("bb")(p2_vb,p1_vb,cind),_a021("aa")(p2_vb,p1_vb,cind))
    (_a001("aa")(p1_va, p2_va)               = -2.0  * _a021("aa")(p1_va, p2_va, cind) * _a007V(cind),
    "_a001( aa )(p1_va, p2_va)               = -2.0  * _a021( aa )(p1_va, p2_va, cind) * _a007V(cind)")
    (_a001("aa")(p1_va, p2_va)              += -1.0  * _a017("aa")(p1_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a001( aa )(p1_va, p2_va)              += -1.0  * _a017( aa )(p1_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)              +=  1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a007V(cind),
    "_a006( aa )(h2_oa, h1_oa)              +=  1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a007V(cind)")
    (_a006("aa")(h3_oa, h1_oa)              += -1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a008("aa")(h3_oa, h2_oa, cind),
    "_a006( aa )(h3_oa, h1_oa)              += -1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a008( aa )(h3_oa, h2_oa, cind)")
    (_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009("aa")(h2_oa, h1_oa, cind) * _a009("bb")(h1_ob, h2_ob, cind),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009( aa )(h2_oa, h1_oa, cind) * _a009( bb )(h1_ob, h2_ob, cind)")
    (_a020("aaaa")(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a021("aa")(p2_va, p1_va, cind),
    "_a020( aaaa )(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a021( aa )(p2_va, p1_va, cind)")
    .exact_copy(_a020("baba")(p2_vb, h2_oa, p1_vb, h1_oa),_a020("aaaa")(p2_vb, h2_oa, p1_vb, h1_oa))
    (_a020("aaaa")(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004("aaaa")(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020( aaaa )(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004( aaaa )(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020("baab")(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004("aaaa")(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020( baab )(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004( aaaa )(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004("abab")(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob),
    "_a020( baba )(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004( abab )(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob)")
    (_a017("aa")(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo("aa")(h1_oa, h2_oa, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo( aa )(h1_oa, h2_oa, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a001("aa")(p2_va, p1_va)                 += -1.0  * f1_vv("aa")(p2_va, p1_va),
    "_a001( aa )(p2_va, p1_va)                 += -1.0  * f1_vv( aa )(p2_va, p1_va)")
    (_a001("aa")(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "_a001( aa )(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * f1_oo("aa")(h2_oa, h1_oa),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * f1_oo( aa )(h2_oa, h1_oa)")
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)")
    .exact_copy(_a017("bb")(p1_vb, h1_ob, cind), _a017("aa")(p1_vb, h1_ob, cind))
    .exact_copy(_a006("bb")(h1_ob, h2_ob), _a006("aa")(h1_ob, h2_ob))
    .exact_copy(_a001("bb")(p1_vb, p2_vb), _a001("aa")(p1_vb, p2_vb))
    .exact_copy(_a021("bb")(p1_vb, p2_vb, cind), _a021("aa")(p1_vb, p2_vb, cind))
    .exact_copy(_a020("bbbb")(p1_vb, h1_ob, p2_vb, h2_ob), _a020("aaaa")(p1_vb, h1_ob, p2_vb, h2_ob))

    (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020("bbbb")(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    "i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020(bbbb)(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob)")
    (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baab")(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    "i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baab)(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baba)(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob)")
    .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017("aa")(p1_va, h1_oa, cind) * _a017("bb")(p1_vb, h2_ob, cind),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017( aa )(p1_va, h1_oa, cind) * _a017( bb )(p1_vb, h2_ob, cind)");

    sch
    // (_a022("abab")(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021("aa")(p1_va,p2_va,cind) * _a021("bb")(p2_vb,p1_vb,cind),
    // "_a022( abab )(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021( aa )(p1_va,p2_va,cind) * _a021( bb )(p2_vb,p1_vb,cind)")
    (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)");


    sch(_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004("abab")(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004( abab )(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p1_vb, p2_vb),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p1_vb, p2_vb)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001("aa")(p1_va, p2_va),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001( aa )(p1_va, p2_va)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("aa")(h1_oa, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( aa )(h1_oa, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("bb")(h1_ob, h2_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( bb )(h1_ob, h2_ob)")
    ;
  // clang-format on
}

template<typename T>
std::tuple<double, double> cd_ccsd_cs_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, Tensor<T> dt1_full,
  Tensor<T> dt2_full, bool ccsd_restart, std::string ccsd_fp, bool computeTData) {
  SystemData& sys_data    = chem_env.sys_data;
  int         maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  int         ndiis       = chem_env.ioptions.ccsd_options.ndiis;
  double      thresh      = chem_env.ioptions.ccsd_options.threshold;
  bool        writet      = chem_env.ioptions.ccsd_options.writet;
  int         writet_iter = chem_env.ioptions.ccsd_options.writet_iter;
  double      zshiftl     = chem_env.ioptions.ccsd_options.lshift;
  bool        profile     = chem_env.ioptions.ccsd_options.profile_ccsd;
  double      residual    = 0.0;
  double      energy      = 0.0;
  int         niter       = 0;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_vir_alpha = static_cast<TAMM_SIZE>(sys_data.n_vir_alpha);

  std::string t1file = ccsd_fp + ".t1amp";
  std::string t2file = ccsd_fp + ".t2amp";

  std::cout.precision(15);

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [cind]              = CI.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
  auto [h3_ob, h4_ob] = o_beta.labels<2>("all");

  Tensor<T> d_e{};

  Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, {2, 2}};

  CCSE_Tensors<T> f1_oo{MO, {O, O}, "f1_oo", {"aa", "bb"}};
  CCSE_Tensors<T> f1_ov{MO, {O, V}, "f1_ov", {"aa", "bb"}};
  CCSE_Tensors<T> f1_vv{MO, {V, V}, "f1_vv", {"aa", "bb"}};

  CCSE_Tensors<T> chol3d_oo{MO, {O, O, CI}, "chol3d_oo", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_ov{MO, {O, V, CI}, "chol3d_ov", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_vv{MO, {V, V, CI}, "chol3d_vv", {"aa", "bb"}};

  std::vector<CCSE_Tensors<T>> f1_se{f1_oo, f1_ov, f1_vv};
  std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo, chol3d_ov, chol3d_vv};

  _a01V = {CI};
  _a02  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02", {"aa"}};
  _a03  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03", {"aa"}};
  _a004 = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004", {"aaaa", "abab"}};

  t2_aaaa_temp = {v_alpha, v_alpha, o_alpha, o_alpha};
  i0_temp      = {v_beta, v_alpha, o_beta, o_alpha};

  // Intermediates
  // T1
  _a02V = {CI};
  _a01  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01", {"aa"}};
  _a04  = CCSE_Tensors<T>{MO, {O, O}, "_a04", {"aa"}};
  _a05  = CCSE_Tensors<T>{MO, {O, V}, "_a05", {"aa", "bb"}};
  _a06  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06", {"aa"}};

  // T2
  _a007V = {CI};
  _a001  = CCSE_Tensors<T>{MO, {V, V}, "_a001", {"aa", "bb"}};
  _a006  = CCSE_Tensors<T>{MO, {O, O}, "_a006", {"aa", "bb"}};

  _a008 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008", {"aa"}};
  _a009 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009", {"aa", "bb"}};
  _a017 = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017", {"aa", "bb"}};
  _a021 = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021", {"aa", "bb"}};

  _a019 = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"abab"}};
  // _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"abab"}};
  _a020 = CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}};

  double total_ccsd_mem =
    sum_tensor_sizes(t1_aa, t2_aaaa, t2_abab, d_f1, r1_aa, r2_abab, cv3d, d_e, i0_temp,
                     t2_aaaa_temp, _a01V) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(_a02, _a03);

  for(size_t ri = 0; ri < d_r1s.size(); ri++)
    total_ccsd_mem += sum_tensor_sizes(d_r1s[ri], d_r2s[ri], d_t1s[ri], d_t2s[ri]);

  // Intermediates
  // const double v4int_size         = CCSE_Tensors<T>::sum_tensor_sizes_list(_a022);
  double total_ccsd_mem_tmp = sum_tensor_sizes(_a02V, _a007V) + /*v4int_size +*/
                              CCSE_Tensors<T>::sum_tensor_sizes_list(_a01, _a04, _a05, _a06, _a001,
                                                                     _a004, _a006, _a008, _a009,
                                                                     _a017, _a019, _a020, _a021);

  if(!ccsd_restart) total_ccsd_mem += total_ccsd_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for Closed Shell Cholesky CCSD calculation: "
              << std::fixed << std::setprecision(2) << total_ccsd_mem << " GiB" << std::endl;
    // std::cout << " (V^4 intermediate size: " << std::fixed << std::setprecision(2) << v4int_size
    //           << " GiB)" << std::endl;
  }
  check_memory_requirements(ec, total_ccsd_mem);

  print_ccsd_header(ec.print());

  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  sch.allocate(t2_aaaa);
  sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, _a02, _a03);

  // clang-format off
  sch
    (chol3d_oo("aa")(h3_oa,h4_oa,cind) = cv3d(h3_oa,h4_oa,cind))
    (chol3d_ov("aa")(h3_oa,p2_va,cind) = cv3d(h3_oa,p2_va,cind))
    (chol3d_vv("aa")(p1_va,p2_va,cind) = cv3d(p1_va,p2_va,cind))
    (chol3d_oo("bb")(h3_ob,h4_ob,cind) = cv3d(h3_ob,h4_ob,cind))
    (chol3d_ov("bb")(h3_ob,p1_vb,cind) = cv3d(h3_ob,p1_vb,cind))
    (chol3d_vv("bb")(p1_vb,p2_vb,cind) = cv3d(p1_vb,p2_vb,cind))

    (f1_oo("aa")(h3_oa,h4_oa) = d_f1(h3_oa,h4_oa))
    (f1_ov("aa")(h3_oa,p2_va) = d_f1(h3_oa,p2_va))
    (f1_vv("aa")(p1_va,p2_va) = d_f1(p1_va,p2_va))
    (f1_oo("bb")(h3_ob,h4_ob) = d_f1(h3_ob,h4_ob))
    (f1_ov("bb")(h3_ob,p1_vb) = d_f1(h3_ob,p1_vb))
    (f1_vv("bb")(p1_vb,p2_vb) = d_f1(p1_vb,p2_vb));
  // clang-format on

  sch.execute();

  if(!ccsd_restart) {
    // allocate all intermediates
    sch.allocate(_a02V, _a007V);
    CCSE_Tensors<T>::allocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                   _a017, _a019, _a020, _a021); //_a022
    sch.execute();

    // clang-format off
        sch
            (r1_aa() = 0)
            (r2_abab() = 0)
            (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_ov("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h3_oa, p2_va, cind))
            .exact_copy(_a004("abab")(p1_va, p1_vb, h3_oa, h3_ob), _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob))
            ;
    // clang-format on

    sch.execute(exhw);

    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;
        // clang-format off
            sch
               ((d_t1s[off])()  = t1_aa())
               ((d_t2s[off])()  = t2_abab())
               .execute();
        // clang-format on

        ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);
        ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se);
        ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

        sch.execute(exhw, profile);

        std::tie(residual, energy) = rest_cs(ec, MO, r1_aa, r2_abab, t1_aa, t2_abab, d_e,
                                             d_r1_residual, d_r2_residual, p_evl_sorted, zshiftl,
                                             n_occ_alpha, n_vir_alpha);

        update_r2(ec, r2_abab());
        // clang-format off
            sch((d_r1s[off])() = r1_aa())
                ((d_r2s[off])() = r2_abab())
                .execute();
        // clang-format on

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

        if(writet && (((iter + 1) % writet_iter == 0) || (residual < thresh))) {
          write_to_disk(t1_aa, t1file);
          write_to_disk(t2_abab, t2file);
        }

        if(residual < thresh) { break; }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec.pg().rank() == 0) {
        std::cout << " MICROCYCLE DIIS UPDATE:";
        std::cout.width(21);
        std::cout << std::right << std::min(titer + ndiis, maxiter) + 1 << std::endl;
      }

      std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
      std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
      std::vector<Tensor<T>>              next_t{t1_aa, t2_abab};
      diis<T>(ec, rs, ts, next_t);
    }

    if(profile && ec.print()) {
      std::string   profile_csv = ccsd_fp + "_profile.csv";
      std::ofstream pds(profile_csv, std::ios::out);
      if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
      std::string header = "ID;Level;OP;total_op_time_min;total_op_time_max;total_op_time_avg;";
      header += "get_time_min;get_time_max;get_time_avg;gemm_time_min;";
      header += "gemm_time_max;gemm_time_avg;acc_time_min;acc_time_max;acc_time_avg";
      pds << header << std::endl;
      pds << ec.get_profile_data().str() << std::endl;
      pds.close();
    }

    // deallocate all intermediates
    sch.deallocate(_a02V, _a007V, d_r1_residual, d_r2_residual);
    CCSE_Tensors<T>::deallocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                     _a017, _a019, _a020, _a021); //_a022

  } // no restart
  else {
    ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

    sch.execute(exhw, profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  sys_data.ccsd_corr_energy = energy;

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CCSD"]["final_energy"]["total"] = sys_data.scf_energy + energy;
    chem_env.write_json_data("CCSD");
  }

  sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::deallocate_list(sch, _a02, _a03);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  sch.execute();

  if(computeTData) {
    Tensor<T> d_t1 = dt1_full;
    Tensor<T> d_t2 = dt2_full;

    // IndexVector perm1 = {1,0,3,2};
    // IndexVector perm2 = {0,1,3,2};
    // IndexVector perm3 = {1,0,2,3};

    // t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb,t2_baba, t2_abba, t2_baab
    Tensor<T> t1_bb{v_beta, o_beta};
    Tensor<T> t2_bbbb{v_beta, v_beta, o_beta, o_beta};
    Tensor<T> t2_baba{v_beta, v_alpha, o_beta, o_alpha};
    Tensor<T> t2_abba{v_alpha, v_beta, o_beta, o_alpha};
    Tensor<T> t2_baab{v_beta, v_alpha, o_alpha, o_beta};

    // clang-format off
    sch.allocate(t1_bb,t2_bbbb,t2_baba,t2_abba,t2_baab)
    .exact_copy(t1_bb(p1_vb,h3_ob),  t1_aa(p1_vb,h3_ob))
    .exact_copy(t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob),  t2_aaaa(p1_vb,p2_vb,h3_ob,h4_ob)).execute();

    // .exact_copy(t2_baba(p1_vb,p2_va,h3_ob,h4_oa),  t2_abab(p1_vb,p2_va,h3_ob,h4_oa),true,1.0,perm)
    // .exact_copy(t2_abba(p1_va,p2_vb,h3_ob,h4_oa),  t2_abab(p1_va,p2_vb,h3_ob,h4_oa),true,-1.0)
    // .exact_copy(t2_baab(p1_vb,p2_va,h3_oa,h4_ob),  t2_abab(p1_vb,p2_va,h3_oa,h4_ob),true,-1.0)

    // sch.exact_copy(t2_baba,t2_abab,true, 1.0,perm1);
    // sch.exact_copy(t2_abba,t2_abab,true,-1.0,perm2);
    // sch.exact_copy(t2_baab,t2_abab,true,-1.0,perm3);

    sch
    (t2_baba(p2_vb,p1_va,h4_ob,h3_oa) =        t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_abba(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_baab(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))

    (d_t1(p1_va,h3_oa)             = t1_aa(p1_va,h3_oa))
    (d_t1(p1_vb,h3_ob)             = t1_bb(p1_vb,h3_ob))
    (d_t2(p1_va,p2_va,h3_oa,h4_oa) = t2_aaaa(p1_va,p2_va,h3_oa,h4_oa))
    (d_t2(p1_va,p2_vb,h3_oa,h4_ob) = t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (d_t2(p1_vb,p2_vb,h3_ob,h4_ob) = t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob))

    (d_t2(p1_vb,p2_va,h3_ob,h4_oa) = t2_baba(p1_vb,p2_va,h3_ob,h4_oa))
    (d_t2(p1_va,p2_vb,h3_ob,h4_oa) = t2_abba(p1_va,p2_vb,h3_ob,h4_oa))
    (d_t2(p1_vb,p2_va,h3_oa,h4_ob) = t2_baab(p1_vb,p2_va,h3_oa,h4_ob))
    .deallocate(t1_bb,t2_bbbb,t2_baba,t2_abba,t2_baab)
    .execute();
    // clang-format on
  }

  sch.deallocate(t2_aaaa).execute();

  return std::make_tuple(residual, energy);
}

using T = double;
template std::tuple<double, double> cd_ccsd_cs_driver<T>(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, Tensor<T> dt1_full,
  Tensor<T> dt2_full, bool ccsd_restart, std::string out_fp, bool computeTData);
