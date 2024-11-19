/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/tamm.hpp"

namespace tamm {

template<typename T>
inline void jacobi(ExecutionContext& ec, Tensor<T>& d_r, Tensor<T>& d_t, T shift, bool transpose,
                   std::vector<double>& evl_sorted, const TAMM_SIZE& n_occ_alpha,
                   const TAMM_SIZE& n_occ_beta) {
  // EXPECTS(transpose == false);
  block_for(ec, d_r(), [&](IndexVector blockid) {
    const TAMM_SIZE rsize = d_r.block_size(blockid);
    std::vector<T>  rbuf(rsize);
    d_r.get(blockid, rbuf);

    const TAMM_SIZE tsize = d_t.block_size(blockid);

    std::vector<T> tbuf(tsize);

    auto& rtiss       = d_r.tiled_index_spaces();
    auto  rblock_dims = d_r.block_dims(blockid);

    TAMM_SIZE           noab = n_occ_alpha + n_occ_beta;
    std::vector<double> p_evl_sorted_occ(noab);
    std::vector<double> p_evl_sorted_virt(evl_sorted.size() - noab);
    std::copy(evl_sorted.begin(), evl_sorted.begin() + noab, p_evl_sorted_occ.begin());
    std::copy(evl_sorted.begin() + noab, evl_sorted.end(), p_evl_sorted_virt.begin());

    if(d_r.num_modes() == 2) {
      auto ioff  = rtiss[0].tile_offset(blockid[0]);
      auto joff  = rtiss[1].tile_offset(blockid[1]);
      auto isize = rblock_dims[0];
      auto jsize = rblock_dims[1];

      if(!transpose) {
        for(auto i = 0U, c = 0U; i < isize; i++) {
          for(auto j = 0U; j < jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (-p_evl_sorted_virt[ioff + i] + p_evl_sorted_occ[joff + j] + shift);
          }
        }
      }
      else {
        for(auto i = 0U, c = 0U; i < isize; i++) {
          for(auto j = 0U; j < jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (p_evl_sorted_occ[ioff + i] - p_evl_sorted_virt[joff + j] + shift);
          }
        }
      }
      d_t.add(blockid, tbuf);
    }
    else if(d_r.num_modes() == 4) {
      auto rblock_offset = d_r.block_offsets(blockid);

      std::vector<size_t> ioff;
      for(auto x: rblock_offset) { ioff.push_back(x); }
      std::vector<size_t> isize;
      for(auto x: rblock_dims) { isize.push_back(x); }

      if(!transpose) {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] = rbuf[c] /
                          (-p_evl_sorted_virt[ioff[0] + i0] - p_evl_sorted_virt[ioff[1] + i1] +
                           p_evl_sorted_occ[ioff[2] + i2] + p_evl_sorted_occ[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      else {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] =
                  rbuf[c] /
                  (p_evl_sorted_occ[ioff[0] + i0] + p_evl_sorted_occ[ioff[1] + i1] -
                   p_evl_sorted_virt[ioff[2] + i2] - p_evl_sorted_virt[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      d_t.add(blockid, tbuf);
    }
    else if(d_r.num_modes() == 6) {
      auto rblock_offset = d_r.block_offsets(blockid);

      std::vector<size_t> ioff;
      for(auto x: rblock_offset) { ioff.push_back(x); }
      std::vector<size_t> isize;
      for(auto x: rblock_dims) { isize.push_back(x); }

      if(!transpose) {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++) {
                for(auto i4 = 0U; i4 < isize[4]; i4++) {
                  for(auto i5 = 0U; i5 < isize[5]; i5++, c++) {
                    tbuf[c] =
                      rbuf[c] /
                      (-p_evl_sorted_virt[ioff[0] + i0] - p_evl_sorted_virt[ioff[1] + i1] -
                       p_evl_sorted_virt[ioff[2] + i2] + p_evl_sorted_occ[ioff[3] + i3] +
                       p_evl_sorted_occ[ioff[4] + i4] + p_evl_sorted_occ[ioff[5] + i5] + shift);
                  }
                }
              }
            }
          }
        }
      }
      else {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++) {
                for(auto i4 = 0U; i4 < isize[4]; i4++) {
                  for(auto i5 = 0U; i5 < isize[5]; i5++, c++) {
                    tbuf[c] =
                      rbuf[c] /
                      (p_evl_sorted_occ[ioff[0] + i0] + p_evl_sorted_occ[ioff[1] + i1] +
                       p_evl_sorted_occ[ioff[2] + i2] - p_evl_sorted_virt[ioff[3] + i3] -
                       p_evl_sorted_virt[ioff[4] + i4] - p_evl_sorted_virt[ioff[5] + i5] + shift);
                  }
                }
              }
            }
          }
        }
      }
      d_t.add(blockid, tbuf);
    }
    else {
      assert(0); // @todo implement
    }
  });
  // GA_Sync();
}

template<typename T>
inline void jacobi_cs(ExecutionContext& ec, Tensor<T>& d_r, Tensor<T>& d_t, T shift, bool transpose,
                      std::vector<double>& evl_sorted, const TAMM_SIZE& n_occ_alpha,
                      const TAMM_SIZE& n_vir_alpha, const bool not_spin_orbital = false) {
  // EXPECTS(transpose == false);
  block_for(ec, d_r(), [&](IndexVector blockid) {
    const TAMM_SIZE rsize = d_r.block_size(blockid);
    std::vector<T>  rbuf(rsize);
    d_r.get(blockid, rbuf);

    const TAMM_SIZE tsize = d_t.block_size(blockid);

    std::vector<T> tbuf(tsize);

    auto& rtiss       = d_r.tiled_index_spaces();
    auto  rblock_dims = d_r.block_dims(blockid);

    TAMM_SIZE           noa  = n_occ_alpha;
    TAMM_SIZE           noab = n_occ_alpha + n_occ_alpha;
    TAMM_SIZE           nva  = n_vir_alpha;
    std::vector<double> p_evl_sorted_occ(noa);
    std::vector<double> p_evl_sorted_virt(nva);
    std::copy(evl_sorted.begin(), evl_sorted.begin() + noa, p_evl_sorted_occ.begin());
    if(not_spin_orbital)
      std::copy(evl_sorted.begin() + noa, evl_sorted.begin() + noa + nva,
                p_evl_sorted_virt.begin());
    else
      std::copy(evl_sorted.begin() + noab, evl_sorted.begin() + noab + nva,
                p_evl_sorted_virt.begin());

    if(d_r.num_modes() == 2) {
      auto ioff  = rtiss[0].tile_offset(blockid[0]);
      auto joff  = rtiss[1].tile_offset(blockid[1]);
      auto isize = rblock_dims[0];
      auto jsize = rblock_dims[1];

      if(!transpose) {
        for(auto i = 0U, c = 0U; i < isize; i++) {
          for(auto j = 0U; j < jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (-p_evl_sorted_virt[ioff + i] + p_evl_sorted_occ[joff + j] + shift);
          }
        }
      }
      else {
        for(auto i = 0U, c = 0U; i < isize; i++) {
          for(auto j = 0U; j < jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (p_evl_sorted_occ[ioff + i] - p_evl_sorted_virt[joff + j] + shift);
          }
        }
      }
      d_t.add(blockid, tbuf);
    }
    else if(d_r.num_modes() == 4) {
      auto rblock_offset = d_r.block_offsets(blockid);

      std::vector<size_t> ioff;
      for(auto x: rblock_offset) { ioff.push_back(x); }
      std::vector<size_t> isize;
      for(auto x: rblock_dims) { isize.push_back(x); }

      if(!transpose) {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] = rbuf[c] /
                          (-p_evl_sorted_virt[ioff[0] + i0] - p_evl_sorted_virt[ioff[1] + i1] +
                           p_evl_sorted_occ[ioff[2] + i2] + p_evl_sorted_occ[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      else {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] =
                  rbuf[c] /
                  (p_evl_sorted_occ[ioff[0] + i0] + p_evl_sorted_occ[ioff[1] + i1] -
                   p_evl_sorted_virt[ioff[2] + i2] - p_evl_sorted_virt[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      d_t.add(blockid, tbuf);
    }
    else {
      assert(0); // @todo implement
    }
  });
  // GA_Sync();
}

template<typename T>
inline void jacobi_eom(ExecutionContext& ec, LabeledTensor<T> d_r_lt, LabeledTensor<T> d_t_lt,
                       T shift, bool transpose, std::vector<double>& evl_sorted,
                       const TAMM_SIZE n_occ_alpha, const TAMM_SIZE n_occ_beta) {
  // EXPECTS(transpose == false);
  Tensor<T> d_r = d_r_lt.tensor();
  Tensor<T> d_t = d_t_lt.tensor();

  block_for(ec, d_r_lt, [&](IndexVector bid) {
    IndexVector blockid = internal::translate_blockid(bid, d_r_lt);

    const TAMM_SIZE rsize = d_r.block_size(blockid);
    std::vector<T>  rbuf(rsize);
    d_r.get(blockid, rbuf);

    const TAMM_SIZE tsize = d_t.block_size(blockid);

    std::vector<T> tbuf(tsize);

    // auto& rtiss      = d_r.tiled_index_spaces();
    auto rblock_dims   = d_r.block_dims(blockid);
    auto rblock_offset = d_r.block_offsets(blockid);

    TAMM_SIZE           noab = n_occ_alpha + n_occ_beta;
    std::vector<double> p_evl_sorted_occ(noab);
    std::vector<double> p_evl_sorted_virt(evl_sorted.size() - noab);
    std::copy(evl_sorted.begin(), evl_sorted.begin() + noab, p_evl_sorted_occ.begin());
    std::copy(evl_sorted.begin() + noab, evl_sorted.end(), p_evl_sorted_virt.begin());

    if(d_r.num_modes() == 3) {
      std::vector<size_t> ioff;
      for(auto x: rblock_offset) { ioff.push_back(x); }
      std::vector<size_t> isize;
      for(auto x: rblock_dims) { isize.push_back(x); }

      if(!transpose) {
        for(auto i = 0U, c = 0U; i < isize[0]; i++) {
          for(auto j = 0U; j < isize[1]; j++, c++) {
            tbuf[c] =
              rbuf[c] / (-p_evl_sorted_virt[ioff[0] + i] + p_evl_sorted_occ[ioff[1] + j] + shift);
          }
        }
      }
      else {
        for(auto i = 0U, c = 0U; i < isize[0]; i++) {
          for(auto j = 0U; j < isize[1]; j++, c++) {
            tbuf[c] =
              rbuf[c] / (p_evl_sorted_occ[ioff[0] + i] - p_evl_sorted_virt[ioff[1] + j] + shift);
          }
        }
      }
      // auto last_id = rtiss[2].translate(blockid[2], d_t.tiled_index_spaces()[2]);
      // blockid[2] = last_id;
      d_t.add(blockid, tbuf);
    }
    else if(d_r.num_modes() == 5) {
      std::vector<size_t> ioff;
      for(auto x: rblock_offset) { ioff.push_back(x); }
      std::vector<size_t> isize;
      for(auto x: rblock_dims) { isize.push_back(x); }

      if(!transpose) {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] = rbuf[c] /
                          (-p_evl_sorted_virt[ioff[0] + i0] - p_evl_sorted_virt[ioff[1] + i1] +
                           p_evl_sorted_occ[ioff[2] + i2] + p_evl_sorted_occ[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      else {
        for(auto i0 = 0U, c = 0U; i0 < isize[0]; i0++) {
          for(auto i1 = 0U; i1 < isize[1]; i1++) {
            for(auto i2 = 0U; i2 < isize[2]; i2++) {
              for(auto i3 = 0U; i3 < isize[3]; i3++, c++) {
                tbuf[c] =
                  rbuf[c] /
                  (p_evl_sorted_occ[ioff[0] + i0] + p_evl_sorted_occ[ioff[1] + i1] -
                   p_evl_sorted_virt[ioff[2] + i2] - p_evl_sorted_virt[ioff[3] + i3] + shift);
              }
            }
          }
        }
      }
      // auto last_id = rtiss[4].translate(blockid[4], d_t.tiled_index_spaces()[4]);
      // blockid[4] = last_id;
      d_t.add(blockid, tbuf);
    }
    else {
      assert(0); // @todo implement
    }
  });
  // GA_Sync();
}

/**
 * @brief dot product between data held in two labeled tensors. Corresponding
 * elements are multiplied.
 *
 * This routine ignores permutation symmetry, and associated symmetrizatin
 * factors
 *
 * @tparam T Type of elements in both tensors
 * @param ec Execution context in which this function is invoked
 * @param lta Labeled tensor A
 * @param ltb labeled Tensor B
 * @return dot product A . B
 */
template<typename T>
inline T ddot(ExecutionContext& ec, LabeledTensor<T> lta, LabeledTensor<T> ltb) {
  T ret = 0;
  block_for(ec.pg(), lta, [&](IndexVector blockid) {
    Tensor<T>       atensor = lta.tensor();
    const TAMM_SIZE asize   = atensor.block_size(blockid);
    std::vector<T>  abuf(asize);

    Tensor<T>       btensor = ltb.tensor();
    const TAMM_SIZE bsize   = btensor.block_size(blockid);
    std::vector<T>  bbuf(bsize);

    const size_t sz = asize;
    for(size_t i = 0; i < sz; i++) { ret += abuf[i] * bbuf[i]; }
  });
  return ret;
}

/**
 * @brief DIIS routine
 * @tparam T Type of element in each tensor
 * @param ec Execution context in which this function invoked
 * @param[in] d_rs Vector of R tensors
 * @param[in] d_ts Vector of T tensors
 * @param[out] d_t Vector of T tensors produced by DIIS
 * @pre d_rs.size() == d_ts.size()
 * @pre 0<=i<d_rs.size(): d_rs[i].size() == d_t.size()
 * @pre 0<=i<d_ts.size(): d_ts[i].size() == d_t.size()
 */
template<typename T>
inline void diis(ExecutionContext& ec, std::vector<std::vector<Tensor<T>>>& d_rs,
                 std::vector<std::vector<Tensor<T>>>& d_ts, std::vector<Tensor<T>> d_t) {
  EXPECTS(d_t.size() == d_rs.size());
  size_t ntensors = d_t.size();
  EXPECTS(ntensors > 0);
  size_t ndiis = d_rs[0].size();
  EXPECTS(ndiis > 0);
  for(auto i = 0U; i < ntensors; i++) { EXPECTS(d_rs[i].size() == ndiis); }

  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Matrix A     = Matrix::Zero(ndiis + 1, ndiis + 1);
  Vector b     = Vector::Zero(ndiis + 1, 1);
  for(auto k = 0U; k < ntensors; k++) {
    for(auto i = 0U; i < ndiis; i++) {
      for(auto j = i; j < ndiis; j++) {
        Tensor<T> d_r1{};
        Tensor<T>::allocate(&ec, d_r1);
        Tensor<T>& t1 = d_rs[k].at(i);
        Tensor<T>& t2 = d_rs[k].at(j);
        // A(i, j) += ddot(ec, (*d_rs[k]->at(i))(), (*d_rs[k]->at(j))());
        Scheduler{ec}(d_r1() = t1() * t2()).execute();

        A(i, j) += get_scalar(d_r1);
        Tensor<T>::deallocate(d_r1);
      }
    }
  }

  for(auto i = 0U; i < ndiis; i++) {
    for(auto j = i; j < ndiis; j++) { A(j, i) = A(i, j); }
  }
  for(auto i = 0U; i < ndiis; i++) {
    A(i, ndiis) = -1.0;
    A(ndiis, i) = -1.0;
  }

  b(ndiis, 0) = -1;

  // Solve AX = B
  // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)
  // Vector x = A.colPivHouseholderQr().solve(b);
  Vector x = A.lu().solve(b);

  auto sch = Scheduler{ec};
  for(auto k = 0U; k < ntensors; k++) {
    Tensor<T>& dt = d_t[k];
    sch(dt() = 0);
    for(auto j = 0U; j < ndiis; j++) {
      auto& tb = d_ts[k].at(j);
      sch(dt() += x(j, 0) * tb());
    }
  }
  sch.execute();
}

} // namespace tamm
