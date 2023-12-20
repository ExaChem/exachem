/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "scf_guess.hpp"
#include <algorithm>
#include <iterator>

/// computes orbital occupation numbers for a subshell of size \c size created
/// by smearing
/// no more than \c ne electrons (corresponds to spherical averaging)
///
/// @param[in,out] occvec occupation vector, increments by \c size on return
/// @param[in] size the size of the subshell
/// @param[in,out] ne the number of electrons, on return contains the number of
/// "remaining" electrons
template<typename Real>
void scf_guess::subshell_occvec(Real& occvec, size_t size, size_t& ne) {
  const auto ne_alloc = (ne > 2 * size) ? 2 * size : ne;
  ne -= ne_alloc;
  occvec += ne_alloc;
}

/// @brief computes the number of electrons is s, p, d, and f shells
/// @return occupation vector corresponding to the ground state electronic
///         configuration of a neutral atom with atomic number \c Z
template<typename Real>
const std::vector<Real> scf_guess::compute_ao_occupation_vector(size_t Z) {
  std::vector<Real> occvec(4, 0.0);
  size_t            num_of_electrons = Z; // # of electrons to allocate

  // neutral atom electronic configurations from NIST:
  // http://www.nist.gov/pml/data/images/illo_for_2014_PT_1.PNG
  subshell_occvec(occvec[0], 1, num_of_electrons);   // 1s
  if(Z > 2) {                                        // Li+
    subshell_occvec(occvec[0], 1, num_of_electrons); // 2s
    subshell_occvec(occvec[1], 3, num_of_electrons); // 2p
  }
  if(Z > 10) {                                       // Na+
    subshell_occvec(occvec[0], 1, num_of_electrons); // 3s
    subshell_occvec(occvec[1], 3, num_of_electrons); // 3p
  }
  if(Z > 18) { // K .. Kr
    // NB 4s is singly occupied in K, Cr, and Cu
    size_t num_of_4s_electrons = (Z == 19 || Z == 24 || Z == 29) ? 1 : 2;
    num_of_electrons -= num_of_4s_electrons;
    subshell_occvec(occvec[0], 1, num_of_4s_electrons); // 4s
    subshell_occvec(occvec[2], 5, num_of_electrons);    // 3d
    subshell_occvec(occvec[1], 3, num_of_electrons);    // 4p
  }
  if(Z > 36) { // Rb .. I

    // NB 5s is singly occupied in Rb, Nb, Mo, Ru, Rh, and Ag
    size_t num_of_5s_electrons = (Z == 37 || Z == 41 || Z == 42 || Z == 44 || Z == 45 || Z == 47)
                                   ? 1
                                 : (Z == 46) ? 0
                                             : 2;
    num_of_electrons -= num_of_5s_electrons;
    subshell_occvec(occvec[0], 1, num_of_5s_electrons); // 5s
    subshell_occvec(occvec[2], 5, num_of_electrons);    // 4d
    subshell_occvec(occvec[1], 3, num_of_electrons);    // 5p
  }
  if(Z > 54) { // Cs .. Rn
    size_t num_of_6s_electrons = (Z == 55 || Z == 78 || Z == 79) ? 1 : 2;
    num_of_electrons -= num_of_6s_electrons;
    subshell_occvec(occvec[0], 1, num_of_6s_electrons); // 6s
    size_t num_of_5d_electrons = (Z == 57 || Z == 58 || Z == 64) ? 1 : 0;
    num_of_electrons -= num_of_5d_electrons;
    subshell_occvec(occvec[2], 5, num_of_5d_electrons); // 5d (Lanthanides)
    subshell_occvec(occvec[3], 7, num_of_electrons);    // 4f
    subshell_occvec(occvec[2], 5, num_of_electrons);    // 5d
    subshell_occvec(occvec[1], 3, num_of_electrons);    // 6p
  }
  if(Z > 86) {                                       // Fr .. Og
    subshell_occvec(occvec[0], 1, num_of_electrons); // 7s
    size_t num_of_6d_electrons = (Z == 89 || Z == 91 || Z == 92 || Z == 93 || Z == 96) ? 1
                                 : (Z == 90)                                           ? 2
                                                                                       : 0;
    num_of_electrons -= num_of_6d_electrons;
    subshell_occvec(occvec[2], 5, num_of_6d_electrons); // 6d (Actinides)
    subshell_occvec(occvec[3], 7, num_of_electrons);    // 5f
    size_t num_of_7p_electrons = (Z == 103) ? 1 : 0;
    num_of_electrons -= num_of_7p_electrons;
    subshell_occvec(occvec[1], 3, num_of_7p_electrons); // 7p (Lawrencium)
    subshell_occvec(occvec[2], 5, num_of_electrons);    // 6d
    subshell_occvec(occvec[1], 3, num_of_electrons);    // 7p
  }
  return occvec;
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {
  // compute number of atomic orbitals
  size_t natoms = atoms.size();
  size_t offset = 0;

  // compute the minimal basis density
  Matrix D = Matrix::Zero(natoms, 4);
  for(const auto& atom: atoms) {
    const auto Z      = atom.atomic_number;
    const auto occvec = scf_guess::compute_ao_occupation_vector(Z);
    for(int i = 0; i < 4; ++i) D(offset, i) = occvec[i];
    ++offset;
  }
  return D; // we use densities normalized to # of electrons/2
}

template<typename TensorType>
void compute_dipole_ints(ExecutionContext& ec, const SCFVars& spvars, Tensor<TensorType>& tensorX,
                         Tensor<TensorType>& tensorY, Tensor<TensorType>& tensorZ,
                         std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                         libint2::Operator otype) {
  using libint2::Atom;
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  const std::vector<Tile>&   AO_tiles       = spvars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = spvars.shell_tile_map;

  Engine engine(otype, max_nprim(shells), max_l(shells), 0);

  // engine.set(otype);
  // auto& buf = (engine.results());

  auto compute_dipole_ints_lambda = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    const TAMM_SIZE         size       = tensorX.block_size(blockid);
    auto                    block_dims = tensorX.block_dims(blockid);
    std::vector<TensorType> dbufX(size);
    std::vector<TensorType> dbufY(size);
    std::vector<TensorType> dbufZ(size);

    auto bd1 = block_dims[1];

    // auto s1 = blockid[0];
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl;
    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
      // this shell
      auto n1 = shells[s1].size();

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl;

      // cout << "screend shell pair list = " << s2spl << endl;
      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        // for (auto s2: spvars.obs_shellpair_list.at(s1)) {
        // auto s2 = blockid[1];
        // if (s2>s1) continue;

        if(s2 > s1) {
          auto s2spl = spvars.obs_shellpair_list.at(s2);
          if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
        }
        else {
          auto s2spl = spvars.obs_shellpair_list.at(s1);
          if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
        }

        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        std::vector<TensorType> tbufX(n1 * n2);
        std::vector<TensorType> tbufY(n1 * n2);
        std::vector<TensorType> tbufZ(n1 * n2);

        // compute shell pair; return is the pointer to the buffer
        const auto& buf = engine.compute(shells[s1], shells[s2]);
        EXPECTS(buf.size() >= 4);
        if(buf[0] == nullptr) continue;
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result

        Eigen::Map<const Matrix> buf_mat_X(buf[1], n1, n2);
        Eigen::Map<Matrix>(&tbufX[0], n1, n2) = buf_mat_X;
        Eigen::Map<const Matrix> buf_mat_Y(buf[2], n1, n2);
        Eigen::Map<Matrix>(&tbufY[0], n1, n2) = buf_mat_Y;
        Eigen::Map<const Matrix> buf_mat_Z(buf[3], n1, n2);
        Eigen::Map<Matrix>(&tbufZ[0], n1, n2) = buf_mat_Z;

        auto curshelloffset_i = 0U;
        auto curshelloffset_j = 0U;
        for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
        for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

        size_t c    = 0;
        auto   dimi = curshelloffset_i + AO_tiles[s1];
        auto   dimj = curshelloffset_j + AO_tiles[s2];

        for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) {
            dbufX[i * bd1 + j] = tbufX[c];
            dbufY[i * bd1 + j] = tbufY[c];
            dbufZ[i * bd1 + j] = tbufZ[c];
          }
        }
      } // s2
    }   // s1

    tensorX.put(blockid, dbufX);
    tensorY.put(blockid, dbufY);
    tensorZ.put(blockid, dbufZ);
  };

  block_for(ec, tensorX(), compute_dipole_ints_lambda);
}

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                        std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                        libint2::Operator otype) {
  using libint2::Atom;
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  const std::vector<Tile>&   AO_tiles       = scf_vars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = scf_vars.shell_tile_map;

  Engine engine(otype, max_nprim(shells), max_l(shells), 0);

  // engine.set(otype);

  if(otype == Operator::nuclear) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for(const auto& atom: atoms)
      q.push_back({static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}});

    engine.set_params(q);
  }

  auto& buf = (engine.results());

  auto compute_1body_ints_lambda = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    const TAMM_SIZE         size       = tensor1e.block_size(blockid);
    auto                    block_dims = tensor1e.block_dims(blockid);
    std::vector<TensorType> dbuf(size);

    auto bd1 = block_dims[1];

    // auto s1 = blockid[0];
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl;
    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
      // this shell
      auto n1 = shells[s1].size();

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl;

      // cout << "screend shell pair list = " << s2spl << endl;
      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        // for (auto s2: scf_vars.obs_shellpair_list.at(s1)) {
        // auto s2 = blockid[1];
        // if (s2>s1) continue;

        if(s2 > s1) {
          auto s2spl = scf_vars.obs_shellpair_list.at(s2);
          if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
        }
        else {
          auto s2spl = scf_vars.obs_shellpair_list.at(s1);
          if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
        }

        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        std::vector<TensorType> tbuf(n1 * n2);

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        if(buf[0] == nullptr) continue;
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        Eigen::Map<Matrix>(&tbuf[0], n1, n2) = buf_mat;
        // tensor1e.put(blockid, tbuf);

        auto curshelloffset_i = 0U;
        auto curshelloffset_j = 0U;
        for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
        for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

        size_t c    = 0;
        auto   dimi = curshelloffset_i + AO_tiles[s1];
        auto   dimj = curshelloffset_j + AO_tiles[s2];

        for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) { dbuf[i * bd1 + j] = tbuf[c]; }
        }

        // if(s1!=s2){
        //     std::vector<TensorType> ttbuf(n1*n2);
        //     Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
        //     // Matrix buf_mat_trans = buf_mat.transpose();
        //     size_t c = 0;
        //     for(size_t j = curshelloffset_j; j < dimj; j++) {
        //       for(size_t i = curshelloffset_i; i < dimi; i++, c++) {
        //             dbuf[j*block_dims[0]+i] = ttbuf[c];
        //       }
        //     }
        // }
        // tensor1e.put({s2,s1}, ttbuf);
      }
    }
    tensor1e.put(blockid, dbuf);
  };

  block_for(ec, tensor1e(), compute_1body_ints_lambda);
}

template<typename TensorType>
void compute_ecp_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                      std::vector<libecpint::GaussianShell>& shells,
                      std::vector<libecpint::ECP>&           ecps) {
  const std::vector<Tile>&   AO_tiles       = scf_vars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = scf_vars.shell_tile_map;

  int maxam     = 0;
  int ecp_maxam = 0;
  for(auto shell: shells)
    if(shell.l > maxam) maxam = shell.l;
  for(auto ecp: ecps)
    if(ecp.L > ecp_maxam) ecp_maxam = ecp.L;

  size_t  size_       = (maxam + 1) * (maxam + 2) * (maxam + 1) * (maxam + 2) / 4;
  double* buffer_     = new double[size_];
  double* buffer_sph_ = new double[size_];
  memset(buffer_, 0, size_ * sizeof(double));

  libecpint::ECPIntegral engine(maxam, ecp_maxam);

  auto compute_ecp_ints_lambda = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    const TAMM_SIZE         size       = tensor1e.block_size(blockid);
    auto                    block_dims = tensor1e.block_dims(blockid);
    std::vector<TensorType> dbuf(size);

    auto bd1 = block_dims[1];

    // auto s1 = blockid[0];
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl;
    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
      // this shell
      auto n1 = 2 * shells[s1].l + 1;

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl;

      // cout << "screend shell pair list = " << s2spl << endl;
      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        // for (auto s2: scf_vars.obs_shellpair_list.at(s1)) {
        // auto s2 = blockid[1];
        // if (s2>s1) continue;

        if(s2 > s1) {
          auto s2spl = scf_vars.obs_shellpair_list.at(s2);
          if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
        }
        else {
          auto s2spl = scf_vars.obs_shellpair_list.at(s1);
          if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
        }

        // auto bf2 = shell2bf[s2];
        auto n2 = 2 * shells[s2].l + 1;

        std::vector<TensorType> tbuf(n1 * n2);
        // cout << "s1,s2,n1,n2 = "  << s1 << "," << s2 <<
        //       "," << n1 <<"," << n2 <<endl;

        // compute shell pair; return is the pointer to the buffer
        const libecpint::GaussianShell& LibECPShell1 = shells[s1];
        const libecpint::GaussianShell& LibECPShell2 = shells[s2];
        size_ = shells[s1].ncartesian() * shells[s2].ncartesian();
        memset(buffer_, 0, size_ * sizeof(double));
        for(const auto& ecp: ecps) {
          libecpint::TwoIndex<double> results;
          engine.compute_shell_pair(ecp, LibECPShell1, LibECPShell2, results);
          // for (auto v: results.data) std::cout << std::setprecision(6) << v << std::endl;
          std::transform(results.data.begin(), results.data.end(), buffer_, buffer_,
                         std::plus<double>());
        }
        libint2::solidharmonics::tform(shells[s1].l, shells[s2].l, buffer_, buffer_sph_);

        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(&buffer_sph_[0], n1, n2);
        Eigen::Map<Matrix>(&tbuf[0], n1, n2) = buf_mat;
        // tensor1e.put(blockid, tbuf);

        auto curshelloffset_i = 0U;
        auto curshelloffset_j = 0U;
        for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
        for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

        size_t c    = 0;
        auto   dimi = curshelloffset_i + AO_tiles[s1];
        auto   dimj = curshelloffset_j + AO_tiles[s2];

        // cout << "curshelloffset_i,curshelloffset_j,dimi,dimj = "  << curshelloffset_i << "," <<
        // curshelloffset_j <<
        //       "," << dimi <<"," << dimj <<endl;

        for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) { dbuf[i * bd1 + j] = tbuf[c]; }
        }

        // if(s1!=s2){
        //     std::vector<TensorType> ttbuf(n1*n2);
        //     Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
        //     // Matrix buf_mat_trans = buf_mat.transpose();
        //     size_t c = 0;
        //     for(size_t j = curshelloffset_j; j < dimj; j++) {
        //       for(size_t i = curshelloffset_i; i < dimi; i++, c++) {
        //             dbuf[j*block_dims[0]+i] = ttbuf[c];
        //       }
        //     }
        // }
        // tensor1e.put({s2,s1}, ttbuf);
      }
    }
    tensor1e.put(blockid, dbuf);
  };
  block_for(ec, tensor1e(), compute_ecp_ints_lambda);
  delete[] buffer_;
  delete[] buffer_sph_;
}

template<typename TensorType>
void compute_pchg_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                       std::vector<std::pair<double, std::array<double, 3>>>& q,
                       libint2::BasisSet& shells, libint2::Operator otype) {
  using libint2::Atom;
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  const std::vector<Tile>&   AO_tiles       = scf_vars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = scf_vars.shell_tile_map;

  Engine engine(otype, max_nprim(shells), max_l(shells), 0);

  // engine.set(otype);
  engine.set_params(q);

  auto& buf = (engine.results());

  auto compute_pchg_ints_lambda = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    const TAMM_SIZE         size       = tensor1e.block_size(blockid);
    auto                    block_dims = tensor1e.block_dims(blockid);
    std::vector<TensorType> dbuf(size);

    auto bd1 = block_dims[1];

    // auto s1 = blockid[0];
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    // cout << "s1-start,end = " << s1range_start << ", " << s1range_end << endl;
    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      // auto bf1 = shell2bf[s1]; //shell2bf[s1]; // first basis function in
      // this shell
      auto n1 = shells[s1].size();

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      // cout << "s2-start,end = " << s2range_start << ", " << s2range_end << endl;

      // cout << "screend shell pair list = " << s2spl << endl;
      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        // for (auto s2: scf_vars.obs_shellpair_list.at(s1)) {
        // auto s2 = blockid[1];
        // if (s2>s1) continue;

        if(s2 > s1) {
          auto s2spl = scf_vars.obs_shellpair_list.at(s2);
          if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
        }
        else {
          auto s2spl = scf_vars.obs_shellpair_list.at(s1);
          if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
        }

        // auto bf2 = shell2bf[s2];
        auto n2 = shells[s2].size();

        std::vector<TensorType> tbuf(n1 * n2);
        // cout << "s1,s2,n1,n2 = "  << s1 << "," << s2 <<
        //       "," << n1 <<"," << n2 <<endl;

        // compute shell pair; return is the pointer to the buffer
        engine.compute(shells[s1], shells[s2]);
        if(buf[0] == nullptr) continue;
        // "map" buffer to a const Eigen Matrix, and copy it to the
        // corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        Eigen::Map<Matrix>(&tbuf[0], n1, n2) = buf_mat;
        // tensor1e.put(blockid, tbuf);

        auto curshelloffset_i = 0U;
        auto curshelloffset_j = 0U;
        for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
        for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

        size_t c    = 0;
        auto   dimi = curshelloffset_i + AO_tiles[s1];
        auto   dimj = curshelloffset_j + AO_tiles[s2];

        // cout << "curshelloffset_i,curshelloffset_j,dimi,dimj = "  << curshelloffset_i << "," <<
        // curshelloffset_j <<
        //       "," << dimi <<"," << dimj <<endl;

        for(size_t i = curshelloffset_i; i < dimi; i++) {
          for(size_t j = curshelloffset_j; j < dimj; j++, c++) { dbuf[i * bd1 + j] = tbuf[c]; }
        }

        // if(s1!=s2){
        //     std::vector<TensorType> ttbuf(n1*n2);
        //     Eigen::Map<Matrix>(ttbuf.data(),n2,n1) = buf_mat.transpose();
        //     // Matrix buf_mat_trans = buf_mat.transpose();
        //     size_t c = 0;
        //     for(size_t j = curshelloffset_j; j < dimj; j++) {
        //       for(size_t i = curshelloffset_i; i < dimi; i++, c++) {
        //             dbuf[j*block_dims[0]+i] = ttbuf[c];
        //       }
        //     }
        // }
        // tensor1e.put({s2,s1}, ttbuf);
      }
    }
    tensor1e.put(blockid, dbuf);
  };

  block_for(ec, tensor1e(), compute_pchg_ints_lambda);
}

template<typename TensorType>
void scf_diagonalize(Scheduler& sch, const SystemData& sys_data, SCFVars& scf_vars,
                     ScalapackInfo& scalapack_info, TAMMTensors& ttensors, EigenTensors& etensors) {
  auto rank = sch.ec().pg().rank();
  // const bool debug      = sys_data.options_map.scf_options.debug && rank==0;

  // solve F C = e S C by (conditioned) transformation to F' C' = e C',
  // where
  // F' = X.transpose() . F . X; the original C is obtained as C = X . C'

  // Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_alpha(X_a.transpose() * F_alpha * X_a);
  // C_alpha = X_a * eig_solver_alpha.eigenvectors();
  // Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_beta( X_b.transpose() * F_beta  * X_b);
  // C_beta  = X_b * eig_solver_beta.eigenvectors();

  const int64_t N      = sys_data.nbf_orig;
  const bool    is_uhf = sys_data.is_unrestricted;
  // const bool is_rhf = sys_data.is_restricted;
  const int nelectrons_alpha = sys_data.nelectrons_alpha;
  const int nelectrons_beta  = sys_data.nelectrons_beta;
  double    hl_gap           = 0;

#if defined(USE_SCALAPACK)

  if(scalapack_info.comm != MPI_COMM_NULL) {
    blacspp::Grid*                  blacs_grid       = scalapack_info.blacs_grid.get();
    scalapackpp::BlockCyclicDist2D* blockcyclic_dist = scalapack_info.blockcyclic_dist.get();

    auto desc_lambda = [&](const int64_t M, const int64_t N) {
      auto [M_loc, N_loc] = (*blockcyclic_dist).get_local_dims(M, N);
      return (*blockcyclic_dist).descinit_noerror(M, N, M_loc);
    };

    const auto& grid   = *blacs_grid;
    const auto  mb     = blockcyclic_dist->mb();
    const auto  Northo = sys_data.nbf;

    if(grid.ipr() >= 0 and grid.ipc() >= 0) {
      // TODO: Optimize intermediates here
      scalapackpp::BlockCyclicMatrix<double>
        // Fa_sca  ( grid, N,      N,      mb, mb ),
        // Xa_sca  ( grid, Northo, N,      mb, mb ), // Xa is row-major
        Fp_sca(grid, Northo, Northo, mb, mb), Ca_sca(grid, Northo, Northo, mb, mb),
        TMP1_sca(grid, N, Northo, mb, mb);

      auto desc_Fa = desc_lambda(N, N);
      auto desc_Xa = desc_lambda(Northo, N);

      tamm::to_block_cyclic_tensor(ttensors.F_alpha, ttensors.F_BC);
      scalapack_info.pg.barrier();

      auto Fa_tamm_lptr = ttensors.F_BC.access_local_buf();
      auto Xa_tamm_lptr = ttensors.X_alpha.access_local_buf();
      auto Ca_tamm_lptr = ttensors.C_alpha_BC.access_local_buf();

      // Compute TMP = F * X -> F * X**T (b/c row-major)
      // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::Trans,
      // 1., Fa_sca, Xa_sca, 0., TMP1_sca );
      scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::Trans, TMP1_sca.m(),
                         TMP1_sca.n(), desc_Fa[3], 1., Fa_tamm_lptr, 1, 1, desc_Fa, Xa_tamm_lptr, 1,
                         1, desc_Xa, 0., TMP1_sca.data(), 1, 1, TMP1_sca.desc());

      // Compute Fp = X**T * TMP -> X * TMP (b/c row-major)
      // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans,
      // 1., Xa_sca, TMP1_sca, 0., Fp_sca );
      scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans, Fp_sca.m(), Fp_sca.n(),
                         desc_Xa[3], 1., Xa_tamm_lptr, 1, 1, desc_Xa, TMP1_sca.data(), 1, 1,
                         TMP1_sca.desc(), 0., Fp_sca.data(), 1, 1, Fp_sca.desc());
      // Solve EVP
      std::vector<TensorType> eps_a(Northo);
      // scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
      //                       Fp_sca, eps_a.data(), Ca_sca );
      /*info=*/scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, Fp_sca.m(),
                                   Fp_sca.data(), 1, 1, Fp_sca.desc(), eps_a.data(), Ca_sca.data(),
                                   1, 1, Ca_sca.desc());

      // Backtransform TMP = X * Ca -> TMP**T = Ca**T * X
      // scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
      //                     1., Ca_sca, Xa_sca, 0., TMP2_sca );
      scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, desc_Xa[2], desc_Xa[3],
                         Ca_sca.m(), 1., Ca_sca.data(), 1, 1, Ca_sca.desc(), Xa_tamm_lptr, 1, 1,
                         desc_Xa, 0., Ca_tamm_lptr, 1, 1, desc_Xa);

      // Gather results
      // if(scalapack_info.pg.rank() == 0) C_alpha.resize(N, Northo);
      // TMP2_sca.gather_from(Northo, N, C_alpha.data(), Northo, 0, 0);

      if(is_uhf) {
        tamm::to_block_cyclic_tensor(ttensors.F_beta, ttensors.F_BC);
        scalapack_info.pg.barrier();
        Fa_tamm_lptr = ttensors.F_BC.access_local_buf();
        Xa_tamm_lptr = ttensors.X_alpha.access_local_buf();
        Ca_tamm_lptr = ttensors.C_beta_BC.access_local_buf();

        // Compute TMP = F * X -> F * X**T (b/c row-major)
        // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::Trans,
        //                     1., Fa_sca, Xa_sca, 0., TMP1_sca );
        scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::Trans, TMP1_sca.m(),
                           TMP1_sca.n(), desc_Fa[3], 1., Fa_tamm_lptr, 1, 1, desc_Fa, Xa_tamm_lptr,
                           1, 1, desc_Xa, 0., TMP1_sca.data(), 1, 1, TMP1_sca.desc());

        // Compute Fp = X**T * TMP -> X * TMP (b/c row-major)
        // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans,
        //                     1., Xa_sca, TMP1_sca, 0., Fp_sca );
        scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans, Fp_sca.m(),
                           Fp_sca.n(), desc_Xa[3], 1., Xa_tamm_lptr, 1, 1, desc_Xa, TMP1_sca.data(),
                           1, 1, TMP1_sca.desc(), 0., Fp_sca.data(), 1, 1, Fp_sca.desc());

        // Solve EVP
        std::vector<double> eps_b(Northo);
        // scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
        //                       Fp_sca, eps_b.data(), Ca_sca );
        /*info=*/scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, Fp_sca.m(),
                                     Fp_sca.data(), 1, 1, Fp_sca.desc(), eps_b.data(),
                                     Ca_sca.data(), 1, 1, Ca_sca.desc());

        // Backtransform TMP = X * Cb -> TMP**T = Cb**T * X
        // scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
        //                     1., Ca_sca, Xa_sca, 0., TMP2_sca );
        scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, desc_Xa[2], desc_Xa[3],
                           Ca_sca.m(), 1., Ca_sca.data(), 1, 1, Ca_sca.desc(), Xa_tamm_lptr, 1, 1,
                           desc_Xa, 0., Ca_tamm_lptr, 1, 1, desc_Xa);

        // Gather results
        // if(scalapack_info.pg.rank() == 0) C_beta.resize(N, Northo);
        // TMP2_sca.gather_from(Northo, N, C_beta.data(), Northo, 0, 0);

        if(!scf_vars.lshift_reset)
          hl_gap = std::min(eps_a[nelectrons_alpha], eps_b[nelectrons_beta]) -
                   std::max(eps_a[nelectrons_alpha - 1], eps_b[nelectrons_beta - 1]);
      }

    } // rank participates in ScaLAPACK call
  }

#else

  Matrix& C_alpha = etensors.C_alpha;
  Matrix& C_beta  = etensors.C_beta;

  const int64_t Northo_a = sys_data.nbf; // X_a.cols();
  // TODO: avoid eigen Fp
  Matrix              X_a;
  std::vector<double> eps_a;
  if(rank == 0) {
    // alpha
    Matrix Fp = tamm_to_eigen_matrix(ttensors.F_alpha);
    X_a       = tamm_to_eigen_matrix(ttensors.X_alpha);
    C_alpha.resize(N, Northo_a);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, N, Northo_a, N, 1.,
               Fp.data(), N, X_a.data(), Northo_a, 0., C_alpha.data(), N);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, Northo_a, Northo_a, N,
               1., X_a.data(), Northo_a, C_alpha.data(), N, 0., Fp.data(), Northo_a);
    eps_a.resize(Northo_a);
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Northo_a, Fp.data(), Northo_a,
                  eps_a.data());
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, Northo_a, N, Northo_a,
               1., Fp.data(), Northo_a, X_a.data(), Northo_a, 0., C_alpha.data(), Northo_a);
  }

  if(is_uhf) {
    const int64_t Northo_b = sys_data.nbf; // X_b.cols();
    if(rank == 0) {
      // beta
      Matrix Fp = tamm_to_eigen_matrix(ttensors.F_beta);
      C_beta.resize(N, Northo_b);
      Matrix& X_b = X_a;
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, N, Northo_b, N, 1.,
                 Fp.data(), N, X_b.data(), Northo_b, 0., C_beta.data(), N);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, Northo_b, Northo_b,
                 N, 1., X_b.data(), Northo_b, C_beta.data(), N, 0., Fp.data(), Northo_b);
      std::vector<double> eps_b(Northo_b);
      lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Northo_b, Fp.data(), Northo_b,
                    eps_b.data());
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, Northo_b, N, Northo_b,
                 1., Fp.data(), Northo_b, X_b.data(), Northo_b, 0., C_beta.data(), Northo_b);

      if(!scf_vars.lshift_reset) {
        hl_gap = std::min(eps_a[nelectrons_alpha], eps_b[nelectrons_beta]) -
                 std::max(eps_a[nelectrons_alpha - 1], eps_b[nelectrons_beta - 1]);
      }
    }
    if(!scf_vars.lshift_reset) sch.ec().pg().broadcast(&hl_gap, 1, 0);
  }
#endif

  if(!scf_vars.lshift_reset && is_uhf) {
    if(hl_gap < 1e-2) {
      scf_vars.lshift_reset = true;
      scf_vars.lshift       = 0.5;
      if(rank == 0) cout << "Resetting lshift to 0.5" << endl;
    }
  }
}

template void scf_guess::subshell_occvec<double>(double& occvec, size_t size, size_t& ne);

template const std::vector<double> scf_guess::compute_ao_occupation_vector<double>(size_t Z);

template void compute_dipole_ints<double>(ExecutionContext& ec, const SCFVars& spvars,
                                          Tensor<TensorType>& tensorX, Tensor<TensorType>& tensorY,
                                          Tensor<TensorType>&         tensorZ,
                                          std::vector<libint2::Atom>& atoms,
                                          libint2::BasisSet& shells, libint2::Operator otype);

template void compute_1body_ints<double>(ExecutionContext& ec, const SCFVars& scf_vars,
                                         Tensor<TensorType>&         tensor1e,
                                         std::vector<libint2::Atom>& atoms,
                                         libint2::BasisSet& shells, libint2::Operator otype);

template void compute_pchg_ints<double>(ExecutionContext& ec, const SCFVars& scf_vars,
                                        Tensor<TensorType>& tensor1e,
                                        std::vector<std::pair<double, std::array<double, 3>>>& q,
                                        libint2::BasisSet& shells, libint2::Operator otype);

template void compute_ecp_ints(ExecutionContext& ec, const SCFVars& scf_vars,
                               Tensor<TensorType>&                    tensor1e,
                               std::vector<libecpint::GaussianShell>& shells,
                               std::vector<libecpint::ECP>&           ecps);

template void scf_diagonalize<double>(Scheduler& sch, const SystemData& sys_data, SCFVars& scf_vars,
                                      ScalapackInfo& scalapack_info, TAMMTensors& ttensors,
                                      EigenTensors& etensors);
