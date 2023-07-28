/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "scf_guess.hpp"

/// computes orbital occupation numbers for a subshell of size \c size created
/// by smearing
/// no more than \c ne electrons (corresponds to spherical averaging)
///
/// @param[in,out] occvec occupation vector, increments by \c size on return
/// @param[in] size the size of the subshell
/// @param[in,out] ne the number of electrons, on return contains the number of
/// "remaining" electrons
template<typename Real>
void scf_guess::subshell_occvec(Real*& occvec, size_t size, size_t& ne) {
  const auto ne_alloc = (ne > 2 * size) ? 2 * size : ne;
  ne -= ne_alloc;
  // # of electrons / orbital compute as precisely as possible
  const double ne_per_orb = (ne_alloc % size == 0) ? static_cast<Real>(ne_alloc / size)
                                                   : (static_cast<Real>(ne_alloc)) / size;
  for(size_t f = 0; f != size; ++f) occvec[f] = ne_per_orb;
  occvec += size;
}

/// @param[in] Z the atomic number of the element
/// @throw if Z > 53
/// @return the number of STO-3G AOs for the element with atomic number \c Z
size_t scf_guess::sto3g_num_ao(size_t Z) {
  size_t nao;
  if(Z == 1 || Z == 2) // H, He
    nao = 1;
  else if(Z <= 10) // Li - Ne
    nao = 5;       // 2p is included even for Li and Be
  else if(Z <= 18) // Na - Ar
    nao = 9;       // 3p is included even for Na and Mg
  else if(Z <= 20) // K, Ca
    nao = 13;      // 4p is included
  else if(Z <= 36) // Sc - Kr
    nao = 18;
  else if(Z <= 38) // Rb, Sr
    nao = 22;      // 5p is included
  else if(Z <= 53) // Y - I
    nao = 27;
  else throw "STO-3G basis is not defined for elements with Z > 53";
  return nao;
}

/// @brief computes average orbital occupancies in the ground state of a neutral
///        atoms
/// @throw if Z > 53
/// @return occupation vector corresponding to the ground state electronic
///         configuration of a neutral atom with atomic number \c Z
///         corresponding to the orbital ordering in STO-3G basis
template<typename Real>
const std::vector<Real>& scf_guess::sto3g_ao_occupation_vector(size_t Z) {
  static std::vector<Real> occvec(27, 0.0);

  occvec.resize(sto3g_num_ao(Z));
  auto* occs_ptr = &occvec[0];
  auto& occs     = occs_ptr;

  size_t num_of_electrons = Z; // # of electrons to allocate

  // neutral atom electronic configurations from NIST:
  // http://www.nist.gov/pml/data/images/illo_for_2014_PT_1.PNG
  subshell_occvec(occs, 1, num_of_electrons);   // 1s
  if(Z > 2) {                                   // Li+
    subshell_occvec(occs, 1, num_of_electrons); // 2s
    subshell_occvec(occs, 3, num_of_electrons); // 2p
  }
  if(Z > 10) {                                  // Na+
    subshell_occvec(occs, 1, num_of_electrons); // 3s
    subshell_occvec(occs, 3, num_of_electrons); // 3p
  }
  if(18 < Z && Z <= 36) { // K .. Kr
    // NB 4s is singly occupied in K, Cr, and Cu
    size_t num_of_4s_electrons = (Z == 19 || Z == 24 || Z == 29) ? 1 : 2;
    num_of_electrons -= num_of_4s_electrons;
    subshell_occvec(occs, 1, num_of_4s_electrons); // 4s

    size_t num_of_4p_electrons = std::min(static_cast<decltype(Z)>(6), (Z > 30) ? Z - 30 : 0);
    num_of_electrons -= num_of_4p_electrons;
    subshell_occvec(occs, 3, num_of_4p_electrons); // 4p

    subshell_occvec(occs, 5, num_of_electrons); // 3d
  }
  if(36 < Z && Z <= 53) { // Rb .. I
    // 3d4s4p are fully occupied ...
    subshell_occvec(occs, 1, num_of_electrons); // 4s
    subshell_occvec(occs, 3, num_of_electrons); // 4p

    // NB 5s is singly occupied in Rb, Nb, Mo, Ru, Rh, and Ag
    size_t num_of_5s_electrons =
      (Z == 37 || Z == 41 || Z == 42 || Z == 44 || Z == 45 || Z == 47) ? 1 : 2;
    num_of_electrons -= num_of_5s_electrons;
    subshell_occvec(occs, 1, num_of_5s_electrons); // 5s

    size_t num_of_5p_electrons = std::min(static_cast<decltype(Z)>(6), (Z > 48) ? Z - 48 : 0);
    num_of_electrons -= num_of_5p_electrons;
    subshell_occvec(occs, 3, num_of_5p_electrons); // 5p

    subshell_occvec(occs, 5, num_of_electrons); // 3d
    subshell_occvec(occs, 5, num_of_electrons); // 4d
  }

  return occvec;
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {
  // compute number of atomic orbitals
  size_t nao = 0;
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    nao += scf_guess::sto3g_num_ao(Z);
  }

  // compute the minimal basis density
  Matrix D         = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto  Z      = atom.atomic_number;
    const auto& occvec = scf_guess::sto3g_ao_occupation_vector(Z);
    for(const auto& occ: occvec) {
      D(ao_offset, ao_offset) = occ;
      ++ao_offset;
    }
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

    // cout << "blockid: [" << blockid[0] <<"," << blockid[1] << "], dims(0,1) = " <<
    //  block_dims[0] << ", " << block_dims[1] << endl;

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

        // cout << buf[1].size() << endl;
        // cout << buf[2].size() << endl;
        // cout << buf[3].size() << endl;

        Eigen::Map<const Matrix> buf_mat_X(buf[1], n1, n2);
        Eigen::Map<Matrix>(&tbufX[0], n1, n2) = buf_mat_X;
        Eigen::Map<const Matrix> buf_mat_Y(buf[2], n1, n2);
        Eigen::Map<Matrix>(&tbufY[0], n1, n2) = buf_mat_Y;
        Eigen::Map<const Matrix> buf_mat_Z(buf[3], n1, n2);
        Eigen::Map<Matrix>(&tbufZ[0], n1, n2) = buf_mat_Z;

        // cout << "buf_mat_X :" << buf_mat_X << endl;
        // cout << "buf_mat_Y :" << buf_mat_Y << endl;
        // cout << "buf_mat_Z :" << buf_mat_Z << endl;

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

    // cout << "blockid: [" << blockid[0] <<"," << blockid[1] << "], dims(0,1) = " <<
    //  block_dims[0] << ", " << block_dims[1] << endl;

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

  block_for(ec, tensor1e(), compute_1body_ints_lambda);
}

template<typename TensorType>
void scf_diagonalize(Scheduler& sch, const SystemData& sys_data, ScalapackInfo& scalapack_info,
                     TAMMTensors& ttensors, EigenTensors& etensors) {
  auto rank = sch.ec().pg().rank();
  // const bool debug      = sys_data.options_map.scf_options.debug && rank==0;

  // solve F C = e S C by (conditioned) transformation to F' C' = e C',
  // where
  // F' = X.transpose() . F . X; the original C is obtained as C = X . C'

  // Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_alpha(X_a.transpose() * F_alpha * X_a);
  // C_alpha = X_a * eig_solver_alpha.eigenvectors();
  // Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_beta( X_b.transpose() * F_beta  * X_b);
  // C_beta  = X_b * eig_solver_beta.eigenvectors();

  Matrix& C_alpha = etensors.C;
  Matrix& C_beta  = etensors.C_beta;

  const int64_t N      = sys_data.nbf_orig;
  const bool    is_uhf = sys_data.is_unrestricted;
  // const bool is_rhf = sys_data.is_restricted;

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
        TMP1_sca(grid, N, Northo, mb, mb), TMP2_sca(grid, Northo, N, mb, mb);

      auto desc_Fa = desc_lambda(N, N);
      auto desc_Xa = desc_lambda(Northo, N);

      tamm::to_block_cyclic_tensor(ttensors.F_alpha, ttensors.F_BC);
      scalapack_info.pg.barrier();

      auto Fa_tamm_lptr = ttensors.F_BC.access_local_buf();
      auto Xa_tamm_lptr = ttensors.X_alpha.access_local_buf();

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
      auto info = scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, Fp_sca.m(),
                                      Fp_sca.data(), 1, 1, Fp_sca.desc(), eps_a.data(),
                                      Ca_sca.data(), 1, 1, Ca_sca.desc());

      // Backtransform TMP = X * Ca -> TMP**T = Ca**T * X
      // scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
      //                     1., Ca_sca, Xa_sca, 0., TMP2_sca );
      scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, TMP2_sca.m(),
                         TMP2_sca.n(), Ca_sca.m(), 1., Ca_sca.data(), 1, 1, Ca_sca.desc(),
                         Xa_tamm_lptr, 1, 1, desc_Xa, 0., TMP2_sca.data(), 1, 1, TMP2_sca.desc());

      // Gather results
      if(scalapack_info.pg.rank() == 0) C_alpha.resize(N, Northo);
      TMP2_sca.gather_from(Northo, N, C_alpha.data(), Northo, 0, 0);

      if(is_uhf) {
        tamm::to_block_cyclic_tensor(ttensors.F_beta, ttensors.F_BC);
        scalapack_info.pg.barrier();
        Fa_tamm_lptr = ttensors.F_BC.access_local_buf();
        Xa_tamm_lptr = ttensors.X_beta.access_local_buf();

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
        std::vector<double> eps_a(Northo);
        // scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
        //                       Fp_sca, eps_a.data(), Ca_sca );
        auto info = scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, Fp_sca.m(),
                                        Fp_sca.data(), 1, 1, Fp_sca.desc(), eps_a.data(),
                                        Ca_sca.data(), 1, 1, Ca_sca.desc());

        // Backtransform TMP = X * Cb -> TMP**T = Cb**T * X
        // scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
        //                     1., Ca_sca, Xa_sca, 0., TMP2_sca );
        scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, TMP2_sca.m(),
                           TMP2_sca.n(), Ca_sca.m(), 1., Ca_sca.data(), 1, 1, Ca_sca.desc(),
                           Xa_tamm_lptr, 1, 1, desc_Xa, 0., TMP2_sca.data(), 1, 1, TMP2_sca.desc());

        // Gather results
        if(scalapack_info.pg.rank() == 0) C_beta.resize(N, Northo);
        TMP2_sca.gather_from(Northo, N, C_beta.data(), Northo, 0, 0);
      }

    } // rank participates in ScaLAPACK call
  }

#else

  const int64_t Northo_a = sys_data.nbf; // X_a.cols();
  // TODO: avoid eigen Fp
  if(rank == 0) {
    // alpha
    Matrix Fp  = tamm_to_eigen_matrix(ttensors.F_alpha);
    Matrix X_a = tamm_to_eigen_matrix(ttensors.X_alpha);
    C_alpha.resize(N, Northo_a);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, N, Northo_a, N, 1.,
               Fp.data(), N, X_a.data(), Northo_a, 0., C_alpha.data(), N);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, Northo_a, Northo_a, N,
               1., X_a.data(), Northo_a, C_alpha.data(), N, 0., Fp.data(), Northo_a);
    std::vector<double> eps_a(Northo_a);
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
      Matrix X_b = tamm_to_eigen_matrix(ttensors.X_beta);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, N, Northo_b, N, 1.,
                 Fp.data(), N, X_b.data(), Northo_b, 0., C_beta.data(), N);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, Northo_b, Northo_b,
                 N, 1., X_b.data(), Northo_b, C_beta.data(), N, 0., Fp.data(), Northo_b);
      std::vector<double> eps_b(Northo_b);
      lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Northo_b, Fp.data(), Northo_b,
                    eps_b.data());
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, Northo_b, N, Northo_b,
                 1., Fp.data(), Northo_b, X_b.data(), Northo_b, 0., C_beta.data(), Northo_b);
    }
  }

#endif
}

template<typename TensorType>
void compute_initial_guess(ExecutionContext& ec, ScalapackInfo& scalapack_info,
                           SystemData& sys_data, SCFVars& scf_vars,
                           const std::vector<libint2::Atom>& atoms, const libint2::BasisSet& shells,
                           const std::string& basis, bool is_spherical, EigenTensors& etensors,
                           TAMMTensors& ttensors, int charge, int multiplicity) {
  auto ig1 = std::chrono::high_resolution_clock::now();

  Scheduler  sch{ec};
  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;

  const auto rank       = ec.pg().rank();
  const auto world_size = ec.pg().size();
  const auto N          = shells.nbf();
  const bool debug      = sys_data.options_map.scf_options.debug;

  // const Matrix& H   = etensors.H;
  Matrix& C_a     = etensors.C;
  Matrix& C_b     = etensors.C_beta;
  Matrix& D_a     = etensors.D;
  Matrix& D_b     = etensors.D_beta;
  Matrix& C_occ_a = etensors.C_occ;
  Matrix& G_a     = etensors.G;
  Matrix& G_b     = etensors.G_beta;

  // compute guess in minimal basis
  Matrix D_minbs_a;
  Matrix D_minbs_b;
  int    neutral_charge = sys_data.nelectrons + charge;
  double N_to_Neu       = (double) sys_data.nelectrons / neutral_charge;
  double Na_to_Neu      = (double) sys_data.nelectrons_alpha / neutral_charge;
  double Nb_to_Na       = (double) sys_data.nelectrons_beta / sys_data.nelectrons_alpha;
  if(is_rhf) D_minbs_a = N_to_Neu * compute_soad(atoms);
  if(is_uhf) {
    D_minbs_a = Na_to_Neu * compute_soad(atoms);
    D_minbs_b = Nb_to_Na * D_minbs_a;
  }

  libint2::BasisSet minbs("STO-3G", atoms);
  if(is_spherical) minbs.set_pure(true);
  else minbs.set_pure(false); // use cartesian gaussians

#ifndef NDEBUG
  std::tie(scf_vars.minbs_shellpair_list, scf_vars.minbs_shellpair_data) =
    compute_shellpairs(minbs);
#endif

  if(rank == 0)
    std::cout << std::endl
              << "Projecting minimal basis SOAD onto basis set specified (" << basis << ")" << endl;

  double precision = std::numeric_limits<double>::epsilon();
  // double precision = std::min(sys_data.options_map.scf_options.tol_int, 1e-3 *
  // sys_data.options_map.scf_options.conve);

  bool                     D_is_shelldiagonal = true;
  const libint2::BasisSet& obs                = shells;
  const libint2::BasisSet& D_bs               = minbs;

  Tensor<TensorType> F_dummy{scf_vars.tAOt, scf_vars.tAOt}; // not allocated

  auto do_t1 = std::chrono::high_resolution_clock::now();

  // construct the 2-electron repulsion integrals engine
  using libint2::BraKet;
  using libint2::Engine;
  using libint2::Operator;

  Engine engine(libint2::Operator::coulomb, std::max(obs.max_nprim(), D_bs.max_nprim()),
                std::max(obs.max_l(), D_bs.max_l()), 0);
  engine.set_precision(precision); // shellset-dependent precision control
                                   // will likely break positive
                                   // definiteness
                                   // stick with this simple recipe

  auto        shell2bf   = obs.shell2bf();
  auto        shell2bf_D = D_bs.shell2bf();
  const auto& buf        = engine.results();

  auto compute_2body_fock_general_lambda = [&](IndexVector blockid) {
    using libint2::Engine;

    auto s1        = blockid[0];
    auto bf1_first = shell2bf[s1];   // first basis function in this shell
    auto n1        = obs[s1].size(); // number of basis functions in this shell

    auto s2 = blockid[1];
    // if(s2>s1) return;

    auto sp12_iter = scf_vars.obs_shellpair_data.at(s1).begin();
    auto s2spl     = scf_vars.obs_shellpair_list.at(s1);
    auto s2_itr    = std::find(s2spl.begin(), s2spl.end(), s2);
    if(s2_itr == s2spl.end()) return;
    auto s2_pos = std::distance(s2spl.begin(), s2_itr);

    std::advance(sp12_iter, s2_pos);
    const auto* sp12 = sp12_iter->get();

    auto bf2_first = shell2bf[s2];
    auto n2        = obs[s2].size();

    for(decltype(s1) s3 = 0; s3 < D_bs.size(); ++s3) {
      auto bf3_first = shell2bf_D[s3];
      auto n3        = D_bs[s3].size();

      auto s4_begin = D_is_shelldiagonal ? s3 : 0;
      auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

#ifndef NDEBUG
      auto sp34_iter = scf_vars.minbs_shellpair_data.at(s3).begin();
#endif

      for(decltype(s1) s4 = s4_begin; s4 != s4_fence; ++s4) {

#ifndef NDEBUG
        auto s4spl  = scf_vars.minbs_shellpair_list.at(s3);
        auto s4_itr = std::find(s4spl.begin(), s4spl.end(), s4);
        if(s4_itr == s4spl.end()) continue;
        auto s4_pos = std::distance(s4spl.begin(), s4_itr);

        std::advance(sp34_iter, s4_pos);
        const auto* sp34 = sp34_iter->get();
#endif

        auto bf4_first = shell2bf_D[s4];
        auto n4        = D_bs[s4].size();

        // compute the permutational degeneracy (i.e. # of equivalents) of
        // the given shell set
        auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

        if(s3 >= s4) {
          auto s34_deg   = (s3 == s4) ? 1.0 : 2.0;
          auto s1234_deg = s12_deg * s34_deg;
// auto s1234_deg = s12_deg;
#ifndef NDEBUG
          engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(obs[s1], obs[s2], D_bs[s3], D_bs[s4],
                                                               sp12, sp34);
#else
          engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(obs[s1], obs[s2], D_bs[s3], D_bs[s4],
                                                               sp12); //,sp34);
#endif

          const double* buf_1234 = buf[0];
          if(buf_1234 != nullptr) {
            for(decltype(n1) f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for(decltype(n2) f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for(decltype(n3) f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value             = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    if(is_rhf) G_a(bf1, bf2) += 1.0 * D_minbs_a(bf3, bf4) * value_scal_by_deg;
                    if(is_uhf) {
                      G_a(bf1, bf2) += 1.0 * D_minbs_a(bf3, bf4) * value_scal_by_deg;
                      G_a(bf1, bf2) += 1.0 * D_minbs_b(bf3, bf4) * value_scal_by_deg;
                      G_b(bf1, bf2) += 1.0 * D_minbs_a(bf3, bf4) * value_scal_by_deg;
                      G_b(bf1, bf2) += 1.0 * D_minbs_b(bf3, bf4) * value_scal_by_deg;
                    }
                  }
                }
              }
            }
          }
        }

        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(obs[s1], D_bs[s3], obs[s2], D_bs[s4]);
        const double* buf_1324 = buf[0];
        if(buf_1324 == nullptr) continue; // if all integrals screened out, skip to next quartet

        for(decltype(n1) f1 = 0, f1324 = 0; f1 != n1; ++f1) {
          const auto bf1 = f1 + bf1_first;
          for(decltype(n3) f3 = 0; f3 != n3; ++f3) {
            const auto bf3 = f3 + bf3_first;
            for(decltype(n2) f2 = 0; f2 != n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(decltype(n4) f4 = 0; f4 != n4; ++f4, ++f1324) {
                const auto bf4               = f4 + bf4_first;
                const auto value             = buf_1324[f1324];
                const auto value_scal_by_deg = value * s12_deg;
                if(is_rhf) G_a(bf1, bf2) -= 0.5 * D_minbs_a(bf3, bf4) * value_scal_by_deg;
                if(is_uhf) {
                  G_a(bf1, bf2) -= 1.0 * D_minbs_a(bf3, bf4) * value_scal_by_deg;
                  G_b(bf1, bf2) -= 1.0 * D_minbs_b(bf3, bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  };

  // block_for(ec, F_dummy(), compute_2body_fock_general_lambda);
  for(Eigen::Index i1 = 0; i1 < etensors.taskmap.rows(); i1++) {
    for(Eigen::Index j1 = 0; j1 < etensors.taskmap.cols(); j1++) {
      if(etensors.taskmap(i1, j1) == -1 || etensors.taskmap(i1, j1) != rank) continue;
      IndexVector blockid{(tamm::Index) i1, (tamm::Index) j1};
      compute_2body_fock_general_lambda(blockid);
    }
  }
  ec.pg().barrier();

  auto do_t2   = std::chrono::high_resolution_clock::now();
  auto do_time = std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

  if(rank == 0 && debug)
    std::cout << "Time to compute Fock Matrix: " << std::fixed << std::setprecision(2) << do_time
              << " secs" << std::endl;

  D_minbs_a.resize(0, 0);
  if(is_uhf) D_minbs_b.resize(0, 0);

  {
    // symmetrize the result
    Matrix Gt_a = 0.5 * (G_a + G_a.transpose());
    G_a         = Gt_a;
    Gt_a.resize(0, 0);
    sch(ttensors.F_alpha_tmp() = 0).execute();
    eigen_to_tamm_tensor_acc(ttensors.F_alpha_tmp, G_a);
    ec.pg().barrier();
    // G_a.resize(0,0);
  }
  if(is_uhf) {
    Matrix Gt_b = 0.5 * (G_b + G_b.transpose());
    G_b         = Gt_b;
    Gt_b.resize(0, 0);
    sch(ttensors.F_beta_tmp() = 0).execute();
    eigen_to_tamm_tensor_acc(ttensors.F_beta_tmp, G_b);
    ec.pg().barrier();
    // G_b.resize(0,0);
  }

  // clang-format off
  sch (ttensors.F_alpha()  = ttensors.H1())
      (ttensors.F_alpha() += ttensors.F_alpha_tmp());
  // clang-format on

  if(is_uhf) {
    // clang-format off
    sch (ttensors.F_beta()   = ttensors.H1())
        (ttensors.F_beta()  += ttensors.F_beta_tmp());
    // clang-format on
  }
  sch.execute();

  scf_diagonalize<TensorType>(sch, sys_data, scalapack_info, ttensors, etensors);

  // compute density
  if(rank == 0) {
    if(is_rhf) {
      C_occ_a = C_a.leftCols(sys_data.nelectrons_alpha);
      D_a     = 2.0 * C_occ_a * C_occ_a.transpose();
    }
    if(is_uhf) {
      C_occ_a      = C_a.leftCols(sys_data.nelectrons_alpha);
      auto C_occ_b = C_b.leftCols(sys_data.nelectrons_beta);
      D_a          = C_occ_a * C_occ_a.transpose();
      D_b          = C_occ_b * C_occ_b.transpose();
    }
  }

  ec.pg().barrier();
  auto ig2    = std::chrono::high_resolution_clock::now();
  auto igtime = std::chrono::duration_cast<std::chrono::duration<double>>((ig2 - ig1)).count();
  if(rank == 0)
    std::cout << "Initial guess: " << std::fixed << std::setprecision(2) << igtime << " secs"
              << endl;
}

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
compute_initial_guess_taskinfo(ExecutionContext& ec, SystemData& sys_data, const SCFVars& scf_vars,
                               const std::vector<libint2::Atom>& atoms,
                               const libint2::BasisSet& shells, const std::string& basis,
                               bool is_spherical, EigenTensors& etensors, TAMMTensors& ttensors,
                               int charge, int multiplicity) {
  const auto rank = ec.pg().rank();

  // compute guess in minimal basis
  libint2::BasisSet minbs("STO-3G", atoms);
  if(is_spherical) minbs.set_pure(true);
  else minbs.set_pure(false); // use cartesian gaussians

  bool                     D_is_shelldiagonal = true;
  const libint2::BasisSet& obs                = shells;
  const libint2::BasisSet& D_bs               = minbs;

  auto shell2bf = obs.shell2bf();

  std::vector<int> s1vec;
  std::vector<int> s2vec;
  std::vector<int> ntask_vec;

  auto compute_2body_fock_general_lambda = [&](IndexVector blockid) {
    auto s1        = blockid[0];
    auto bf1_first = shell2bf[s1];   // first basis function in this shell
    auto n1        = obs[s1].size(); // number of basis functions in this shell

    auto s2 = blockid[1];
    // if(s2>s1) return;

    auto sp12_iter = scf_vars.obs_shellpair_data.at(s1).begin();
    auto s2spl     = scf_vars.obs_shellpair_list.at(s1);
    auto s2_itr    = std::find(s2spl.begin(), s2spl.end(), s2);
    if(s2_itr == s2spl.end()) return;
    auto s2_pos = std::distance(s2spl.begin(), s2_itr);

    std::advance(sp12_iter, s2_pos);

    size_t taskid = 0;

    for(decltype(s1) s3 = 0; s3 < D_bs.size(); ++s3) {
      auto s4_begin = D_is_shelldiagonal ? s3 : 0;
      auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

      for(decltype(s1) s4 = s4_begin; s4 != s4_fence; ++s4) {
        if(s3 >= s4) taskid++;
        taskid++;
      }
    }
    s1vec.push_back(s1);
    s2vec.push_back(s2);
    ntask_vec.push_back(taskid);
  };

  block_for(ec, ttensors.F_dummy(), compute_2body_fock_general_lambda);
  return std::make_tuple(s1vec, s2vec, ntask_vec);
}

template void scf_guess::subshell_occvec<double>(double*& occvec, size_t size, size_t& ne);

template const std::vector<double>& scf_guess::sto3g_ao_occupation_vector<double>(size_t Z);

template void compute_dipole_ints<double>(ExecutionContext& ec, const SCFVars& spvars,
                                          Tensor<TensorType>& tensorX, Tensor<TensorType>& tensorY,
                                          Tensor<TensorType>&         tensorZ,
                                          std::vector<libint2::Atom>& atoms,
                                          libint2::BasisSet& shells, libint2::Operator otype);

template void compute_1body_ints<double>(ExecutionContext& ec, const SCFVars& scf_vars,
                                         Tensor<TensorType>&         tensor1e,
                                         std::vector<libint2::Atom>& atoms,
                                         libint2::BasisSet& shells, libint2::Operator otype);

template void scf_diagonalize<double>(Scheduler& sch, const SystemData& sys_data,
                                      ScalapackInfo& scalapack_info, TAMMTensors& ttensors,
                                      EigenTensors& etensors);

template void compute_initial_guess<double>(ExecutionContext& ec, ScalapackInfo& scalapack_info,
                                            SystemData& sys_data, SCFVars& scf_vars,
                                            const std::vector<libint2::Atom>& atoms,
                                            const libint2::BasisSet&          shells,
                                            const std::string& basis, bool is_spherical,
                                            EigenTensors& etensors, TAMMTensors& ttensors,
                                            int charge, int multiplicity);

template std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
compute_initial_guess_taskinfo<double>(ExecutionContext& ec, SystemData& sys_data,
                                       const SCFVars&                    scf_vars,
                                       const std::vector<libint2::Atom>& atoms,
                                       const libint2::BasisSet& shells, const std::string& basis,
                                       bool is_spherical, EigenTensors& etensors,
                                       TAMMTensors& ttensors, int charge, int multiplicity);
