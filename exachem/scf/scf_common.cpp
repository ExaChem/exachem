/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "scf_common.hpp"

BasisSetMap construct_basisset_maps(std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                                    bool is_spherical) {
  BasisSetMap bsm;

  auto   a2s_map = shells.atom2shell(atoms);
  size_t natoms  = atoms.size();
  size_t nshells = shells.size();
  auto   nbf     = shells.nbf();

  std::vector<long> shell2atom_map = shells.shell2atom(atoms);
  auto              bf2shell       = map_basis_function_to_shell(shells);
  auto              shell2bf       = map_shell_to_basis_function(shells);

  std::vector<AtomInfo>         atominfo(natoms);
  std::vector<size_t>           bf2atom(nbf);
  std::vector<size_t>           nbf_atom(natoms);
  std::vector<size_t>           nshells_atom(natoms);
  std::vector<size_t>           first_bf_atom(natoms);
  std::vector<size_t>           first_bf_shell(nshells);
  std::vector<size_t>           first_shell_atom(natoms);
  std::map<size_t, std::string> bf_comp;

  for(size_t s1 = 0; s1 != nshells; ++s1) first_bf_shell[s1] = shells[s1].size();

  std::map<int, std::string>              gaus_comp_map{{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"},
                                           {4, "g"}, {5, "h"}, {6, "i"}};
  std::map<int, std::vector<std::string>> cart_comp_map{
    {1, {"x", "y", "z"}},
    {2, {"xx", "xy", "xz", "yy", "yz", "zz"}},
    {3, {"xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"}},
    {4,
     {"xxxx", "xxxy", "xxxz", "xxyy", "xxyz", "xxzz", "xyyy", "xyyz", "xyzz", "xzzz", "yyyy",
      "yyyz", "yyzz", "yzzz", "zzzz"}},
    {5, {"xxxxx", "xxxxy", "xxxxz", "xxxyy", "xxxyz", "xxxzz", "xxyyy",
         "xxyyz", "xxyzz", "xxzzz", "xyyyy", "xyyyz", "xyyzz", "xyzzz",
         "xzzzz", "yyyyy", "yyyyz", "yyyzz", "yyzzz", "yzzzz", "zzzzz"}},
    {6, {"xxxxxx", "xxxxxy", "xxxxxz", "xxxxyy", "xxxxyz", "xxxxzz", "xxxyyy",
         "xxxyyz", "xxxyzz", "xxxzzz", "xxyyyy", "xxyyyz", "xxyyzz", "xxyzzz",
         "xxzzzz", "xyyyyy", "xyyyyz", "xyyyzz", "xyyzzz", "xyzzzz", "xzzzzz",
         "yyyyyy", "yyyyyz", "yyyyzz", "yyyzzz", "yyzzzz", "yzzzzz", "zzzzzz"}}};

  for(size_t ai = 0; ai < natoms; ai++) {
    auto                        nshells_ai = a2s_map[ai].size();
    auto                        first      = a2s_map[ai][0];
    auto                        last       = a2s_map[ai][nshells_ai - 1];
    std::vector<libint2::Shell> atom_shells(nshells_ai);
    int                         as_index = 0;
    size_t                      atom_nbf = 0;
    first_shell_atom[ai]                 = first;
    for(auto si = first; si <= last; si++) {
      atom_shells[as_index] = shells[si];
      as_index++;
      atom_nbf += shells[si].size();
    }
    for(const auto& e: libint2::chemistry::get_element_info()) {
      if(e.Z == atoms[ai].atomic_number) {
        atominfo[ai].symbol = e.symbol;
        break;
      }
    }
    atominfo[ai].atomic_number = atoms[ai].atomic_number;
    atominfo[ai].shells        = atom_shells;
    atominfo[ai].nbf           = atom_nbf;
    atominfo[ai].nbf_lo        = 0;
    atominfo[ai].nbf_hi        = atom_nbf;
    if(ai > 0) {
      atominfo[ai].nbf_lo = atominfo[ai - 1].nbf_hi;
      atominfo[ai].nbf_hi = atominfo[ai].nbf_lo + atom_nbf;
    }

    nbf_atom[ai]      = atom_nbf;
    nshells_atom[ai]  = nshells_ai;
    first_bf_atom[ai] = atominfo[ai].nbf_lo;
    for(auto nlo = atominfo[ai].nbf_lo; nlo < atominfo[ai].nbf_hi; nlo++) bf2atom[nlo] = ai;

    int alo = atominfo[ai].nbf_lo;
    for(auto s: atominfo[ai].shells) {
      auto l = s.contr[0].l;
      if(is_spherical) {
        for(int i = 0; i < 2 * l + 1; i++) {
          std::stringstream tmps;
          if(l == 0) tmps << "";
          else if(l == 1) {
            if(i == 0) tmps << "_y";
            else if(i == 1) tmps << "_z";
            else if(i == 2) tmps << "_x";
          }
          else if(l <= 6) tmps << std::showpos << i - l;
          else NOT_IMPLEMENTED();
          bf_comp[alo] = gaus_comp_map[l] + tmps.str();
          alo++;
        }
      }
      else { // cartesian
        const auto ncfuncs  = ((l + 1) * (l + 2)) / 2;
        auto       cart_vec = cart_comp_map[l];
        for(int i = 0; i < ncfuncs; i++) {
          std::string tmps;
          if(l == 0) tmps = "";
          else if(l <= 6) tmps = "_" + cart_vec[i];
          else NOT_IMPLEMENTED();
          bf_comp[alo] = gaus_comp_map[l] + tmps;
          alo++;
        }
      }
    }
  }

  bsm.nbf              = nbf;
  bsm.natoms           = natoms;
  bsm.nshells          = nshells;
  bsm.atominfo         = atominfo;
  bsm.bf2shell         = bf2shell;
  bsm.shell2bf         = shell2bf;
  bsm.bf2atom          = bf2atom;
  bsm.nbf_atom         = nbf_atom;
  bsm.atom2shell       = a2s_map;
  bsm.nshells_atom     = nshells_atom;
  bsm.first_bf_atom    = first_bf_atom;
  bsm.first_bf_shell   = first_bf_shell;
  bsm.shell2atom       = shell2atom_map;
  bsm.first_shell_atom = first_shell_atom;
  bsm.bf_comp          = bf_comp;

  return bsm;
}

template<typename T>
Matrix read_scf_mat(std::string matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  auto mfile_id = H5Fopen(matfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int64_t> rdims(2);
  auto                 attr_dataset = H5Dopen(mfile_id, "rdims", H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());

  Matrix mat         = Matrix::Zero(rdims[0], rdims[1]);
  auto   mdataset_id = H5Dopen(mfile_id, mname.c_str(), H5P_DEFAULT);

  /* Read the datasets. */
  H5Dread(mdataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mat;
}

template<typename T>
void write_scf_mat(Matrix& C, std::string matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  const auto  N      = C.rows();
  const auto  Northo = C.cols();
  TensorType* buf    = C.data();

  /* Create a file. */
  hid_t file_id = H5Fcreate(matfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize        = N * Northo;
  hid_t   dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, mname.c_str(), get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - dims */
  std::vector<int64_t> rdims{N, Northo};
  hsize_t              attr_size      = rdims.size();
  auto                 attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT64, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v) {
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&v](size_t x, size_t y) { return v[x] < v[y]; });

  return idx;
}

template<typename T, int ndim>
void t2e_hf_helper(const ExecutionContext& ec, tamm::Tensor<T>& ttensor, Matrix& etensor,
                   const std::string& ustr) {
  const string pstr = "(" + ustr + ")";

  const auto rank = ec.pg().rank();
  const auto N    = etensor.rows(); // TODO

  if(rank == 0) tamm_to_eigen_tensor(ttensor, etensor);
  ec.pg().barrier();
  std::vector<T> Hbufv(N * N);
  T*             Hbuf            = &Hbufv[0]; // Hbufv.data();
  Eigen::Map<Matrix>(Hbuf, N, N) = etensor;
  // GA_Brdcst(Hbuf,N*N*sizeof(T),0);
  ec.pg().broadcast(Hbuf, N * N, 0);
  etensor = Eigen::Map<Matrix>(Hbuf, N, N);
  Hbufv.clear();
  Hbufv.shrink_to_fit();
}

void compute_shellpair_list(const ExecutionContext& ec, const libint2::BasisSet& shells,
                            SCFVars& scf_vars) {
  auto rank = ec.pg().rank();

  // compute OBS non-negligible shell-pair list
  std::tie(scf_vars.obs_shellpair_list, scf_vars.obs_shellpair_data) = compute_shellpairs(shells);
  size_t nsp                                                         = 0;
  for(auto& sp: scf_vars.obs_shellpair_list) { nsp += sp.second.size(); }
  if(rank == 0)
    std::cout << "# of {all,non-negligible} shell-pairs = {"
              << shells.size() * (shells.size() + 1) / 2 << "," << nsp << "}" << endl;
}

std::tuple<int, double> compute_NRE(const ExecutionContext& ec, std::vector<libint2::Atom>& atoms) {
  auto rank = ec.pg().rank();
  //  std::cout << "Geometries in bohr units " << std::endl;
  //  for (auto i = 0; i < atoms.size(); ++i)
  //    std::cout << atoms[i].atomic_number << "  " << atoms[i].x<< "  " <<
  //    atoms[i].y<< "  " << atoms[i].z << endl;
  // count the number of electrons
  auto nelectron = 0;
  for(size_t i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;

  // compute the nuclear repulsion energy
  double enuc = 0.0;
  for(size_t i = 0; i < atoms.size(); i++)
    for(size_t j = i + 1; j < atoms.size(); j++) {
      double xij = atoms[i].x - atoms[j].x;
      double yij = atoms[i].y - atoms[j].y;
      double zij = atoms[i].z - atoms[j].z;
      double r2  = xij * xij + yij * yij + zij * zij;
      double r   = sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }

  return std::make_tuple(nelectron, enuc);
}

std::tuple<std::vector<size_t>, std::vector<Tile>, std::vector<Tile>>
compute_AO_tiles(const ExecutionContext& ec, const SystemData& sys_data, libint2::BasisSet& shells,
                 const bool is_df) {
  auto rank      = ec.pg().rank();
  int  tile_size = sys_data.options_map.scf_options.AO_tilesize;
  if(is_df) tile_size = sys_data.options_map.scf_options.dfAO_tilesize;

  std::vector<Tile> AO_tiles;
  for(auto s: shells) AO_tiles.push_back(s.size());
  if(rank == 0) cout << "Number of AO tiles = " << AO_tiles.size() << endl;

  tamm::Tile          est_ts = 0;
  std::vector<Tile>   AO_opttiles;
  std::vector<size_t> shell_tile_map;
  for(auto s = 0U; s < shells.size(); s++) {
    est_ts += shells[s].size();
    if(est_ts >= tile_size) {
      AO_opttiles.push_back(est_ts);
      shell_tile_map.push_back(s); // shell id specifying tile boundary
      est_ts = 0;
    }
  }
  if(est_ts > 0) {
    AO_opttiles.push_back(est_ts);
    shell_tile_map.push_back(shells.size() - 1);
  }

  // std::vector<int> vtc(AO_tiles.size());
  // std::iota (std::begin(vtc), std::end(vtc), 0);
  // cout << "AO tile indexes = " << vtc;
  // cout << "orig AO tiles = " << AO_tiles;

  // cout << "print new opt AO tiles = " << AO_opttiles;
  // cout << "print shell-tile map = " << shell_tile_map;

  return std::make_tuple(shell_tile_map, AO_tiles, AO_opttiles);
}

// returns {X,X^{-1},S_condition_number_after_conditioning}, where
// X is the generalized square-root-inverse such that X.transpose() * S * X = I
// columns of Xinv is the basis conditioned such that
// the condition number of its metric (Xinv.transpose . Xinv) <
// S_condition_number_threshold
void compute_orthogonalizer(ExecutionContext& ec, SystemData& sys_data, SCFVars& scf_vars,
                            ScalapackInfo& scalapack_info, TAMMTensors& ttensors) {
  auto hf_t1 = std::chrono::high_resolution_clock::now();
  auto rank  = ec.pg().rank();

  // compute orthogonalizer X such that X.transpose() . S . X = I
  double XtX_condition_number; // condition number of "re-conditioned"
                               // overlap obtained as Xinv.transpose() . Xinv
  // by default assume can manage to compute with condition number of S <= 1/eps
  // this is probably too optimistic, but in well-behaved cases even 10^11 is OK
  size_t obs_rank;
  double S_condition_number;
  double S_condition_number_threshold = sys_data.options_map.scf_options.tol_lindep;

  std::tie(obs_rank, S_condition_number, XtX_condition_number) = gensqrtinv(
    ec, sys_data, scf_vars, scalapack_info, ttensors, false, S_condition_number_threshold);

  // TODO: Redeclare TAMM S1 with new dims?
  auto hf_t2   = std::chrono::high_resolution_clock::now();
  auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2)
              << "Time for computing orthogonalizer: " << hf_time << " secs" << endl
              << endl;
}

template<typename TensorType>
void compute_hamiltonian(ExecutionContext& ec, const SCFVars& scf_vars,
                         std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                         TAMMTensors& ttensors, EigenTensors& etensors) {
  using libint2::Operator;
  // const size_t N = shells.nbf();
  auto rank = ec.pg().rank();

  ttensors.H1 = {scf_vars.tAO, scf_vars.tAO};
  ttensors.S1 = {scf_vars.tAO, scf_vars.tAO};
  ttensors.T1 = {scf_vars.tAO, scf_vars.tAO};
  ttensors.V1 = {scf_vars.tAO, scf_vars.tAO};
  Tensor<TensorType>::allocate(&ec, ttensors.H1, ttensors.S1, ttensors.T1, ttensors.V1);

  auto [mu, nu] = scf_vars.tAO.labels<2>("all");

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  compute_1body_ints(ec, scf_vars, ttensors.S1, atoms, shells, Operator::overlap);
  compute_1body_ints(ec, scf_vars, ttensors.T1, atoms, shells, Operator::kinetic);
  compute_1body_ints(ec, scf_vars, ttensors.V1, atoms, shells, Operator::nuclear);
  auto hf_t2   = std::chrono::high_resolution_clock::now();
  auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "Time for computing 1-e integrals T, V, S: " << hf_time << " secs" << endl;

  // Core Hamiltonian = T + V
  // clang-format off
  Scheduler{ec}
    (ttensors.H1(mu, nu)  =  ttensors.T1(mu, nu))
    (ttensors.H1(mu, nu) +=  ttensors.V1(mu, nu)).execute();
  // clang-format on

  // tamm::scale_ip(ttensors.H1(),2.0);
}

void scf_restart_test(const ExecutionContext& ec, const SystemData& sys_data,
                      const std::string& filename, bool restart, std::string files_prefix) {
  if(!restart) return;
  const auto rank   = ec.pg().rank();
  const bool is_uhf = (sys_data.is_unrestricted);

  int rstatus = 1;

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";
  std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  std::string densityfile_beta  = files_prefix + ".beta.density";
  bool        status            = false;

  if(rank == 0) {
    status = fs::exists(movecsfile_alpha) && fs::exists(densityfile_alpha);
    if(is_uhf) status = status && fs::exists(movecsfile_beta) && fs::exists(densityfile_beta);
  }
  rstatus = status;
  ec.pg().barrier();
  ec.pg().broadcast(&rstatus, 0);
  std::string fnf = movecsfile_alpha + "; " + densityfile_alpha;
  if(is_uhf) fnf = fnf + "; " + movecsfile_beta + "; " + densityfile_beta;
  if(rstatus == 0) tamm_terminate("Error reading one or all of the files: [" + fnf + "]");
}

void scf_restart(const ExecutionContext& ec, const SystemData& sys_data,
                 const std::string& filename, EigenTensors& etensors, std::string files_prefix) {
  const auto rank   = ec.pg().rank();
  const auto N      = sys_data.nbf_orig;
  const auto Northo = N - sys_data.n_lindep;
  const bool is_uhf = sys_data.is_unrestricted;

  EXPECTS(Northo == sys_data.nbf);

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";

  if(rank == 0) {
    cout << "Reading movecs and density files ... ";
    etensors.C = read_scf_mat<TensorType>(movecsfile_alpha);
    etensors.D = read_scf_mat<TensorType>(densityfile_alpha);

    if(is_uhf) {
      std::string movecsfile_beta  = files_prefix + ".beta.movecs";
      std::string densityfile_beta = files_prefix + ".beta.density";
      etensors.C_beta              = read_scf_mat<TensorType>(movecsfile_beta);
      etensors.D_beta              = read_scf_mat<TensorType>(densityfile_beta);
    }
    cout << "done" << endl;
  }
  ec.pg().barrier();
}

template<typename TensorType>
double tt_trace(ExecutionContext& ec, Tensor<TensorType>& T1, Tensor<TensorType>& T2) {
  Tensor<TensorType> tensor{T1.tiled_index_spaces()}; //{tAO, tAO};
  Tensor<TensorType>::allocate(&ec, tensor);
  const TiledIndexSpace tis_ao = T1.tiled_index_spaces()[0];
  auto [mu, nu, ku]            = tis_ao.labels<3>("all");
  Scheduler{ec}(tensor(mu, nu) = T1(mu, ku) * T2(ku, nu)).execute();
  double trace = tamm::trace(tensor);
  Tensor<TensorType>::deallocate(tensor);
  return trace;
}

void print_energies(ExecutionContext& ec, TAMMTensors& ttensors, EigenTensors& etensors,
                    const SystemData& sys_data, SCFVars& scf_vars, bool debug) {
  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  const bool is_ks  = sys_data.is_ks;

  double nelectrons = 0.0;
  double kinetic_1e = 0.0;
  double NE_1e      = 0.0;
  double energy_1e  = 0.0;
  double energy_2e  = 0.0;

  if(is_rhf) {
    nelectrons = tt_trace(ec, ttensors.D_tamm, ttensors.S1);
    kinetic_1e = tt_trace(ec, ttensors.D_tamm, ttensors.T1);
    NE_1e      = tt_trace(ec, ttensors.D_tamm, ttensors.V1);
    energy_1e  = tt_trace(ec, ttensors.D_tamm, ttensors.H1);
    energy_2e  = 0.5 * tt_trace(ec, ttensors.D_tamm, ttensors.F_alpha_tmp);

    if(is_ks) { energy_2e += scf_vars.exc; }
  }
  if(is_uhf) {
    nelectrons = tt_trace(ec, ttensors.D_tamm, ttensors.S1);
    kinetic_1e = tt_trace(ec, ttensors.D_tamm, ttensors.T1);
    NE_1e      = tt_trace(ec, ttensors.D_tamm, ttensors.V1);
    energy_1e  = tt_trace(ec, ttensors.D_tamm, ttensors.H1);
    energy_2e  = 0.5 * tt_trace(ec, ttensors.D_tamm, ttensors.F_alpha_tmp);
    nelectrons += tt_trace(ec, ttensors.D_beta_tamm, ttensors.S1);
    kinetic_1e += tt_trace(ec, ttensors.D_beta_tamm, ttensors.T1);
    NE_1e += tt_trace(ec, ttensors.D_beta_tamm, ttensors.V1);
    energy_1e += tt_trace(ec, ttensors.D_beta_tamm, ttensors.H1);
    energy_2e += 0.5 * tt_trace(ec, ttensors.D_beta_tamm, ttensors.F_beta_tmp);
    if(is_ks) { energy_2e += scf_vars.exc; }
  }

  if(ec.pg().rank() == 0) {
    std::cout << "#electrons        = " << nelectrons << endl;
    std::cout << "1e energy kinetic = " << std::setprecision(16) << kinetic_1e << endl;
    std::cout << "1e energy N-e     = " << NE_1e << endl;
    std::cout << "1e energy         = " << energy_1e << endl;
    std::cout << "2e energy         = " << energy_2e << std::endl;
  }
}

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
std::tuple<size_t, double, double> gensqrtinv(ExecutionContext& ec, SystemData& sys_data,
                                              SCFVars& scf_vars, ScalapackInfo& scalapack_info,
                                              TAMMTensors& ttensors, bool symmetric,
                                              double threshold) {
  using T = TensorType;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  int world_rank = ec.pg().rank().value();
  int world_size = ec.pg().size().value();

  int64_t       n_cond{}, n_illcond{};
  double        condition_number{}, result_condition_number{};
  const int64_t N = sys_data.nbf_orig;

  // TODO: avoid eigen matrices
  Matrix         X, V;
  std::vector<T> eps(N);

#if defined(USE_SCALAPACK)
  Tensor<T> V_sca;
  if(scalapack_info.comm != MPI_COMM_NULL) {
    blacspp::Grid*                  blacs_grid       = scalapack_info.blacs_grid.get();
    const auto&                     grid             = *blacs_grid;
    scalapackpp::BlockCyclicDist2D* blockcyclic_dist = scalapack_info.blockcyclic_dist.get();
    const tamm::Tile                mb               = blockcyclic_dist->mb();

    TiledIndexSpace tN_bc{IndexSpace{range(sys_data.nbf_orig)}, mb};
    Tensor<T>       S_BC{tN_bc, tN_bc};
    V_sca = {tN_bc, tN_bc};
    S_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    V_sca.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    Tensor<T>::allocate(&scalapack_info.ec, S_BC, V_sca);

    tamm::to_block_cyclic_tensor(ttensors.S1, S_BC);

    auto desc_lambda = [&](const int64_t M, const int64_t N) {
      auto [M_loc, N_loc] = (*blockcyclic_dist).get_local_dims(M, N);
      return (*blockcyclic_dist).descinit_noerror(M, N, M_loc);
    };

    if(grid.ipr() >= 0 and grid.ipc() >= 0) {
      auto desc_S = desc_lambda(N, N);
      auto desc_V = desc_lambda(N, N);
      // scalapackpp::BlockCyclicMatrix<T> V_sca(grid, N, N, mb, mb);

      auto info = scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, desc_S[2],
                                      S_BC.access_local_buf(), 1, 1, desc_S, eps.data(),
                                      V_sca.access_local_buf(), 1, 1, desc_V);

      // Gather results
      // if( scalapack_info.pg.rank() == 0 ) V.resize(N,N);
      // V_sca.gather_from( N, N, V.data(), N, 0, 0 );
    }

    Tensor<T>::deallocate(S_BC);
  }

#else

  if(world_rank == 0) {
    // Eigen decompose S -> VsV**T
    V.resize(N, N);
    tamm_to_eigen_tensor(ttensors.S1, V);
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, V.data(), N, eps.data());
  }

#endif

  std::vector<T>::iterator first_above_thresh;
  if(world_rank == 0) {
    // condition_number = std::min(
    //   eps.back() / std::max( eps.front(), std::numeric_limits<double>::min() ),
    //   1.       / std::numeric_limits<double>::epsilon()
    // );

    // const auto threshold = eps.back() / max_condition_number;
    first_above_thresh =
      std::find_if(eps.begin(), eps.end(), [&](const auto& x) { return x >= threshold; });
    result_condition_number = eps.back() / *first_above_thresh;

    n_illcond = std::distance(eps.begin(), first_above_thresh);
    n_cond    = N - n_illcond;

    if(n_illcond > 0) {
      std::cout << std::endl
                << "WARNING: Found " << n_illcond << " linear dependencies" << std::endl;
      cout << std::defaultfloat << "First eigen value above tol_lindep = " << *first_above_thresh
           << endl;
      std::cout << "The overlap matrix has " << n_illcond
                << " vectors deemed linearly dependent with eigenvalues:" << std::endl;

      for(int64_t i = 0; i < n_illcond; i++)
        cout << std::defaultfloat << i + 1 << ": " << eps[i] << endl;
    }
  }

  if(world_size > 1) {
    // TODO: Should buffer this
    ec.pg().broadcast(&n_cond, 0);
    ec.pg().broadcast(&n_illcond, 0);
    // ec.pg().broadcast( &condition_number,        0 );
    ec.pg().broadcast(&result_condition_number, 0);
  }

#if defined(USE_SCALAPACK)
  if(scalapack_info.comm != MPI_COMM_NULL) {
    Tensor<T> V_t = from_block_cyclic_tensor(V_sca);
    Tensor<T> X_t = tensor_block(V_t, {n_illcond, 0}, {N, N}, {1, 0});
    if(scalapack_info.pg.rank() == 0) X = tamm_to_eigen_matrix<T>(X_t);
    Tensor<T>::deallocate(V_sca, V_t, X_t);
  }
#else
  if(world_rank == 0) {
    // auto* V_cond = Vbuf + n_illcond * N;
    Matrix V_cond = V.block(n_illcond, 0, N - n_illcond, N);
    V.resize(0, 0);
    X.resize(N, n_cond); // Xinv.resize( N, n_cond );
    // Matrix V_cond(n_cond,N);
    // V_cond = Eigen::Map<Matrix>(Vbuf + n_illcond * N,n_cond,N);
    X = V_cond.transpose();
    V_cond.resize(0, 0);
  }
#endif

  if(world_rank == 0) {
    // Form canonical X/Xinv
    for(auto i = 0; i < n_cond; ++i) {
      const double srt = std::sqrt(*(first_above_thresh + i));

      // X is row major...
      auto* X_col = X.data() + i;
      // auto* Xinv_col = Xinv.data() + i;

      blas::scal(N, 1. / srt, X_col, n_cond);
      // blas::scal( N, srt, Xinv_col, n_cond );
    }

    if(symmetric) {
      assert(not symmetric);
      /*
      // X is row major, thus we need to form X**T = V_cond * X**T
      Matrix TMP = X;
      X.resize( N, N );
      blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans, N, N, n_cond, 1., V_cond, N, TMP.data(),
      n_cond, 0., X.data(), N );
      */
    }
  } // compute on root

  sys_data.n_lindep = n_illcond;
  sys_data.nbf      = sys_data.nbf_orig - sys_data.n_lindep;

  const bool is_uhf  = (sys_data.is_unrestricted);
  scf_vars.tAO_ortho = TiledIndexSpace{IndexSpace{range(0, (size_t) (sys_data.nbf))},
                                       sys_data.options_map.scf_options.AO_tilesize};

#if defined(USE_SCALAPACK)
  if(scalapack_info.comm != MPI_COMM_NULL) {
    const tamm::Tile _mb = (scalapack_info.blockcyclic_dist.get())->mb();
    scf_vars.tN_bc       = TiledIndexSpace{IndexSpace{range(sys_data.nbf_orig)}, _mb};
    scf_vars.tNortho_bc  = TiledIndexSpace{IndexSpace{range(sys_data.nbf)}, _mb};
    ttensors.X_alpha     = {scf_vars.tN_bc, scf_vars.tNortho_bc};
    ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_alpha);
    if(is_uhf) {
      ttensors.X_beta = {scf_vars.tN_bc, scf_vars.tNortho_bc};
      ttensors.X_beta.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_beta);
    }
  }
#else
  ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
  sch.allocate(ttensors.X_alpha).execute();
  if(is_uhf) {
    ttensors.X_beta = {scf_vars.tAO, scf_vars.tAO_ortho};
    sch.allocate(ttensors.X_beta).execute();
  }
#endif

  if(world_rank == 0) eigen_to_tamm_tensor(ttensors.X_alpha, X);
  if(is_uhf)
    if(world_rank == 0) eigen_to_tamm_tensor(ttensors.X_beta, X); // X_beta=X_alpha here

  ec.pg().barrier();

  return std::make_tuple(size_t(n_cond), condition_number, result_condition_number);
}

std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(const libint2::BasisSet& bs1,
                                                                  const libint2::BasisSet& _bs2,
                                                                  const double threshold) {
  using libint2::BasisSet;
  using libint2::BraKet;
  using libint2::Engine;
  using libint2::Operator;

  const BasisSet& bs2           = (_bs2.empty() ? bs1 : _bs2);
  const auto      nsh1          = bs1.size();
  const auto      nsh2          = bs2.size();
  const auto      bs1_equiv_bs2 = (&bs1 == &bs2);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::overlap, std::max(bs1.max_nprim(), bs2.max_nprim()),
                std::max(bs1.max_l(), bs2.max_l()), 0);

  shellpair_list_t splist;

  const auto& buf = engine.results();

  // loop over permutationally-unique set of shells
  for(size_t s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
    // mx.lock();
    if(splist.find(s1) == splist.end()) splist.insert(std::make_pair(s1, std::vector<size_t>()));
    // mx.unlock();

    auto n1 = bs1[s1].size(); // number of basis functions in this shell

    auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
    for(decltype(s2_max) s2 = 0; s2 <= s2_max; ++s2, ++s12) {
      // if (s12 % nthreads != thread_id) continue;

      auto on_same_center = (bs1[s1].O == bs2[s2].O);
      bool significant    = on_same_center;
      if(not on_same_center) {
        auto n2 = bs2[s2].size();
        engine.compute(bs1[s1], bs2[s2]);
        Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
        auto                     norm = buf_mat.norm();
        significant                   = (norm >= threshold);
      }

      if(significant) {
        // mx.lock();
        splist[s1].emplace_back(s2);
        // mx.unlock();
      }
    }
  }

  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
  // N.B. only parallelized over 1 shell index
  for(size_t s1 = 0l; s1 != nsh1; ++s1) {
    // if (s1 % nthreads == thread_id) {
    auto& list = splist[s1];
    std::sort(list.begin(), list.end());
  }
  // }

  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto       ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());

  for(size_t s1 = 0l; s1 != nsh1; ++s1) {
    // if (s1 % nthreads == thread_id) {
    for(const auto& s2: splist[s1]) {
      spdata[s1].emplace_back(
        std::make_shared<libint2::ShellPair>(bs1[s1], bs2[s2], ln_max_engine_precision));
    }
    // }
  }

  return std::make_tuple(splist, spdata);
}

template<libint2::Operator Kernel>
Matrix compute_schwarz_ints(ExecutionContext& ec, const SCFVars& scf_vars,
                            const libint2::BasisSet& bs1, const libint2::BasisSet& _bs2,
                            bool                                                        use_2norm,
                            typename libint2::operator_traits<Kernel>::oper_params_type params) {
  using libint2::BasisSet;
  using libint2::BraKet;
  using libint2::Engine;

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  const BasisSet& bs2           = (_bs2.empty() ? bs1 : _bs2);
  const auto      nsh1          = bs1.size();
  const auto      nsh2          = bs2.size();
  const auto      bs1_equiv_bs2 = (&bs1 == &bs2);

  EXPECTS(nsh1 == nsh2);
  Matrix K = Matrix::Zero(nsh1, nsh2);

  // construct the 2-electron repulsion integrals engine
  // !!! very important: cannot screen primitives in Schwarz computation !!!
  double epsilon = 0.0;
  Engine engine  = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                          std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);

  auto& buf = engine.results();

  const std::vector<Tile>&   AO_tiles       = scf_vars.AO_tiles;
  const std::vector<size_t>& shell_tile_map = scf_vars.shell_tile_map;

  TiledIndexSpace    tnsh{IndexSpace{range(0, nsh1)}, static_cast<Tile>(std::ceil(nsh1 * 0.05))};
  Tensor<TensorType> schwarz{scf_vars.tAO, scf_vars.tAO};
  Tensor<TensorType> schwarz_mat{tnsh, tnsh};
  Tensor<TensorType>::allocate(&ec, schwarz_mat);

  auto compute_schwarz_matrix = [&](const IndexVector& blockid) {
    auto bi0 = blockid[0];
    auto bi1 = blockid[1];

    // loop over permutationally-unique set of shells
    auto                  s1range_end   = shell_tile_map[bi0];
    decltype(s1range_end) s1range_start = 0l;
    if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

    for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
      auto n1 = bs1[s1].size();

      auto                  s2range_end   = shell_tile_map[bi1];
      decltype(s2range_end) s2range_start = 0l;
      if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

      for(auto s2 = s2range_start; s2 <= s2range_end; ++s2) {
        auto n2  = bs2[s2].size();
        auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        engine.compute2<Kernel, BraKet::xx_xx, 0>(bs1[s1], bs2[s2], bs1[s1], bs2[s2]);

        EXPECTS(buf[0] != nullptr && "turn off primitive screening to compute Schwarz ints");

        // to apply Schwarz inequality to individual integrals must use the diagonal elements
        // to apply it to sets of functions (e.g. shells) use the whole shell-set of ints here
        Eigen::Map<const Matrix> buf_mat(buf[0], n12, n12);
        auto norm2 = use_2norm ? buf_mat.norm() : buf_mat.lpNorm<Eigen::Infinity>();
        K(s1, s2)  = std::sqrt(norm2);
        if(bs1_equiv_bs2) K(s2, s1) = K(s1, s2);
      }
    }
  };

  block_for(ec, schwarz(), compute_schwarz_matrix);
  ec.pg().barrier();

  eigen_to_tamm_tensor_acc(schwarz_mat, K);
  ec.pg().barrier();
  K.resize(0, 0);

  K = tamm_to_eigen_matrix<TensorType>(schwarz_mat);
  Tensor<TensorType>::deallocate(schwarz_mat);

  auto hf_t2   = std::chrono::high_resolution_clock::now();
  auto hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  // if(ec.pg().rank() == 0)
  //   std::cout << std::fixed << std::setprecision(2) << "Time to compute schwarz matrix: " <<
  //   hf_time
  //             << " secs" << endl;

  return K;
}

Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A) {
  const auto nsh = obs.size();
  Matrix     Ash = Matrix::Zero(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for(size_t s1 = 0; s1 != nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size  = obs[s1].size();
    for(size_t s2 = 0; s2 != nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size  = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size).lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}

void print_mulliken(OptionsMap& options_map, libint2::BasisSet& shells, Matrix& D, Matrix& D_beta,
                    Matrix& S, bool is_uhf) {
  auto        ec_atoms = options_map.options.ec_atoms;
  BasisSetMap bsm      = construct_basisset_maps(options_map.options.atoms, shells);

  const int                        natoms = ec_atoms.size();
  std::vector<double>              cs_acharge(natoms, 0);
  std::vector<double>              os_acharge(natoms, 0);
  std::vector<double>              net_acharge(natoms, 0);
  std::vector<std::vector<double>> cs_charge_shell(natoms);
  std::vector<std::vector<double>> os_charge_shell(natoms);
  std::vector<std::vector<double>> net_charge_shell(natoms);

  int j = 0;
  for(size_t x = 0; x < natoms; x++) { // loop over atoms
    auto atom_shells = bsm.atominfo[x].shells;
    auto nshells     = atom_shells.size(); // #shells for atom x
    cs_charge_shell[x].resize(nshells);
    if(is_uhf) os_charge_shell[x].resize(nshells);
    for(auto s = 0; s < nshells; s++) { // loop over each shell for atom x
      double cs_scharge = 0.0, os_scharge = 0.0;
      for(auto si = 0; si < atom_shells[s].size(); si++) {
        for(size_t i = 0; i < S.rows(); i++) {
          const auto ds_cs = D(j, i) * S(j, i);
          cs_scharge += ds_cs;
          cs_acharge[x] += ds_cs;
          if(is_uhf) {
            const auto ds_os = D_beta(j, i) * S(j, i);
            os_scharge += ds_os;
            os_acharge[x] += ds_os;
          }
        }
        j++;
      }
      cs_charge_shell[x][s] = cs_scharge;
      if(is_uhf) os_charge_shell[x][s] = os_scharge;
    }
  }

  net_acharge      = cs_acharge;
  net_charge_shell = cs_charge_shell;
  if(is_uhf) {
    for(size_t x = 0; x < natoms; x++) { // loop over atoms
      net_charge_shell[x].resize(cs_charge_shell[x].size());
      net_acharge[x] = cs_acharge[x] + os_acharge[x];
      std::transform(cs_charge_shell[x].begin(), cs_charge_shell[x].end(),
                     os_charge_shell[x].begin(), net_charge_shell[x].begin(), std::plus<double>());
    }
  }
  const auto mksp = std::string(5, ' ');

  auto print_ma = [&](const std::string dtype, std::vector<double>& acharge,
                      std::vector<std::vector<double>>& charge_shell) {
    std::cout << std::endl
              << mksp << "Mulliken analysis of the " << dtype << " density" << std::endl;
    std::cout << mksp << std::string(50, '-') << std::endl << std::endl;
    std::cout << mksp << "   Atom   " << mksp << " Charge " << mksp << "  Shell Charges  "
              << std::endl;
    std::cout << mksp << "----------" << mksp << "--------" << mksp << std::string(50, '-')
              << std::endl;

    for(size_t x = 0; x < natoms; x++) { // loop over atoms
      const auto Z        = ec_atoms[x].atom.atomic_number;
      const auto e_symbol = ec_atoms[x].esymbol;
      std::cout << mksp << std::setw(3) << std::right << x + 1 << " " << std::left << std::setw(2)
                << e_symbol << " " << std::setw(3) << std::right << Z << mksp << std::fixed
                << std::setprecision(2) << std::right << " " << std::setw(5) << acharge[x] << mksp
                << "  " << std::right;
      for(auto csx: charge_shell[x]) std::cout << std::setw(5) << csx << " ";
      std::cout << std::endl;
    }
  };
  print_ma("total", net_acharge, net_charge_shell);
  if(is_uhf) {
    print_ma("alpha", cs_acharge, cs_charge_shell);
    print_ma("beta", os_acharge, os_charge_shell);
  }
}

template<typename TensorType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                    std::vector<int>& ntask_vec) {
  const int rank   = ec.pg().rank().value();
  const int nranks = ec.pg().size().value();

#ifdef USE_UPCXX
  upcxx::global_ptr<int> s1_count = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int> s2_count = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int> nt_count = upcxx::new_array<int>(nranks);
  assert(s1_count && s2_count && nt_count);

  upcxx::dist_object<upcxx::global_ptr<int>> s1_count_dobj(s1_count, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> s2_count_dobj(s2_count, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> nt_count_dobj(nt_count, *ec.pg().team());
#else
  std::vector<int> s1_count(nranks);
  std::vector<int> s2_count(nranks);
  std::vector<int> nt_count(nranks);
#endif

  int s1vec_size = (int) s1vec.size();
  int s2vec_size = (int) s2vec.size();
  int ntvec_size = (int) ntask_vec.size();

  // Root gathers number of elements at each rank.
#ifdef USE_UPCXX
  ec.pg().gather(&s1vec_size, s1_count_dobj.fetch(0).wait());
  ec.pg().gather(&s2vec_size, s2_count_dobj.fetch(0).wait());
  ec.pg().gather(&ntvec_size, nt_count_dobj.fetch(0).wait());
#else
  ec.pg().gather(&s1vec_size, s1_count.data(), 0);
  ec.pg().gather(&s2vec_size, s2_count.data(), 0);
  ec.pg().gather(&ntvec_size, nt_count.data(), 0);
#endif

  // Displacements in the receive buffer for GATHERV
#ifdef USE_UPCXX
  upcxx::global_ptr<int>                     disps_s1 = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int>                     disps_s2 = upcxx::new_array<int>(nranks);
  upcxx::global_ptr<int>                     disps_nt = upcxx::new_array<int>(nranks);
  upcxx::dist_object<upcxx::global_ptr<int>> disps_s1_dobj(disps_s1, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> disps_s2_dobj(disps_s2, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> disps_nt_dobj(disps_nt, *ec.pg().team());
#else
  std::vector<int> disps_s1(nranks);
  std::vector<int> disps_s2(nranks);
  std::vector<int> disps_nt(nranks);
#endif
  for(int i = 0; i < nranks; i++) {
#ifdef USE_UPCXX
    disps_s1.local()[i] = (i > 0) ? (disps_s1.local()[i - 1] + s1_count.local()[i - 1]) : 0;
    disps_s2.local()[i] = (i > 0) ? (disps_s2.local()[i - 1] + s2_count.local()[i - 1]) : 0;
    disps_nt.local()[i] = (i > 0) ? (disps_nt.local()[i - 1] + nt_count.local()[i - 1]) : 0;
#else
    disps_s1[i] = (i > 0) ? (disps_s1[i - 1] + s1_count[i - 1]) : 0;
    disps_s2[i] = (i > 0) ? (disps_s2[i - 1] + s2_count[i - 1]) : 0;
    disps_nt[i] = (i > 0) ? (disps_nt[i - 1] + nt_count[i - 1]) : 0;
#endif
  }

  // Allocate vectors to gather data at root
#ifdef USE_UPCXX
  upcxx::global_ptr<int> s1_all;
  upcxx::global_ptr<int> s2_all;
  upcxx::global_ptr<int> ntasks_all;
  std::vector<int>       s1_all_v;
  std::vector<int>       s2_all_v;
  std::vector<int>       ntasks_all_v;
#else
  std::vector<int> s1_all;
  std::vector<int> s2_all;
  std::vector<int> ntasks_all;
#endif
  if(rank == 0) {
#ifdef USE_UPCXX
    s1_all     = upcxx::new_array<int>(disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]);
    s2_all     = upcxx::new_array<int>(disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]);
    ntasks_all = upcxx::new_array<int>(disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]);

    s1_all_v.resize(disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]);
    s2_all_v.resize(disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]);
    ntasks_all_v.resize(disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]);
#else
    s1_all.resize(disps_s1[nranks - 1] + s1_count[nranks - 1]);
    s2_all.resize(disps_s2[nranks - 1] + s2_count[nranks - 1]);
    ntasks_all.resize(disps_nt[nranks - 1] + nt_count[nranks - 1]);
#endif
  }

#ifdef USE_UPCXX
  upcxx::dist_object<upcxx::global_ptr<int>> s1_all_dobj(s1_all, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> s2_all_dobj(s2_all, *ec.pg().team());
  upcxx::dist_object<upcxx::global_ptr<int>> ntasks_all_dobj(ntasks_all, *ec.pg().team());
#endif

  // Gather at root
#ifdef USE_UPCXX
  ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all_dobj.fetch(0).wait(), s1_count.local(),
                  disps_s1_dobj.fetch(0).wait());
  ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all_dobj.fetch(0).wait(), s2_count.local(),
                  disps_s2_dobj.fetch(0).wait());
  ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all_dobj.fetch(0).wait(), nt_count.local(),
                  disps_nt_dobj.fetch(0).wait());
#else
  ec.pg().gatherv(s1vec.data(), s1vec_size, s1_all.data(), s1_count.data(), disps_s1.data(), 0);
  ec.pg().gatherv(s2vec.data(), s2vec_size, s2_all.data(), s2_count.data(), disps_s2.data(), 0);
  ec.pg().gatherv(ntask_vec.data(), ntvec_size, ntasks_all.data(), nt_count.data(), disps_nt.data(),
                  0);
#endif

#ifdef USE_UPCXX
  if(rank == 0) {
    memcpy(s1_all_v.data(), s1_all.local(),
           (disps_s1.local()[nranks - 1] + s1_count.local()[nranks - 1]) * sizeof(int));
    memcpy(s2_all_v.data(), s2_all.local(),
           (disps_s2.local()[nranks - 1] + s2_count.local()[nranks - 1]) * sizeof(int));
    memcpy(ntasks_all_v.data(), ntasks_all.local(),
           (disps_nt.local()[nranks - 1] + nt_count.local()[nranks - 1]) * sizeof(int));
  }

  upcxx::delete_array(s1_count);
  upcxx::delete_array(s2_count);
  upcxx::delete_array(nt_count);
  upcxx::delete_array(disps_s1);
  upcxx::delete_array(disps_s2);
  upcxx::delete_array(disps_nt);
  if(rank == 0) {
    upcxx::delete_array(s1_all);
    upcxx::delete_array(s2_all);
    upcxx::delete_array(ntasks_all);
  }
  return std::make_tuple(s1_all_v, s2_all_v, ntasks_all_v);
#else
  EXPECTS(s1_all.size() == s2_all.size());
  EXPECTS(s1_all.size() == ntasks_all.size());
  return std::make_tuple(s1_all, s2_all, ntasks_all);
#endif
}

#if defined(USE_GAUXC)

GauXC::Molecule gauxc_util::make_gauxc_molecule(const std::vector<libint2::Atom>& atoms) {
  GauXC::Molecule mol;
  mol.resize(atoms.size());
  std::transform(atoms.begin(), atoms.end(), mol.begin(), [](const libint2::Atom& atom) {
    GauXC::Atom gauxc_atom(GauXC::AtomicNumber(atom.atomic_number), atom.x, atom.y, atom.z);
    return gauxc_atom;
  });
  return mol;
}

GauXC::BasisSet<double> gauxc_util::make_gauxc_basis(const libint2::BasisSet& basis) {
  using shell_t = GauXC::Shell<double>;
  using prim_t  = typename shell_t::prim_array;
  using cart_t  = typename shell_t::cart_array;

  GauXC::BasisSet<double> gauxc_basis;
  for(const auto& shell: basis) {
    prim_t prim_array, coeff_array;
    cart_t origin;

    std::copy(shell.alpha.begin(), shell.alpha.end(), prim_array.begin());
    std::copy(shell.contr[0].coeff.begin(), shell.contr[0].coeff.end(), coeff_array.begin());
    std::copy(shell.O.begin(), shell.O.end(), origin.begin());

    gauxc_basis.emplace_back(
      GauXC::PrimSize(shell.alpha.size()), GauXC::AngularMomentum(shell.contr[0].l),
      GauXC::SphericalType(shell.contr[0].pure), prim_array, coeff_array, origin, false);
  }
  // gauxc_basis.generate_shell_to_ao();
  return gauxc_basis;
}

template<typename TensorType>
TensorType gauxc_util::compute_xcf(ExecutionContext& ec, const SystemData& sys_data,
                                   TAMMTensors& ttensors, EigenTensors& etensors,
                                   GauXC::XCIntegrator<Matrix>& xc_integrator) {
  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  auto       rank0  = ec.pg().rank() == 0;

  double  EXC{};
  Matrix& vxc_alpha = etensors.G;
  Matrix& vxc_beta  = etensors.G_beta;

  if(is_rhf) {
    std::tie(EXC, vxc_alpha) = xc_integrator.eval_exc_vxc(0.5 * etensors.D);
    if(rank0) eigen_to_tamm_tensor(ttensors.VXC_alpha, vxc_alpha);
  }
  else if(is_uhf) {
    std::tie(EXC, vxc_alpha, vxc_beta) = xc_integrator.eval_exc_vxc(
      0.5 * (etensors.D + etensors.D_beta), 0.5 * (etensors.D - etensors.D_beta));
    if(rank0) {
      eigen_to_tamm_tensor(ttensors.VXC_alpha, vxc_alpha);
      eigen_to_tamm_tensor(ttensors.VXC_beta, vxc_beta);
    }
  }
  ec.pg().barrier();

  return EXC;
}

template double gauxc_util::compute_xcf<double>(ExecutionContext& ec, const SystemData& sys_data,
                                                TAMMTensors& ttensors, EigenTensors& etensors,
                                                GauXC::XCIntegrator<Matrix>& xc_integrator);

#endif

std::tuple<size_t, double, double>
gensqrtinv_atscf(ExecutionContext& ec, SystemData& sys_data, SCFVars& scf_vars,
                 ScalapackInfo& scalapack_info, Tensor<double> S1, Tensor<double>& X_alpha,
                 Tensor<double>& X_beta, TiledIndexSpace& tao_atom, bool symmetric,
                 double threshold) {
  using T = double;

  Scheduler sch{ec};
  // auto world = ec.pg().comm();
  int world_rank = ec.pg().rank().value();
  int world_size = ec.pg().size().value();

  int64_t       n_cond{}, n_illcond{};
  double        condition_number{}, result_condition_number{};
  const int64_t N = tao_atom.index_space().num_indices();

  // TODO: avoid eigen matrices
  Matrix         X, V;
  std::vector<T> eps(N);

  if(world_rank == 0) {
    // Eigen decompose S -> VsV**T
    V.resize(N, N);
    tamm_to_eigen_tensor(S1, V);
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, V.data(), N, eps.data());
  }

  std::vector<T>::iterator first_above_thresh;
  if(world_rank == 0) {
    // condition_number = std::min(
    //   eps.back() / std::max( eps.front(), std::numeric_limits<double>::min() ),
    //   1.       / std::numeric_limits<double>::epsilon()
    // );

    // const auto threshold = eps.back() / max_condition_number;
    first_above_thresh =
      std::find_if(eps.begin(), eps.end(), [&](const auto& x) { return x >= threshold; });
    result_condition_number = eps.back() / *first_above_thresh;

    n_illcond = std::distance(eps.begin(), first_above_thresh);
    n_cond    = N - n_illcond;

    if(n_illcond > 0) {
      std::cout << std::endl
                << "WARNING: Found " << n_illcond << " linear dependencies" << std::endl;
      cout << std::defaultfloat << "First eigen value above tol_lindep = " << *first_above_thresh
           << endl;
      std::cout << "The overlap matrix has " << n_illcond
                << " vectors deemed linearly dependent with eigenvalues:" << std::endl;

      for(int64_t i = 0; i < n_illcond; i++)
        cout << std::defaultfloat << i + 1 << ": " << eps[i] << endl;
    }
  }

  if(world_size > 1) {
    // TODO: Should buffer this
    ec.pg().broadcast(&n_cond, 0);
    ec.pg().broadcast(&n_illcond, 0);
    // ec.pg().broadcast( &condition_number,        0 );
    ec.pg().broadcast(&result_condition_number, 0);
  }

  if(world_rank == 0) {
    // auto* V_cond = Vbuf + n_illcond * N;
    Matrix V_cond = V.block(n_illcond, 0, N - n_illcond, N);
    V.resize(0, 0);
    X.resize(N, n_cond); // Xinv.resize( N, n_cond );
    // Matrix V_cond(n_cond,N);
    // V_cond = Eigen::Map<Matrix>(Vbuf + n_illcond * N,n_cond,N);
    X = V_cond.transpose();
    V_cond.resize(0, 0);
  }

  if(world_rank == 0) {
    // Form canonical X/Xinv
    for(auto i = 0; i < n_cond; ++i) {
      const double srt = std::sqrt(*(first_above_thresh + i));

      // X is row major...
      auto* X_col = X.data() + i;
      // auto* Xinv_col = Xinv.data() + i;

      blas::scal(N, 1. / srt, X_col, n_cond);
      // blas::scal( N, srt, Xinv_col, n_cond );
    }

    if(symmetric) {
      assert(not symmetric);
      /*
      // X is row major, thus we need to form X**T = V_cond * X**T
      Matrix TMP = X;
      X.resize( N, N );
      blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans, N, N, n_cond, 1., V_cond, N, TMP.data(),
      n_cond, 0., X.data(), N );
      */
    }
  } // compute on root

  auto nbf_new = N - n_illcond;

  const bool is_uhf    = (sys_data.is_unrestricted);
  auto       tAO_ortho = TiledIndexSpace{IndexSpace{range(0, (size_t) nbf_new)},
                                   sys_data.options_map.scf_options.AO_tilesize};

  X_alpha = {tao_atom, tAO_ortho};
  sch.allocate(X_alpha).execute();
  if(is_uhf) {
    X_beta = {tao_atom, tAO_ortho};
    sch.allocate(X_beta).execute();
  }

  if(world_rank == 0) eigen_to_tamm_tensor(X_alpha, X);
  if(is_uhf)
    if(world_rank == 0) eigen_to_tamm_tensor(X_beta, X); // X_beta=X_alpha here

  ec.pg().barrier();

  return std::make_tuple(size_t(n_cond), condition_number, result_condition_number);
}

template void write_scf_mat<double>(Matrix& C, std::string matfile);

template std::vector<size_t> sort_indexes<double>(std::vector<double>& v);

template Matrix compute_schwarz_ints<libint2::Operator::coulomb>(
  ExecutionContext& ec, const SCFVars& scf_vars, const libint2::BasisSet& bs1,
  const libint2::BasisSet& bs2, bool use_2norm,
  typename libint2::operator_traits<libint2::Operator::coulomb>::oper_params_type params);

template void t2e_hf_helper<double, 2>(const ExecutionContext& ec, tamm::Tensor<double>& ttensor,
                                       Matrix& etensor, const std::string&);

template void compute_hamiltonian<double>(ExecutionContext& ec, const SCFVars& scf_vars,
                                          std::vector<libint2::Atom>& atoms,
                                          libint2::BasisSet& shells, TAMMTensors& ttensors,
                                          EigenTensors& etensors);

template double tt_trace<double>(ExecutionContext& ec, Tensor<TensorType>& T1,
                                 Tensor<TensorType>& T2);

// template Matrix compute_schwarz_ints<libint2::Operator>(ExecutionContext& ec, const SCFVars&
// scf_vars,
//                             const libint2::BasisSet& bs1, const libint2::BasisSet& _bs2,
//                             bool use_2norm,
//                             typename
//                             libint2::operator_traits<libint2::Operator>::oper_params_type
//                             params);

template std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gather_task_vectors<double>(ExecutionContext& ec, std::vector<int>& s1vec, std::vector<int>& s2vec,
                            std::vector<int>& ntask_vec);
