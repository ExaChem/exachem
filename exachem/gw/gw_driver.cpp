/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

// Spectral-decomposition GW (SD-GW): G0W0, evGW and evGW0 on top of an RHF/RKS or UHF/UKS
// reference.

#include "exachem/common/constants.hpp"
#include "exachem/gw/gw.hpp"
#include "exachem/scf/scf_iter.hpp"

#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::gw {

namespace {

using T = double;
using exachem::constants::ha2ev;

// Fetch the patch [lo, hi] (inclusive, row-major) of a dense tensor into buf.
// This is the "arbitrary slice" primitive: a GA patch get on the dense tensor's handle.
void get_dense_patch(Tensor<T>& tensor, std::vector<int64_t> lo, std::vector<int64_t> hi, T* buf) {
  const int ndims = tensor.num_modes();
  EXPECTS(tensor.kind() == TensorBase::TensorKind::dense);
  std::vector<int64_t> ld(std::max(ndims - 1, 1), 1);
  for(int i = 1; i < ndims; i++) ld[i - 1] = hi[i] - lo[i] + 1;
  NGA_Get64(tensor.ga_handle(), lo.data(), hi.data(), buf, ld.data());
}

// Visit every block of a tensor on the calling rank (used by rank 0 to gather/scatter
// small replicated matrices without going through an intermediate Eigen tensor).
template<typename Lambda>
void for_each_block(Tensor<T>& tensor, Lambda&& func) {
  for(const auto& blockid: tensor.loop_nest()) {
    std::vector<T> buf(tensor.block_size(blockid));
    func(blockid, tensor.block_offsets(blockid), tensor.block_dims(blockid), buf);
  }
}

// Per spin channel tensors and scalars in the MO basis.
struct SpinChannel {
  TiledIndexSpace     MO; // spatial MOs with occ/virt subspaces
  TiledIndexSpace     W;  // window virtuals [nocc, hi) (only if nw > 0)
  int                 nocc{}, nvir{}, nw{}, lo{}, hi{};
  Tensor<T>           Pov;                 // (i, a, K)  regular, used in contractions
  Tensor<T>           Poo_d, Pov_d, Pvv_d; // dense copies, used for per-QP row picks
  std::vector<double> evals;               // reference eigenvalues relative to the Fermi level
  std::vector<double> vxc;                 // <p|Vxc|p> for p in the window
  std::vector<double> sigmax;              // exchange self-energy for p in the window
  double              efermi{};
};

} // namespace

void gw_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using namespace exachem::scf;

  if(!chem_env.scf_context.skip_scf) scf::scf_driver(ec, chem_env);

  const auto rank  = ec.pg().rank();
  const bool mrank = (rank == 0);
  Scheduler  sch{ec};
  const auto ex_hw = ec.exhw();

  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};

  SystemData& sys_data   = chem_env.sys_data;
  GWOptions   gw_options = chem_env.ioptions.gw_options;
  const bool  debug      = mrank && gw_options.debug;

  if(mrank) gw_options.print();

  if(txt_utils::to_lower(gw_options.method) != "sdgw")
    tamm_terminate("GW error: only method=sdgw is implemented");
  if(gw_options.cdbasis.empty())
    tamm_terminate("GW error: CD basis set name (cdbasis) not provided!");
  if(sys_data.is_restricted_os) tamm_terminate("GW error: ROHF references are not supported");
  if(sys_data.is_ks && !chem_env.scf_context.has_vxc)
    tamm_terminate("GW error: KS reference without an exchange-correlation potential");

  GWData gwd = gw_pars(chem_env, mrank);
  if(debug) gwd.print();

  const int    ipol   = gwd.ipol;
  const int    nmo    = gwd.nmo;
  const double exx    = chem_env.scf_context.xHF;
  const double ieta   = gw_options.ieta;
  const bool   evgw   = gw_options.evgw;
  const bool   evgw0  = gw_options.evgw0;
  const T      factor = (ipol == 1) ? 2.0 : 1.0; // spin factor for W_mn

  // ------------------------------------------------------------------
  // Reference quantities: C, eps, Vxc (per spin channel)
  // ------------------------------------------------------------------
  const TiledIndexSpace& tAO   = chem_env.is_context.AO_opt;
  const tamm::Tile mo_tilesize = static_cast<tamm::Tile>(chem_env.ioptions.scf_options.AO_tilesize);

  std::vector<Tensor<T>> C_AO   = {chem_env.scf_context.C_AO, chem_env.scf_context.C_beta_AO};
  std::vector<Tensor<T>> F_AO   = {chem_env.scf_context.F_AO, chem_env.scf_context.F_beta_AO};
  std::vector<Tensor<T>> VXC_AO = {chem_env.scf_context.VXC_alpha_AO,
                                   chem_env.scf_context.VXC_beta_AO};

  std::vector<SpinChannel> spin(ipol);
  std::vector<Matrix>      C_eig(ipol);

  for(int s = 0; s < ipol; s++) {
    SpinChannel& sc = spin[s];
    sc.nocc         = gwd.nocc[s];
    sc.nvir         = gwd.nvir[s];
    sc.lo           = gwd.lo[s];
    sc.hi           = gwd.hi[s];
    sc.nw           = std::max(0, sc.hi - sc.nocc);
    sc.MO           = gw_setupMOIS(nmo, sc.nocc, mo_tilesize);
    if(sc.nw > 0) sc.W = TiledIndexSpace{IndexSpace{range(sc.nw)}, mo_tilesize};

    // Replicated small matrices: C (nao x nmo), F and Vxc (nao x nao)
    C_eig[s]     = tamm_to_eigen_matrix(C_AO[s]);
    Matrix F_eig = tamm_to_eigen_matrix(F_AO[s]);
    Matrix F_MO  = C_eig[s].transpose() * F_eig * C_eig[s];
    F_eig.resize(0, 0);
    EXPECTS(C_eig[s].cols() == nmo);

    sc.evals.resize(nmo);
    for(int p = 0; p < nmo; p++) sc.evals[p] = F_MO(p, p);
    F_MO.resize(0, 0);

    if(sc.nocc < 1 || sc.nocc >= nmo)
      tamm_terminate("GW error: need at least one occupied and one virtual orbital");
    sc.efermi = 0.5 * (sc.evals[sc.nocc] + sc.evals[sc.nocc - 1]);
    for(auto& e: sc.evals) e -= sc.efermi;

    // Vxc in the MO basis, diagonal elements of the window only
    sc.vxc.assign(sc.hi - sc.lo, 0.0);
    if(sys_data.is_ks) {
      Matrix V_eig = tamm_to_eigen_matrix(VXC_AO[s]);
      Matrix VC    = V_eig * C_eig[s].middleCols(sc.lo, sc.hi - sc.lo);
      for(int p = sc.lo; p < sc.hi; p++) sc.vxc[p - sc.lo] = C_eig[s].col(p).dot(VC.col(p - sc.lo));
    }
  }
  ec.pg().barrier();

  // ------------------------------------------------------------------
  // Three-center integrals in the CD basis, orthonormalized: xyK(mu,nu,K)
  // Reuses the SCF density-fitting machinery with a GW-owned SCFData.
  // ------------------------------------------------------------------
  auto t0 = std::chrono::high_resolution_clock::now();
  if(mrank) std::cout << "\n\t Two-center integrals ...               ";

  libint2::BasisSet cdbs(gw_options.cdbasis, chem_env.atoms);
  cdbs.set_pure(chem_env.ioptions.scf_options.gaussian_type == "spherical");
  const size_t ndf = cdbs.nbf();
  if(mrank) std::cout << "\n\t CD basis set rank = " << ndf << std::endl;

  SCFCompute<T> scf_compute;
  SCFIter<T>    scf_iter;
  ScalapackInfo scalapack_info;
  SCFData       gw_scf_data;
#if defined(USE_SCALAPACK)
  // In a ScaLAPACK build compute_Vm12 diagonalizes the CD-basis metric only on a valid
  // ScaLAPACK subgroup (there is no rank-0 LAPACK fallback), so set one up as SCF does.
  ProcGroupData pgdata =
    get_spg_data(ec, chem_env.shells.nbf(), -1, 50, chem_env.ioptions.scf_options.nnodes);
  setup_scalapack_info(ec, chem_env, scalapack_info, pgdata);
#endif

  gw_scf_data.tAO = tAO;
  std::tie(gw_scf_data.shell_tile_map, gw_scf_data.AO_tiles, gw_scf_data.AO_opttiles) =
    scf_compute.compute_AO_tiles(ec, chem_env, chem_env.shells);
  std::tie(gw_scf_data.mu, gw_scf_data.nu, gw_scf_data.ku) = tAO.labels<3>("all");

  gw_scf_data.dfbs = cdbs;
  gw_scf_data.dfAO = IndexSpace{range(0, ndf)};
  std::tie(gw_scf_data.df_shell_tile_map, gw_scf_data.dfAO_tiles, gw_scf_data.dfAO_opttiles) =
    scf_compute.compute_AO_tiles(ec, chem_env, cdbs, true);
  gw_scf_data.tdfAO  = TiledIndexSpace{gw_scf_data.dfAO, gw_scf_data.dfAO_opttiles};
  gw_scf_data.tdfAOt = TiledIndexSpace{gw_scf_data.dfAO, gw_scf_data.dfAO_tiles};
  std::tie(gw_scf_data.d_mu, gw_scf_data.d_nu, gw_scf_data.d_ku) =
    gw_scf_data.tdfAO.labels<3>("all");

  const TiledIndexSpace& tdfAO = gw_scf_data.tdfAO;
  auto                   K     = tdfAO.label("all");
  auto [mu, nu]                = tAO.labels<2>("all");

  TAMMTensors<T>& tt = gw_scf_data.ttensors;
  tt.xyZ             = Tensor<T>{tAO, tAO, tdfAO};
  tt.xyK             = Tensor<T>{tAO, tAO, tdfAO};
  tt.Vm1             = Tensor<T>{tdfAO, tdfAO};
  Tensor<T>::allocate(&ec, tt.xyK);

  if(mrank) std::cout << "\t Three-center integrals in AO basis ... ";
  // computes V^{-1/2} (Vm1), the 3c integrals (xyZ) and xyK = xyZ * Vm1
  scf_iter.init_ri(ec, chem_env, scalapack_info, gw_scf_data, gw_scf_data.etensors, tt);
  Tensor<T>::deallocate(tt.Vm1);
#if defined(USE_SCALAPACK)
  if(scalapack_info.pg.is_valid()) {
    scalapack_info.ec.flush_and_sync();
    scalapack_info.ec.pg().destroy_coll();
  }
#endif
  Tensor<T>& xyK = tt.xyK;
  if(mrank)
    std::cout << gw_strfmt("\t %8.2f seconds",
                           std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::high_resolution_clock::now() - t0)
                             .count())
              << std::endl;

  // ------------------------------------------------------------------
  // Transform to the MO basis: Poo(i,j,K), Pov(i,a,K), Pvv(w,b,K) for window virtuals w
  // ------------------------------------------------------------------
  t0 = std::chrono::high_resolution_clock::now();
  if(mrank) std::cout << "\t Three-center integrals in MO basis ... ";

  for(int s = 0; s < ipol; s++) {
    SpinChannel& sc = spin[s];
    auto [i, j]     = sc.MO.labels<2>("occ");
    auto [a, b]     = sc.MO.labels<2>("virt");
    TiledIndexLabel w;
    if(sc.nw > 0) w = sc.W.label("all");

    Tensor<T> C_t{tAO, sc.MO};
    Tensor<T> tmp_o{tAO, sc.MO("occ"), tdfAO};
    Tensor<T> Poo{sc.MO("occ"), sc.MO("occ"), tdfAO};
    sc.Pov = Tensor<T>{sc.MO("occ"), sc.MO("virt"), tdfAO};
    sch.allocate(C_t, tmp_o, Poo, sc.Pov).execute();
    if(mrank) eigen_to_tamm_tensor(C_t, C_eig[s]);
    ec.pg().barrier();

    // clang-format off
    sch(tmp_o(nu, i, K)  = xyK(mu, nu, K) * C_t(mu, i))
       (Poo(i, j, K)     = tmp_o(nu, j, K) * C_t(nu, i))
       (sc.Pov(i, a, K)  = tmp_o(nu, i, K) * C_t(nu, a))
       .deallocate(tmp_o)
       .execute(ex_hw);
    // clang-format on

    Tensor<T> Pvv;
    if(sc.nw > 0) {
      Tensor<T> Cw{tAO, sc.W};
      Tensor<T> tmp_w{tAO, sc.W, tdfAO};
      Pvv = Tensor<T>{sc.W, sc.MO("virt"), tdfAO};
      sch.allocate(Cw, tmp_w, Pvv).execute();
      if(mrank) {
        Matrix Cw_eig = C_eig[s].middleCols(sc.nocc, sc.nw);
        eigen_to_tamm_tensor(Cw, Cw_eig);
      }
      ec.pg().barrier();
      // clang-format off
      sch(tmp_w(nu, w, K) = xyK(mu, nu, K) * Cw(mu, w))
         (Pvv(w, b, K)    = tmp_w(nu, w, K) * C_t(nu, b))
         .deallocate(tmp_w, Cw)
         .execute(ex_hw);
      // clang-format on
    }
    sch.deallocate(C_t).execute();

    // Exchange self-energy for the window: Sigma_x(p) = -sum_j sum_K P(p,j,K)^2
    // (this is the only use of the bare-Coulomb super-diagonal V_mn)
    std::vector<double> sx_occ(sc.nocc, 0.0), sx_vir(sc.nvir, 0.0);
    {
      std::vector<double> l_occ(sc.nocc, 0.0), l_vir(sc.nvir, 0.0);
      auto                poo_lambda = [&](const IndexVector& blockid) {
        std::vector<T> buf(Poo.block_size(blockid));
        Poo.get(blockid, buf);
        const auto bd = Poo.block_dims(blockid);
        const auto bo = Poo.block_offsets(blockid);
        for(size_t ii = 0; ii < bd[0]; ii++)
          for(size_t jj = 0; jj < bd[1]; jj++)
            for(size_t kk = 0; kk < bd[2]; kk++) {
              const T v = buf[(ii * bd[1] + jj) * bd[2] + kk];
              l_occ[bo[0] + ii] += v * v;
            }
      };
      auto pov_lambda = [&](const IndexVector& blockid) {
        std::vector<T> buf(sc.Pov.block_size(blockid));
        sc.Pov.get(blockid, buf);
        const auto bd = sc.Pov.block_dims(blockid);
        const auto bo = sc.Pov.block_offsets(blockid);
        for(size_t ii = 0; ii < bd[0]; ii++)
          for(size_t aa = 0; aa < bd[1]; aa++)
            for(size_t kk = 0; kk < bd[2]; kk++) {
              const T v = buf[(ii * bd[1] + aa) * bd[2] + kk];
              l_vir[bo[1] + aa] += v * v; // Pov offsets are relative to the virt subspace
            }
      };
      block_for(ec, Poo(), poo_lambda);
      block_for(ec, sc.Pov(), pov_lambda);
      ec.pg().allreduce(l_occ.data(), sx_occ.data(), sc.nocc, ReduceOp::sum);
      ec.pg().allreduce(l_vir.data(), sx_vir.data(), sc.nvir, ReduceOp::sum);
    }
    sc.sigmax.resize(sc.hi - sc.lo);
    for(int p = sc.lo; p < sc.hi; p++)
      sc.sigmax[p - sc.lo] = (p < sc.nocc) ? -sx_occ[p] : -sx_vir[p - sc.nocc];

    if(debug) {
      std::cout << "\n[GW debug] spin " << s << ": efermi = " << sc.efermi << " Ha, exx = " << exx
                << std::endl;
      std::cout << "[GW debug]  p   eps(abs, eV)   vxc(eV)   sigmax(eV)" << std::endl;
      for(int p = sc.lo; p < sc.hi; p++)
        std::cout << gw_strfmt("[GW debug] %3d %12.4f %12.4f %12.4f", p + 1,
                               (sc.evals[p] + sc.efermi) * ha2ev, sc.vxc[p - sc.lo] * ha2ev,
                               sc.sigmax[p - sc.lo] * ha2ev)
                  << std::endl;
    }

    // Dense copies for the per-QP row picks
    sc.Poo_d = to_dense_tensor(ec_dense, Poo);
    sc.Pov_d = to_dense_tensor(ec_dense, sc.Pov);
    Tensor<T>::deallocate(Poo);
    if(sc.nw > 0) {
      sc.Pvv_d = to_dense_tensor(ec_dense, Pvv);
      Tensor<T>::deallocate(Pvv);
    }
  }
  Tensor<T>::deallocate(xyK);
  if(mrank)
    std::cout << gw_strfmt("\t %8.2f seconds",
                           std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::high_resolution_clock::now() - t0)
                             .count())
              << std::endl;

  // ------------------------------------------------------------------
  // GW iterations
  // ------------------------------------------------------------------
  int              n_ov = 0;
  std::vector<int> ov_offset(ipol, 0);
  for(int s = 0; s < ipol; s++) {
    ov_offset[s] = n_ov;
    n_ov += spin[s].nocc * spin[s].nvir;
  }
  const tamm::Tile s_tilesize = static_cast<tamm::Tile>(std::min(n_ov, 128));
  TiledIndexSpace  S{IndexSpace{range(n_ov)}, s_tilesize};
  auto [t_lbl] = S.labels<1>("all");

  Tensor<T>           Qs{S, tdfAO}; // Q_s(K) = sum_ia Pov(ia,K) (X+Y)(ia,s)
  std::vector<double> Omega(n_ov, 0.0);
  bool                Qs_allocated = false;

  std::vector<std::vector<double>> newevals(ipol), oldevals(ipol);
  for(int s = 0; s < ipol; s++) newevals[s] = spin[s].evals;

  json& jgw = sys_data.results["output"]["GW"];
  for(int s = 0; s < ipol; s++)
    jgw["fermi_level_eV"][(s == 0) ? "alpha" : "beta"] = spin[s].efermi * ha2ev;

  const auto gw_t0 = std::chrono::high_resolution_clock::now();

  for(int eviter = 0; eviter < gwd.maxev; eviter++) {
    std::string iter_label = "G0W0";
    if(evgw) iter_label = gw_strfmt("G%dW%d", eviter, eviter);
    else if(evgw0) iter_label = gw_strfmt("G%dW0", eviter);
    if(mrank) std::cout << "\n\t " << iter_label << std::endl;

    for(int s = 0; s < ipol; s++) oldevals[s] = newevals[s];

    // ---------------- RPA polarizability (screening) ----------------
    if(eviter == 0 || evgw) {
      // wia: eigenvalue differences, concatenated over spin channels
      std::vector<double> wall(n_ov);
      for(int s = 0; s < ipol; s++) {
        const SpinChannel& sc = spin[s];
        for(int i = 0; i < sc.nocc; i++)
          for(int a = 0; a < sc.nvir; a++)
            wall[ov_offset[s] + i * sc.nvir + a] = newevals[s][sc.nocc + a] - newevals[s][i];
      }

      // RPA matrix on rank 0, built from the distributed (ia|jb) contractions
      std::vector<double> RPA;
      if(mrank) RPA.assign(static_cast<size_t>(n_ov) * n_ov, 0.0);

      for(int s1 = 0; s1 < ipol; s1++) {
        for(int s2 = s1; s2 < ipol; s2++) {
          SpinChannel& c1 = spin[s1];
          SpinChannel& c2 = spin[s2];
          auto [i]        = c1.MO.labels<1>("occ");
          auto [a]        = c1.MO.labels<1>("virt");
          auto [j]        = c2.MO.labels<1>("occ");
          auto [b]        = c2.MO.labels<1>("virt");
          Tensor<T> R4{c1.MO("occ"), c1.MO("virt"), c2.MO("occ"), c2.MO("virt")};
          sch.allocate(R4)(R4(i, a, j, b) = c1.Pov(i, a, K) * c2.Pov(j, b, K)).execute(ex_hw);

          if(mrank) {
            const double pref = (ipol == 1) ? 4.0 : 2.0;
            for_each_block(R4, [&](const IndexVector& blockid, const auto& bo, const auto& bd,
                                   std::vector<T>& buf) {
              R4.get(blockid, buf);
              for(size_t ii = 0; ii < bd[0]; ii++)
                for(size_t aa = 0; aa < bd[1]; aa++)
                  for(size_t jj = 0; jj < bd[2]; jj++)
                    for(size_t bb = 0; bb < bd[3]; bb++) {
                      const size_t r    = ov_offset[s1] + (bo[0] + ii) * c1.nvir + (bo[1] + aa);
                      const size_t c    = ov_offset[s2] + (bo[2] + jj) * c2.nvir + (bo[3] + bb);
                      const T      v    = pref * buf[((ii * bd[1] + aa) * bd[2] + jj) * bd[3] + bb];
                      RPA[r * n_ov + c] = v;
                      if(s1 != s2) RPA[c * n_ov + r] = v;
                    }
            });
          }
          ec.pg().barrier();
          Tensor<T>::deallocate(R4);
        }
      }

      // Casida-type symmetric eigenproblem: rank-0 LAPACK (ScaLAPACK path: later)
      std::vector<double> XPY; // (X+Y)(r,s) stored column-major in s
      if(mrank) {
        std::vector<double> AmB(n_ov);
        for(int r = 0; r < n_ov; r++) {
          RPA[static_cast<size_t>(r) * n_ov + r] += wall[r];
          AmB[r] = std::sqrt(wall[r]);
        }
        for(int r = 0; r < n_ov; r++)
          for(int c = 0; c < n_ov; c++) RPA[static_cast<size_t>(r) * n_ov + c] *= AmB[r] * AmB[c];

        std::vector<double> lam(n_ov);
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, n_ov, RPA.data(), n_ov, lam.data());
        // RPA now holds the eigenvectors U(r,s) = RPA[s*n_ov + r] (column-major)

        for(int t = 0; t < n_ov; t++) {
          if(lam[t] <= 0.0)
            tamm_terminate("GW error: non-positive RPA excitation energy (unstable reference?)");
          Omega[t] = std::sqrt(lam[t]);
        }
        XPY.resize(static_cast<size_t>(n_ov) * n_ov);
        for(int t = 0; t < n_ov; t++) {
          const double sw = 1.0 / std::sqrt(Omega[t]);
          for(int r = 0; r < n_ov; r++)
            XPY[static_cast<size_t>(t) * n_ov + r] =
              AmB[r] * RPA[static_cast<size_t>(t) * n_ov + r] * sw;
        }
        RPA.clear();
        RPA.shrink_to_fit();
      }
      ec.pg().broadcast(Omega.data(), n_ov, 0);
      if(debug)
        std::cout << gw_strfmt("[GW debug] n_ov = %d, Omega (eV): min %10.4f, max %10.4f", n_ov,
                               Omega.front() * ha2ev, Omega.back() * ha2ev)
                  << std::endl;

      // Qs(t,K) = sum_s sum_ia Pov_s(i,a,K) (X+Y)_s(i,a,t)
      if(!Qs_allocated) {
        sch.allocate(Qs).execute();
        Qs_allocated = true;
      }
      sch(Qs() = 0.0).execute();
      for(int s = 0; s < ipol; s++) {
        SpinChannel& sc = spin[s];
        auto [i]        = sc.MO.labels<1>("occ");
        auto [a]        = sc.MO.labels<1>("virt");
        Tensor<T> XPY_t{sc.MO("occ"), sc.MO("virt"), S};
        sch.allocate(XPY_t).execute();
        if(mrank) {
          for_each_block(XPY_t, [&](const IndexVector& blockid, const auto& bo, const auto& bd,
                                    std::vector<T>& buf) {
            for(size_t ii = 0; ii < bd[0]; ii++)
              for(size_t aa = 0; aa < bd[1]; aa++)
                for(size_t tt = 0; tt < bd[2]; tt++) {
                  const size_t r = ov_offset[s] + (bo[0] + ii) * sc.nvir + (bo[1] + aa);
                  const size_t t = bo[2] + tt;
                  buf[(ii * bd[1] + aa) * bd[2] + tt] = XPY[t * n_ov + r];
                }
            XPY_t.put(blockid, buf);
          });
        }
        ec.pg().barrier();
        sch(Qs(t_lbl, K) += sc.Pov(i, a, K) * XPY_t(i, a, t_lbl)).deallocate(XPY_t).execute(ex_hw);
      }
      if(mrank) {
        XPY.clear();
        XPY.shrink_to_fit();
      }
    }

    // ---------------- Quasiparticle equations, per spin channel ----------------
    for(int s = 0; s < ipol; s++) {
      SpinChannel&      sc     = spin[s];
      const std::string sname  = (s == 0) ? "Alpha" : "Beta";
      const std::string slabel = (s == 0) ? "alpha" : "beta";
      const int         nocc   = sc.nocc;
      const int         nvir   = sc.nvir;
      const int         nqp    = gwd.nqp[s];
      const int         lo     = sc.lo;

      if(nqp < 1) continue;

      if(mrank) {
        std::cout << gw_strfmt("\t                 %s Orbitals             ", sname.c_str())
                  << std::endl;
        std::cout << "\t      State      Energy (eV)      Error (eV)  " << std::endl;
        std::cout << "\t      --------------------------------------  " << std::endl;
      }

      bool                warning = false;
      std::vector<bool>   fixed(nqp, false);
      std::vector<double> esterror(nqp, 0.0);

      std::vector<double> window_vals(oldevals[s].begin() + lo, oldevals[s].end());
      std::vector<int>    clusters = gw_findclusters(window_vals, nqp);

      auto [p_lbl] = sc.MO.labels<1>("all");

      int ulqp = -1;
      for(size_t icluster = 0; icluster < clusters.size(); icluster++) {
        const int llqp = ulqp + 1;
        ulqp           = ulqp + clusters[icluster];
        int mylo       = llqp;
        int myhi       = ulqp;

        while(true) {
          // occupied states: from upper to lower; virtual states: from lower to upper
          const int iqp = (lo + llqp < nocc) ? myhi : mylo;
          const int imo = lo + iqp;

          double eout = oldevals[s][imo];
          if(eviter < 2) { // guess from the previously solved QP of the cluster
            if(myhi < ulqp) eout = newevals[s][imo + 1];
            else if(mylo > llqp) eout = newevals[s][imo - 1];
          }

          // ---- W_mn for this state: wmn(n,t) = factor * [sum_K P(imo,n,K) Qs(t,K)]^2 ----
          Tensor<T> Pq{sc.MO, tdfAO};
          Tensor<T> wmn{sc.MO, S};
          sch.allocate(Pq, wmn).execute();
          if(mrank) {
            Matrix Pq_eig(nmo, ndf);
            Pq_eig.setZero();
            const int64_t kd = static_cast<int64_t>(ndf) - 1;
            if(imo < nocc) {
              // rows j: Poo(imo, j, K); rows a: Pov(imo, a, K)
              get_dense_patch(sc.Poo_d, {imo, 0, 0}, {imo, nocc - 1, kd}, Pq_eig.data());
              get_dense_patch(sc.Pov_d, {imo, 0, 0}, {imo, nvir - 1, kd},
                              Pq_eig.data() + static_cast<size_t>(nocc) * ndf);
            }
            else {
              const int w = imo - nocc;
              // rows j: Pov(j, w, K) (strided patch); rows a: Pvv(w, a, K)
              get_dense_patch(sc.Pov_d, {0, w, 0}, {nocc - 1, w, kd}, Pq_eig.data());
              get_dense_patch(sc.Pvv_d, {w, 0, 0}, {w, nvir - 1, kd},
                              Pq_eig.data() + static_cast<size_t>(nocc) * ndf);
            }
            eigen_to_tamm_tensor(Pq, Pq_eig);
          }
          ec.pg().barrier();
          sch(wmn(p_lbl, t_lbl) = Pq(p_lbl, K) * Qs(t_lbl, K)).deallocate(Pq).execute(ex_hw);

          // cache this rank's share of wmn^2 locally for the Newton iterations
          struct WBlock {
            size_t         n0, t0, dn, dt;
            std::vector<T> w2;
          };
          std::vector<WBlock> wblocks;
          auto                cache_lambda = [&](const IndexVector& blockid) {
            WBlock     blk;
            const auto bd = wmn.block_dims(blockid);
            const auto bo = wmn.block_offsets(blockid);
            blk.n0        = bo[0];
            blk.t0        = bo[1];
            blk.dn        = bd[0];
            blk.dt        = bd[1];
            blk.w2.resize(wmn.block_size(blockid));
            wmn.get(blockid, blk.w2);
            for(auto& v: blk.w2) v = factor * v * v;
            wblocks.push_back(std::move(blk));
          };
          block_for(ec, wmn(), cache_lambda);
          Tensor<T>::deallocate(wmn);

          // Sigma_c(omega) and its derivative, reduced over all ranks
          auto getsigmac = [&](double omega, double& sigmac, double& dsigmac) {
            double lsum[2] = {0.0, 0.0};
            for(const auto& blk: wblocks) {
              for(size_t nn = 0; nn < blk.dn; nn++) {
                const double e   = oldevals[s][blk.n0 + nn];
                const double sgn = (e > 0.0) ? -1.0 : ((e < 0.0) ? 1.0 : 0.0); // -sign(e)
                for(size_t tt = 0; tt < blk.dt; tt++) {
                  const double w2    = blk.w2[nn * blk.dt + tt];
                  const double temp  = omega - e + sgn * Omega[blk.t0 + tt];
                  const double denom = 1.0 / (temp * temp + 9.0 * ieta * ieta);
                  lsum[0] += w2 * temp * denom;
                  lsum[1] += w2 * (ieta * ieta - temp * temp) * denom * denom;
                }
              }
            }
            double gsum[2] = {0.0, 0.0};
            ec.pg().allreduce(lsum, gsum, 2, ReduceOp::sum);
            sigmac  = gsum[0];
            dsigmac = gsum[1];
          };

          if(gw_options.debug) { // collective: every rank must call getsigmac
            double s0, ds0;
            getsigmac(oldevals[s][imo], s0, ds0);
            if(mrank)
              std::cout << gw_strfmt(
                             "[GW debug] state %3d: Sigma_c(eps) = %10.4f eV, dSigma_c = %10.4f",
                             imo + 1, s0 * ha2ev, ds0)
                        << std::endl;
          }

          // ---- Newton iteration with bracketing ----
          const int           maxnewton = gw_options.maxnewton;
          std::vector<double> values(maxnewton, 0.0), errors(maxnewton, 0.0);
          double              eupper = 1.0, elower = 0.0, rupper = 0.0, rlower = 0.0;
          const double        constant = sc.evals[imo] - sc.vxc[iqp] + (1.0 - exx) * sc.sigmax[iqp];
          bool                bracket  = false;
          bool                converged = false;
          double              residual  = 0.0;

          for(int inewton = 0; inewton < maxnewton; inewton++) {
            const double ein = eout;
            double       sigmac, dsigmac;
            getsigmac(ein, sigmac, dsigmac);

            residual               = sigmac - ein + constant;
            const double dresidual = dsigmac - 1.0;
            values[inewton]        = ein;
            errors[inewton]        = residual;

            if(!bracket && inewton > 0) {
              if(errors[inewton] * errors[inewton - 1] < 0.0) {
                bracket = true;
                if(values[inewton] > values[inewton - 1]) {
                  elower = values[inewton - 1];
                  eupper = values[inewton];
                  rlower = errors[inewton - 1];
                  rupper = errors[inewton];
                }
                else {
                  elower = values[inewton];
                  eupper = values[inewton - 1];
                  rlower = errors[inewton];
                  rupper = errors[inewton - 1];
                }
              }
            }
            else if(bracket) {
              if(std::abs(rupper) < std::abs(rlower)) {
                if(errors[inewton] * rupper < 0.0) {
                  elower = values[inewton];
                  rlower = errors[inewton];
                }
                else if(errors[inewton] * rlower < 0.0) {
                  eupper = values[inewton];
                  rupper = errors[inewton];
                }
              }
              else {
                if(errors[inewton] * rlower < 0.0) {
                  eupper = values[inewton];
                  rupper = errors[inewton];
                }
                else if(errors[inewton] * rupper < 0.0) {
                  elower = values[inewton];
                  rlower = errors[inewton];
                }
              }
            }

            converged = std::abs(residual) < 0.005 / ha2ev ||
                        (bracket && std::abs(eupper - elower) < 0.005 / ha2ev);

            if(converged) {
              eout = ein;
              if(debug) {
                std::cout << iqp << std::endl;
                gw_print_iter(inewton, ein + sc.efermi, eout + sc.efermi, elower + sc.efermi,
                              eupper + sc.efermi, bracket);
              }
              break;
            }

            const double z    = -1.0 / dresidual;
            const double step = z * residual;

            if(z > 0.3 && z < 1.0) eout = ein + step;
            else if(bracket && inewton % 3 == 0) eout = elower + 0.6180 * (eupper - elower);
            else if(bracket) eout = eupper - 0.6180 * (eupper - elower);
            else if(z > 0.1) eout = ein + 0.6180 * step;
            else eout = ein + ((residual > 0) - (residual < 0)) * 0.005;

            if(debug) {
              std::cout << iqp << std::endl;
              gw_print_iter(inewton, ein + sc.efermi, eout + sc.efermi, elower + sc.efermi,
                            eupper + sc.efermi, bracket);
            }
          }

          newevals[s][imo] = eout;
          if(converged) fixed[iqp] = true;
          esterror[iqp] = bracket ? std::min(eupper - elower, std::abs(residual))
                                  : std::abs(residual);

          if(imo < nocc) myhi -= 1;
          else mylo += 1;
          if(mylo > myhi) break;
        }

        // Print output for the states in the current cluster
        if(mrank) {
          for(int jqp = llqp; jqp <= ulqp; jqp++) {
            const int state = lo + jqp + 1;
            std::cout << gw_strfmt("\t       %3d        %8.3f       %8.3f", state,
                                   (newevals[s][lo + jqp] + sc.efermi) * ha2ev,
                                   esterror[jqp] * ha2ev);
            if(fixed[jqp]) std::cout << std::endl;
            else {
              warning = true;
              std::cout << " ***" << std::endl;
            }
          }
        }
      }

      if(mrank) {
        std::cout << "\t      --------------------------------------  " << std::endl;
        if(warning) std::cout << "\n\t *** Result did not converge\n" << std::endl;

        json& jit = jgw["iterations"][iter_label][slabel];
        for(int jqp = 0; jqp < nqp; jqp++) {
          jit["state"].push_back(lo + jqp + 1);
          jit["energy_eV"].push_back((newevals[s][lo + jqp] + sc.efermi) * ha2ev);
          jit["error_eV"].push_back(esterror[jqp] * ha2ev);
          jit["converged"].push_back(static_cast<bool>(fixed[jqp]));
        }
      }
    }

    // Scissor shift for evGW and evGW_0 calculations
    if(evgw || evgw0) {
      for(int s = 0; s < ipol; s++) {
        SpinChannel& sc = spin[s];
        gw_scissor(oldevals[s], newevals[s], gwd.noqp[s], gwd.nvqp[s], sc.nocc, sc.lo, sc.hi, nmo,
                   (s == 0) ? "Alpha" : "Beta", mrank);
      }
    }
  }

  if(mrank) std::cout << "\t GW iteration Done! " << std::endl;

  // ------------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------------
  if(Qs_allocated) Tensor<T>::deallocate(Qs);
  for(int s = 0; s < ipol; s++) {
    SpinChannel& sc = spin[s];
    Tensor<T>::deallocate(sc.Pov, sc.Poo_d, sc.Pov_d);
    if(sc.nw > 0) Tensor<T>::deallocate(sc.Pvv_d);
  }
  Tensor<T>::deallocate(chem_env.scf_context.C_AO, chem_env.scf_context.F_AO);
  if(sys_data.is_unrestricted)
    Tensor<T>::deallocate(chem_env.scf_context.C_beta_AO, chem_env.scf_context.F_beta_AO);
  if(chem_env.scf_context.has_vxc) {
    Tensor<T>::deallocate(chem_env.scf_context.VXC_alpha_AO);
    if(sys_data.is_unrestricted) Tensor<T>::deallocate(chem_env.scf_context.VXC_beta_AO);
    chem_env.scf_context.has_vxc = false;
  }

  if(mrank) {
    std::cout << std::endl
              << "Time taken for " << txt_utils::to_upper(gw_options.method) << ": " << std::fixed
              << std::setprecision(2)
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::high_resolution_clock::now() - gw_t0)
                   .count()
              << " secs" << std::endl;
    chem_env.write_json_data();
  }
}

} // namespace exachem::gw
