/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/gfcc/gfccsd_ip_a.hpp"
using T = double;
namespace exachem::cc::gfcc {

template<typename T>
void GFCCSD_IP_A_Driver<T>::gfccsd_driver_ip_a(
  ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO, Tensor<T>& t1_a,
  Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab, Tensor<T>& f1,
  Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
  Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
  Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
  Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
  Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
  Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
  Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
  Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba, Tensor<T>& v2ijab_aaaa,
  Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, std::vector<T>& p_evl_sorted_occ,
  std::vector<T>& p_evl_sorted_virt, const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi,
  const TiledIndexSpace& unit_tis, std::string files_prefix, std::string levelstr,
  double gf_omega) {
  using ComplexTensor  = Tensor<std::complex<T>>;
  using VComplexTensor = std::vector<Tensor<std::complex<T>>>;
  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const CCSDOptions& ccsd_options       = chem_env.ioptions.ccsd_options;
  auto               debug              = ccsd_options.debug;
  const int          gf_nprocs_poi      = ccsd_options.gf_nprocs_poi;
  const size_t       gf_maxiter         = ccsd_options.gf_maxiter;
  const double       gf_eta             = ccsd_options.gf_eta;
  const double       gf_lshift          = ccsd_options.gf_lshift;
  const double       gf_threshold       = ccsd_options.gf_threshold;
  const bool         gf_preconditioning = ccsd_options.gf_preconditioning;
  // const double gf_damping_factor    = ccsd_options.gf_damping_factor;
  std::vector<size_t> gf_orbitals = ccsd_options.gf_orbitals;

  ProcGroup&        sub_pg = chem_env.cc_context.sub_pg;
  ExecutionContext& sub_ec = (*chem_env.cc_context.sub_ec);

  const int noa = chem_env.sys_data.n_occ_alpha;
  // const int total_orbitals = chem_env.sys_data.nmo;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  // const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [p1_va]        = v_alpha.labels<1>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob, h2_ob] = o_beta.labels<2>("all");

  std::cout.precision(15);

  Scheduler gsch{gec};
  auto      rank = gec.pg().rank();

  std::stringstream gfo;
  gfo << std::fixed << std::setprecision(2) << gf_omega;

  // PRINT THE HEADER FOR GF-CCSD ITERATIONS
  if(rank == 0) {
    std::stringstream gfp;
    gfp << std::endl << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_a{o_alpha};
  ComplexTensor dtmp_aaa{v_alpha, o_alpha, o_alpha};
  ComplexTensor dtmp_bab{v_beta, o_alpha, o_beta};
  ComplexTensor::allocate(&gec, dtmp_a, dtmp_aaa, dtmp_bab);

  // double au2ev = 27.2113961;

  std::string dtmp_a_file   = files_prefix + ".W" + gfo.str() + ".r_dtmp_a.l" + levelstr;
  std::string dtmp_aaa_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_aaa.l" + levelstr;
  std::string dtmp_bab_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_bab.l" + levelstr;

  if(fs::exists(dtmp_a_file) && fs::exists(dtmp_aaa_file) && fs::exists(dtmp_bab_file)) {
    read_from_disk(dtmp_a, dtmp_a_file);
    read_from_disk(dtmp_aaa, dtmp_aaa_file);
    read_from_disk(dtmp_bab, dtmp_bab_file);
  }
  else {
    ComplexTensor DEArr_IP1{O};
    ComplexTensor DEArr_IP2{V, O, O};

    double denominator = 0.0;
    //
    auto DEArr_lambda1 = [&](const IndexVector& bid) {
      const IndexVector            blockid = internal::translate_blockid(bid, DEArr_IP1());
      const TAMM_SIZE              size    = DEArr_IP1.block_size(blockid);
      std::vector<std::complex<T>> buf(size);

      auto   block_dims   = DEArr_IP1.block_dims(blockid);
      auto   block_offset = DEArr_IP1.block_offsets(blockid);
      size_t c            = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        denominator = gf_omega - p_evl_sorted_occ[i];
        if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * gf_lshift; }
        else if(denominator > 0.0 && denominator < 1.0) { denominator += 1.0 * gf_lshift; }
        buf[c] = 1.0 / std::complex<T>(denominator, -1.0 * gf_eta);
      }
      DEArr_IP1.put(blockid, buf);
    };
    //
    auto DEArr_lambda2 = [&](const IndexVector& bid) {
      const IndexVector            blockid = internal::translate_blockid(bid, DEArr_IP2());
      const TAMM_SIZE              size    = DEArr_IP2.block_size(blockid);
      std::vector<std::complex<T>> buf(size);

      auto   block_dims   = DEArr_IP2.block_dims(blockid);
      auto   block_offset = DEArr_IP2.block_offsets(blockid);
      size_t c            = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            denominator =
              gf_omega + p_evl_sorted_virt[i] - p_evl_sorted_occ[j] - p_evl_sorted_occ[k];
            if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * gf_lshift; }
            else if(denominator > 0.0 && denominator < 1.0) { denominator += 1.0 * gf_lshift; }
            buf[c] = 1.0 / std::complex<T>(denominator, -1.0 * gf_eta);
          }
        }
      }
      DEArr_IP2.put(blockid, buf);
    };

    gsch.allocate(DEArr_IP1).execute();
    gsch.allocate(DEArr_IP2).execute();
    if(sub_pg.is_valid()) {
      Scheduler sub_sch{sub_ec};
      sub_sch(DEArr_IP1() = 0).execute();
      sub_sch(DEArr_IP2() = 0).execute();
      block_for(sub_ec, DEArr_IP1(), DEArr_lambda1);
      block_for(sub_ec, DEArr_IP2(), DEArr_lambda2);
      sub_sch(dtmp_a() = 0)(dtmp_aaa() = 0)(dtmp_bab() = 0)(dtmp_a(h1_oa) = DEArr_IP1(h1_oa))(
        dtmp_aaa(p1_va, h1_oa, h2_oa) = DEArr_IP2(p1_va, h1_oa, h2_oa))(
        dtmp_bab(p1_vb, h1_oa, h2_ob) = DEArr_IP2(p1_vb, h1_oa, h2_ob))
        .execute();
    }
    gec.pg().barrier();
    gsch.deallocate(DEArr_IP1).execute();
    gsch.deallocate(DEArr_IP2).execute();
    write_to_disk(dtmp_a, dtmp_a_file);
    write_to_disk(dtmp_aaa, dtmp_aaa_file);
    write_to_disk(dtmp_bab, dtmp_bab_file);
  }

  //------------------------
  auto nranks = gec.pg().size().value();

  const size_t        num_oi           = noa;
  size_t              num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  if(!gf_orbitals.empty()) pi_tbp = gf_orbitals;
  // Check pi's already processed
  for(size_t pi = 0; pi < num_oi; pi++) {
    std::string x1_a_conv_wpi_file =
      files_prefix + ".x1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string x2_aaa_conv_wpi_file =
      files_prefix + ".x2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string x2_bab_conv_wpi_file =
      files_prefix + ".x2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);

    if(fs::exists(x1_a_conv_wpi_file) && fs::exists(x2_aaa_conv_wpi_file) &&
       fs::exists(x2_bab_conv_wpi_file))
      num_pi_processed++;
    else if(std::find(gf_orbitals.begin(), gf_orbitals.end(), pi) == gf_orbitals.end())
      pi_tbp.push_back(pi);
  }

  size_t num_pi_remain = num_oi - num_pi_processed;
  if(num_pi_remain == 0) {
    gsch.deallocate(dtmp_a, dtmp_aaa, dtmp_bab).execute();
    return;
  }
  EXPECTS(num_pi_remain == pi_tbp.size());
  const int  ppn       = gec.ppn();
  const int  nnodes    = gec.nnodes();
  int        subranks  = std::floor(nranks / num_pi_remain);
  const bool no_pg     = (subranks == 0 || subranks == 1);
  int        sub_nodes = 0;

  if(no_pg) {
    subranks  = nranks;
    sub_nodes = nnodes;
  }
  else {
    int sub_nodes = subranks / ppn;
    if(subranks % ppn > 0 || sub_nodes == 0) sub_nodes++;
    if(sub_nodes > nnodes) sub_nodes = nnodes;
    subranks = sub_nodes * ppn;
  }

  if(gf_nprocs_poi > 0) {
    if(nnodes > 1 && gf_nprocs_poi % ppn != 0)
      tamm_terminate("[ERROR] gf_nprocs_poi should be a muliple of user mpi ranks per node");
    if(nnodes == 1) {
      // TODO: This applies only when using GA's PR runtime
      int ga_num_pr = 1;
      if(const char* ga_npr = std::getenv("GA_NUM_PROGRESS_RANKS_PER_NODE")) {
        ga_num_pr = std::atoi(ga_npr);
      }
      if(ga_num_pr > 1)
        tamm_terminate("[ERROR] use of multiple GA progress ranks for a single node gfccsd "
                       "calculation is not allowed");
    }
    subranks  = gf_nprocs_poi;
    sub_nodes = subranks / ppn;
  }
  if(sub_nodes == 0) sub_nodes++;

  int num_oi_can_bp = nnodes / sub_nodes;
  if(nnodes % sub_nodes > 0) num_oi_can_bp++;
  // when using 1 node
  if(gf_nprocs_poi > 0 && nnodes == 1) {
    num_oi_can_bp = ppn / gf_nprocs_poi;
    if(ppn % gf_nprocs_poi > 0) num_oi_can_bp++;
  }

  if(rank == 0) {
    std::cout << "Total number of process groups = " << num_oi_can_bp << std::endl;
    std::cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain
              << ", " << num_oi_can_bp << std::endl;
    std::cout << "No of processes used to compute each orbital = " << subranks << std::endl;
    // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (alpha)        //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
  ProcGroup        pg = ProcGroup::create_subgroups(gec.pg(), subranks);
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};
#else
  Scheduler&        sch = gsch;
  ExecutionContext& ec  = gec;
#endif

  AtomicCounter* ac = new AtomicCounterGA(gec.pg(), 1);
  ac->allocate(0);
  int64_t taskcount   = 0;
  int64_t next        = -1;
  int     total_pi_pg = 0;

  int root_ppi = ec.pg().rank().value();
  // for (size_t pib=0; pib < num_pi_remain; pib+=num_oi_can_bp){
  //   size_t piv_end = num_oi_can_bp;
  //   if(pib+num_oi_can_bp > num_pi_remain) piv_end = pib+num_oi_can_bp-num_pi_remain;
  // for (size_t piv=0; piv < piv_end; piv++){
  //   size_t pi = pi_tbp[piv+pib];
  int pg_id = rank.value() / subranks;
  if(root_ppi == 0) next = ac->fetch_add(0, 1);
  ec.pg().broadcast(&next, 0);

  for(size_t piv = 0; piv < pi_tbp.size(); piv++) {
    // #if GF_PGROUPS
    // if( (rank >= piv*subranks && rank < (piv*subranks+subranks) ) || no_pg){
    // if(!no_pg) root_ppi = piv*subranks; //root of sub-group
    // #endif
    if(next == taskcount) {
      total_pi_pg++;
      size_t pi = pi_tbp[piv];
      if(root_ppi == 0 && debug)
        std::cout << "Process group " << pg_id << " is executing orbital " << pi << std::endl;

      auto gf_t1 = std::chrono::high_resolution_clock::now();

      bool          gf_conv = false;
      ComplexTensor Minv{{O, O}, {1, 1}};
      ComplexTensor x1{O};
      Tensor<T>     B1{O};

      ComplexTensor Hx1_a{o_alpha};
      ComplexTensor Hx2_aaa{v_alpha, o_alpha, o_alpha};
      ComplexTensor Hx2_bab{v_beta, o_alpha, o_beta};
      ComplexTensor x1_a{o_alpha};
      ComplexTensor x2_aaa{v_alpha, o_alpha, o_alpha};
      ComplexTensor x2_bab{v_beta, o_alpha, o_beta};
      ComplexTensor dx1_a{o_alpha};
      ComplexTensor dx2_aaa{v_alpha, o_alpha, o_alpha};
      ComplexTensor dx2_bab{v_beta, o_alpha, o_beta};
      ComplexTensor Minv_a{o_alpha, o_alpha};
      Tensor<T>     B1_a{o_alpha};

      // if(rank==0) cout << "allocate B" << endl;
      sch.allocate(B1, B1_a).execute();
      LabelLoopNest loop_nest{B1().labels()};
      sch(B1() = 0).execute();

      for(const IndexVector& bid: loop_nest) {
        const IndexVector blockid = internal::translate_blockid(bid, B1());

        const TAMM_SIZE size = B1.block_size(blockid);
        std::vector<T>  buf(size);
        B1.get(blockid, buf);
        auto block_dims   = B1.block_dims(blockid);
        auto block_offset = B1.block_offsets(blockid);
        auto dim          = block_dims[0];
        auto offset       = block_offset[0];
        if(pi >= offset && pi < offset + dim) buf[pi - offset] = 1.0;
        B1.put(blockid, buf);
      }

      sch(B1_a(h1_oa) = B1(h1_oa)).deallocate(B1).execute();
      ec.pg().barrier();

      sch.allocate(Hx1_a, Hx2_aaa, Hx2_bab, dx1_a, dx2_aaa, dx2_bab, x1_a, x2_aaa, x2_bab)
        .execute();

      double gf_t_guess     = 0.0;
      double gf_t_x1_tot    = 0.0;
      double gf_t_x2_tot    = 0.0;
      double gf_t_res_tot   = 0.0;
      double gf_t_res_tot_1 = 0.0;
      double gf_t_res_tot_2 = 0.0;
      double gf_t_upd_tot   = 0.0;
      double gf_t_dis_tot   = 0.0;
      size_t gf_iter        = 0;

      // clang-format off
      sch
        .allocate(x1,Minv,Minv_a)
        (x1()      = 0)
        (Minv()    = 0)
        (Minv_a()  = 0)        
        .execute();
      // clang-format on

      gf_guess_ip(ec, MO, nocc, gf_omega, gf_eta, pi, p_evl_sorted_occ, t2v2_o, x1, Minv, true);

      // clang-format off
      sch
        (x1_a(h1_oa) = x1(h1_oa))
        (Minv_a(h1_oa,h2_oa)  = Minv(h1_oa,h2_oa))
        .deallocate(x1,Minv)
        .execute();
      // clang-format on

      std::string x1_a_inter_wpi_file =
        files_prefix + ".x1_a.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
      std::string x2_aaa_inter_wpi_file =
        files_prefix + ".x2_aaa.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
      std::string x2_bab_inter_wpi_file =
        files_prefix + ".x2_bab.inter.w" + gfo.str() + ".oi" + std::to_string(pi);

      if(fs::exists(x1_a_inter_wpi_file) && fs::exists(x2_aaa_inter_wpi_file) &&
         fs::exists(x2_bab_inter_wpi_file)) {
        read_from_disk(x1_a, x1_a_inter_wpi_file);
        read_from_disk(x2_aaa, x2_aaa_inter_wpi_file);
        read_from_disk(x2_bab, x2_bab_inter_wpi_file);
      }

      // GMRES
      ComplexTensor tmp{};
      sch.allocate(tmp).execute();

      do {
        int64_t gmres_hist = ccsd_options.gf_ngmres;

        gf_iter++;

        auto gf_gmres_0 = std::chrono::high_resolution_clock::now();

        ComplexTensor  r1_a{o_alpha};
        ComplexTensor  r2_aaa{v_alpha, o_alpha, o_alpha};
        ComplexTensor  r2_bab{v_beta, o_alpha, o_beta};
        VComplexTensor Q1_a;
        VComplexTensor Q2_aaa;
        VComplexTensor Q2_bab;

        GFCCSD_IP_A<T> gfccsd_ip_a;
        gfccsd_ip_a.gfccsd_x1_a(sch, MO, Hx1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_a, x2_aaa,
                                x2_bab, f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b, ix2_6_3_aaaa,
                                ix2_6_3_abab, unit_tis, false);

        gfccsd_ip_a.gfccsd_x2_a(
          sch, MO, Hx2_aaa, Hx2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_a, x2_aaa, x2_bab,
          f1, ix2_1_aaaa, ix2_1_abab, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab,
          ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_6_2_a, ix2_6_2_b,
          ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab, v2ijab_aaaa,
          v2ijab_abab, v2ijab_bbbb, unit_tis, false);

        // clang-format off
        sch
          .allocate(r1_a,  r2_aaa,  r2_bab)
          (dx1_a()   = -1.0 * Hx1_a())
          (dx2_aaa() = -1.0 * Hx2_aaa())
          (dx2_bab() = -1.0 * Hx2_bab())
          (dx1_a(h1_oa)               -= std::complex<double>(gf_omega,-1.0*gf_eta) * x1_a(h1_oa))
          (dx2_aaa(p1_va,h1_oa,h2_oa) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_aaa(p1_va,h1_oa,h2_oa))
          (dx2_bab(p1_vb,h1_oa,h2_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_bab(p1_vb,h1_oa,h2_ob))
          (dx1_a() += B1_a());
        // clang-format on

        // applying right preconditioning
        // clang-format off
        if (gf_preconditioning) {
          sch
            (r1_a(h1_oa) = dtmp_a(h1_oa) * dx1_a(h1_oa))
            (r2_aaa(p1_va,h1_oa,h2_oa) = dtmp_aaa(p1_va,h1_oa,h2_oa) * dx2_aaa(p1_va,h1_oa,h2_oa))
            (r2_bab(p1_vb,h1_oa,h2_ob) = dtmp_bab(p1_vb,h1_oa,h2_ob) * dx2_bab(p1_vb,h1_oa,h2_ob));
        } else {
          sch
            (r1_a() = 1.0 * dx1_a())
            (r2_aaa() = 1.0 * dx2_aaa())
            (r2_bab() = 1.0 * dx2_bab());
        }
        // clang-format on

        sch.execute(sch.ec().exhw());

        auto r1_a_norm   = norm(r1_a);
        auto r2_aaa_norm = norm(r2_aaa);
        auto r2_bab_norm = norm(r2_bab);

        auto gf_residual =
          sqrt(r1_a_norm * r1_a_norm + 0.5 * r2_aaa_norm * r2_aaa_norm + r2_bab_norm * r2_bab_norm);

        auto   gf_gmres = std::chrono::high_resolution_clock::now();
        double gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres - gf_gmres_0))
            .count();
        if(root_ppi == 0 && debug) {
          std::cout << "----------------" << std::endl;
          std::cout << "  #iter " << gf_iter << ", T(x_update contraction): " << std::fixed
                    << std::setprecision(6) << gftime << std::endl;
          std::cout << std::fixed << std::setprecision(2) << "  w,oi (" << gfo.str() << "," << pi
                    << "), residual = " << std::fixed << std::setprecision(6) << gf_residual
                    << std::endl;
        }

        if(std::abs(gf_residual) < gf_threshold || gf_iter > gf_maxiter) {
          sch.deallocate(r1_a, r2_aaa, r2_bab).execute();
          free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
          Q1_a.clear();
          Q2_aaa.clear();
          Q2_bab.clear();
          if(std::abs(gf_residual) < gf_threshold) gf_conv = true;
          break;
        }

        tamm::scale_ip(r1_a, 1.0 / gf_residual);
        tamm::scale_ip(r2_aaa, 1.0 / gf_residual);
        tamm::scale_ip(r2_bab, 1.0 / gf_residual);
        Q1_a.push_back(r1_a);
        Q2_aaa.push_back(r2_aaa);
        Q2_bab.push_back(r2_bab);
        CMatrix cn = CMatrix::Zero(gmres_hist, 1);
        CMatrix sn = CMatrix::Zero(gmres_hist, 1);
        CMatrix H  = CMatrix::Zero(gmres_hist + 1, gmres_hist);
        CMatrix b  = CMatrix::Zero(gmres_hist + 1, 1);
        b(0, 0)    = gf_residual;

        // GMRES inner loop
        int64_t k = 0;
        for(k = 0; k < gmres_hist; k++) {
          ComplexTensor q1_a{o_alpha};
          ComplexTensor q2_aaa{v_alpha, o_alpha, o_alpha};
          ComplexTensor q2_bab{v_beta, o_alpha, o_beta};

          auto gf_gmres_1 = std::chrono::high_resolution_clock::now();

          gfccsd_ip_a.gfccsd_x1_a(sch, MO, Hx1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k],
                                  Q2_aaa[k], Q2_bab[k], f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b,
                                  ix2_6_3_aaaa, ix2_6_3_abab, unit_tis, false);

          gfccsd_ip_a.gfccsd_x2_a(
            sch, MO, Hx2_aaa, Hx2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k], Q2_aaa[k],
            Q2_bab[k], f1, ix2_1_aaaa, ix2_1_abab, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa,
            ix2_4_abab, ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_6_2_a,
            ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab,
            v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, unit_tis, false);

          // clang-format off
          sch 
            .allocate(q1_a,q2_aaa,q2_bab)
            (dx1_a()    = 1.0 * Hx1_a())
            (dx2_aaa()  = 1.0 * Hx2_aaa())
            (dx2_bab()  = 1.0 * Hx2_bab())
            (dx1_a()   += std::complex<double>(gf_omega,-1.0*gf_eta) * Q1_a[k]())
            (dx2_aaa() += std::complex<double>(gf_omega,-1.0*gf_eta) * Q2_aaa[k]())
            (dx2_bab() += std::complex<double>(gf_omega,-1.0*gf_eta) * Q2_bab[k]());
          // clang-format on

          // clang-format off
          if (gf_preconditioning) {
            sch
              (q1_a(h1_oa) = dtmp_a(h1_oa) * dx1_a(h1_oa))
              (q2_aaa(p1_va,h1_oa,h2_oa) = dtmp_aaa(p1_va,h1_oa,h2_oa) * dx2_aaa(p1_va,h1_oa,h2_oa))
              (q2_bab(p1_vb,h1_oa,h2_ob) = dtmp_bab(p1_vb,h1_oa,h2_ob) * dx2_bab(p1_vb,h1_oa,h2_ob));
          } else {
            sch
              (q1_a() = 1.0 * dx1_a())
              (q2_aaa() = 1.0 * dx2_aaa())
              (q2_bab() = 1.0 * dx2_bab());
          }
          // clang-format on

          sch.execute(sch.ec().exhw());

          auto   gf_gmres_2 = std::chrono::high_resolution_clock::now();
          double gftime =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_2 - gf_gmres_1))
              .count();
          if(root_ppi == 0 && debug)
            std::cout << "    k: " << k << ", T(gfcc contraction): " << std::fixed
                      << std::setprecision(6) << gftime << std::endl;

          // Arnoldi iteration or G-S orthogonalization
          for(auto j = 0; j <= k; j++) {
            auto conj_a   = tamm::conj(Q1_a[j]);
            auto conj_aaa = tamm::conj(Q2_aaa[j]);
            auto conj_bab = tamm::conj(Q2_bab[j]);

            // clang-format off
            sch
              (tmp()  = 1.0 * conj_a(h1_oa) * q1_a(h1_oa))
              (tmp() += 0.5 * conj_aaa(p1_va,h1_oa,h2_oa) * q2_aaa(p1_va,h1_oa,h2_oa))
              (tmp() += 1.0 * conj_bab(p1_vb,h1_oa,h2_ob) * q2_bab(p1_vb,h1_oa,h2_ob))
              (q1_a()   -= tmp() * Q1_a[j]())
              (q2_aaa() -= tmp() * Q2_aaa[j]())
              (q2_bab() -= tmp() * Q2_bab[j]());
              // .deallocate(conj_a,conj_aaa,conj_bab);
              // .execute();
            // clang-format on

            sch.execute(sch.ec().exhw());

            H(j, k) = get_scalar(tmp);

            // re-orthogonalization

            // clang-format off
            sch
              (tmp()  = 1.0 * conj_a(h1_oa) * q1_a(h1_oa))
              (tmp() += 0.5 * conj_aaa(p1_va,h1_oa,h2_oa) * q2_aaa(p1_va,h1_oa,h2_oa))
              (tmp() += 1.0 * conj_bab(p1_vb,h1_oa,h2_ob) * q2_bab(p1_vb,h1_oa,h2_ob))
              (q1_a()   -= tmp() * Q1_a[j]())
              (q2_aaa() -= tmp() * Q2_aaa[j]())
              (q2_bab() -= tmp() * Q2_bab[j]())
              .deallocate(conj_a,conj_aaa,conj_bab);
            // .execute();
            // clang-format on

            sch.execute(sch.ec().exhw());

            H(j, k) += get_scalar(tmp);
          } // j loop

          r1_a_norm   = norm(q1_a);
          r2_aaa_norm = norm(q2_aaa);
          r2_bab_norm = norm(q2_bab);

          H(k + 1, k) = sqrt(r1_a_norm * r1_a_norm + 0.5 * r2_aaa_norm * r2_aaa_norm +
                             r2_bab_norm * r2_bab_norm);

          // if(std::abs(H(k+1,k))<1e-16) {
          //   gmres_hist = k+1;
          //   sch.deallocate(q1_a,q2_aaa,q2_bab).execute();
          //   break;
          // }
          std::complex<double> scaling = 1.0 / H(k + 1, k);

          auto gf_gmres_3 = std::chrono::high_resolution_clock::now();
          gftime =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_3 - gf_gmres_2))
              .count();
          if(root_ppi == 0 && debug)
            std::cout << "    k: " << k << ", T(Arnoldi): " << gftime << std::endl;

          CMatrix Hsub = H.block(0, 0, k + 2, k + 1);
          CMatrix bsub = b.block(0, 0, k + 2, 1);

          // apply givens rotation for complex tensors (IMPORTANT: this is different from the real
          // case)
          for(auto i = 0; i < k; i++) {
            auto temp   = cn(i, 0) * H(i, k) + sn(i, 0) * H(i + 1, k);
            H(i + 1, k) = -std::conj(sn(i, 0)) * H(i, k) + cn(i, 0) * H(i + 1, k);
            H(i, k)     = temp;
          }

          std::complex<double> scr1 = H(k, k);
          std::complex<double> scr2 = H(k + 1, k);
          // std::complex<double> s_scr  = std::complex<double>(0.0, 0.0);
          T cnk0_r = cn(k, 0).real();
          blas::rotg(&scr1, &scr2, &cnk0_r, &sn(k, 0));
          cn(k, 0) = std::complex<T>(cnk0_r, cn(k, 0).imag());

          // if(root_ppi==0 && debug) {
          //   cout << "cn/sn from rotg: " << cn(k,0) << "," << sn(k,0) << endl;
          //   auto t = sqrt(std::conj(H(k,k))*H(k,k)+std::conj(H(k+1,k))*H(k+1,k));
          //   auto cn_t = std::abs(H(k,k))/t;
          //   auto sn_t = H(k,k)*std::conj(H(k+1,k))/(std::abs(H(k,k))*t);
          //   cout << "cn/sn from self-computing: " << cn_t << "," << sn_t << endl;
          // }

          if(root_ppi == 0 && debug) {
            // cout << "cn/sn from self-computing: " << cn(k,0) << "," << sn(k,0) << endl;
            std::cout << "    k: " << k << ", Checking if H(k+1,k) is zero: " << std::fixed
                      << std::setprecision(6)
                      << -std::conj(sn(k, 0)) * H(k, k) + cn(k, 0) * H(k + 1, k)
                      << std::endl; //"," << H(k,k) << "," << H(k+1,k) << endl;
          }

          H(k, k)     = cn(k, 0) * H(k, k) + sn(k, 0) * H(k + 1, k);
          H(k + 1, k) = std::complex<double>(0, 0);

          b(k + 1, 0) = -std::conj(sn(k, 0)) * b(k, 0);
          b(k, 0)     = cn(k, 0) * b(k, 0);

          auto gf_gmres_4 = std::chrono::high_resolution_clock::now();
          gftime =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_4 - gf_gmres_3))
              .count();
          if(root_ppi == 0 && debug) {
            std::cout << "    k: " << k << ", T(Givens rotation): " << gftime
                      << ", error: " << std::abs(b(k + 1, 0)) << std::endl;
            std::cout << "    ----------" << std::endl;
          } // if(root_ppi==0&&debug) cout<< "k: " << k << ", error: " << std::abs(b(k+1,0)) <<
            // endl;

          // normalization
          tamm::scale_ip(q1_a, scaling);
          tamm::scale_ip(q2_aaa, scaling);
          tamm::scale_ip(q2_bab, scaling);
          Q1_a.push_back(q1_a);
          Q2_aaa.push_back(q2_aaa);
          Q2_bab.push_back(q2_bab);

          if(std::abs(b(k + 1, 0)) < gf_threshold) {
            gmres_hist = k + 1;
            break;
          }
        } // k loop

        auto gf_gmres_5 = std::chrono::high_resolution_clock::now();
        gftime = std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_5 - gf_gmres))
                   .count();
        if(root_ppi == 0 && debug)
          std::cout << "  pi: " << pi << ", k: " << k << ", gmres_hist: " << gmres_hist
                    << ", #iter: " << gf_iter << ", T(micro_tot): " << gftime << std::endl;

        // solve a least square problem in the subspace
        CMatrix Hsub = H.block(0, 0, gmres_hist, gmres_hist);
        CMatrix bsub = b.block(0, 0, gmres_hist, 1);
        CMatrix y    = Hsub.householderQr().solve(bsub);

        // if(rank==0 && debug) cout << "residual: " << (bsub-Hsub*y).norm() << endl;

        for(auto i = 0; i < gmres_hist; i++) {
          // clang-format off
          sch
            (x1_a(h1_oa)               += y(i,0) * Q1_a[i](h1_oa))
            (x2_aaa(p1_va,h1_oa,h2_oa) += y(i,0) * Q2_aaa[i](p1_va,h1_oa,h2_oa))
            (x2_bab(p1_vb,h1_oa,h2_ob) += y(i,0) * Q2_bab[i](p1_vb,h1_oa,h2_ob));
          // clang-format on
        }
        sch.execute();

        write_to_disk(x1_a, x1_a_inter_wpi_file);
        write_to_disk(x2_aaa, x2_aaa_inter_wpi_file);
        write_to_disk(x2_bab, x2_bab_inter_wpi_file);

        free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
        Q1_a.clear();
        Q2_aaa.clear();
        Q2_bab.clear();

        // auto gf_gmres_6 = std::chrono::high_resolution_clock::now();
        // gftime =
        //   std::chrono::duration_cast<std::chrono::duration<double>>((gf_gmres_6 -
        //   gf_gmres_5)).count();
        // if(root_ppi==0 && debug) cout << "  #iter " << gf_iter << ",
        // T(least_square+X_updat+misc.): " << gftime << endl;

      } while(true);

      // deallocate memory
      sch.deallocate(tmp).execute();

      if(gf_conv) {
        std::string x1_a_conv_wpi_file =
          files_prefix + ".x1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string x2_aaa_conv_wpi_file =
          files_prefix + ".x2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string x2_bab_conv_wpi_file =
          files_prefix + ".x2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);
        write_to_disk(x1_a, x1_a_conv_wpi_file);
        write_to_disk(x2_aaa, x2_aaa_conv_wpi_file);
        write_to_disk(x2_bab, x2_bab_conv_wpi_file);
        fs::remove(x1_a_inter_wpi_file);
        fs::remove(x2_aaa_inter_wpi_file);
        fs::remove(x2_bab_inter_wpi_file);
      }

      if(!gf_conv) { //&& root_ppi == 0
        std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
        tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
      }

      auto   gf_t2 = std::chrono::high_resolution_clock::now();
      double gftime =
        std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
      if(root_ppi == 0) {
        std::string gf_stats;
        gf_stats = gfacc_str("R-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                             ") = ", std::to_string(gftime),
                             " secs, #iter = ", std::to_string(gf_iter), ", using PG ",
                             std::to_string(pg_id));

        if(debug) {
          gf_stats += gfacc_str("|----------initial guess  : ", std::to_string(gf_t_guess));
          gf_stats += gfacc_str("|----------x1 contraction : ", std::to_string(gf_t_x1_tot));
          gf_stats += gfacc_str("|----------x2 contraction : ", std::to_string(gf_t_x2_tot));
          gf_stats += gfacc_str("|----------computing res. : ", std::to_string(gf_t_res_tot));
          gf_stats +=
            gfacc_str("           |----------misc. contr. : ", std::to_string(gf_t_res_tot_1));
          gf_stats +=
            gfacc_str("           |----------compt. norm  : ", std::to_string(gf_t_res_tot_2));
          gf_stats += gfacc_str("|----------updating x1/x2 : ", std::to_string(gf_t_upd_tot));
          gf_stats += gfacc_str("|----------diis update    : ", std::to_string(gf_t_dis_tot));
        }
        std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
      }

      sch
        .deallocate(Hx1_a, Hx2_aaa, Hx2_bab, dx1_a, dx2_aaa, dx2_bab, Minv_a, x1_a, x2_aaa, x2_bab,
                    B1_a)
        .execute();

      if(root_ppi == 0) next = ac->fetch_add(0, 1);
      ec.pg().broadcast(&next, 0);
    }
    // #if GF_PGROUPS
    // }
    // #endif
    if(root_ppi == 0) taskcount++;
    ec.pg().broadcast(&taskcount, 0);
    // ec.pg().barrier();
  } // end all remaining pi

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(root_ppi == 0) {
    std::cout << "Total orbitals executed by process group " << pg_id << " = " << total_pi_pg
              << std::endl;
    std::cout << "  --> Total R-GF-CCSD Time = " << time << " secs" << std::endl;
  }

#if GF_PGROUPS
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  pg.destroy_coll();
#endif
  ac->deallocate();
  delete ac;
  gec.pg().barrier();

  cc_t2 = std::chrono::high_resolution_clock::now();
  time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total R-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  gsch.deallocate(dtmp_a, dtmp_aaa, dtmp_bab).execute();
}

// Explicit template instantiation
template class GFCCSD_IP_A_Driver<double>;

} // namespace exachem::cc::gfcc