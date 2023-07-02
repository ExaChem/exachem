/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "gf_diis.hpp"
#include "gf_guess.hpp"
#include "gfccsd_ea.hpp"
#include "gfccsd_ip.hpp"

#include "cc/ccsd_lambda.hpp"
#include <algorithm>

using namespace tamm;

#include <filesystem>
namespace fs = std::filesystem;

// TODO input file
size_t ndiis;
size_t ngmres;
size_t gf_maxiter;

int                 gf_nprocs_poi;
bool                gf_profile;
double              gf_omega;
size_t              p_oi; // number of occupied/all MOs
double              gf_eta;
double              gf_lshift;
double              gf_threshold;
bool                gf_preconditioning;
double              omega_min_ip;
double              omega_max_ip;
double              lomega_min_ip;
double              lomega_max_ip;
double              omega_min_ea;
double              omega_max_ea;
double              lomega_min_ea;
double              lomega_max_ea;
double              omega_delta;
int64_t             omega_npts_ip;
int64_t             lomega_npts_ip;
int64_t             omega_npts_ea;
int64_t             lomega_npts_ea;
double              omega_delta_e;
double              gf_damping_factor;
int                 gf_extrapolate_level;
int                 gf_analyze_level;
int                 gf_analyze_num_omega;
std::vector<size_t> gf_orbitals;
std::vector<double> gf_analyze_omega;

#define GF_PGROUPS 1
#define GF_IN_SG 0
#define GF_GS_SG 0

// Tensor<double> lambda_y1, lambda_y2,
// Tensor<double> d_t1_a, d_t1_b,
//                d_t2_aaaa, d_t2_bbbb, d_t2_abab,
//                v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb,
//                cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a, cholVV_b;

Tensor<double> t2v2_o, lt12_o_a, lt12_o_b, ix1_1_1_a, ix1_1_1_b, ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb,
  ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, ix2_5_aaaa,
  ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_5_baba, ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa,
  ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba, t2v2_v, lt12_v_a, lt12_v_b,
  iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy1_a, iy1_b, iy2_a, iy2_b, iy3_1_aaaa, iy3_1_abba,
  iy3_1_baba, iy3_1_bbbb, iy3_1_baab, iy3_1_abab, iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_baab,
  iy3_abba, iy3_bbbb, iy3_baba, iy3_abab, iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb,
  iy4_1_abba, iy4_1_abab, iy4_2_aaaa, iy4_2_abba, iy4_2_bbbb, iy4_2_baab, iy5_aaaa, iy5_baab,
  iy5_baba, iy5_abba, iy5_bbbb, iy5_abab, iy6_a, iy6_b;

TiledIndexSpace diis_tis;
// std::ofstream ofs_profile;

template<typename T>
T find_closest(T w, std::vector<T>& wlist) {
  double diff = std::abs(wlist[0] - w);
  int    idx  = 0;
  for(size_t c = 1; c < wlist.size(); c++) {
    double cdiff = std::abs(wlist[c] - w);
    if(cdiff < diff) {
      idx  = c;
      diff = cdiff;
    }
  }

  return wlist[idx];
}

template<typename... Ts>
std::string gfacc_str(Ts&&... args) {
  std::string res;
  (res.append(args), ...);
  res.append("\n");
  return res;
}

void write_results_to_json(ExecutionContext& ec, SystemData& sys_data, int level,
                           std::vector<double>& ni_w, std::vector<double>& ni_A,
                           std::string gfcc_type) {
  auto lomega_npts = ni_w.size();
  // std::vector<double> r_ni_w;
  // std::vector<double> r_ni_A;
  // if(ec.pg().rank()==0) {
  //   r_ni_w.resize(lomega_npts,0);
  //   r_ni_A.resize(lomega_npts,0);
  // }
  // ec.pg().reduce(ni_w.data(), r_ni_w.data(), lomega_npts, ReduceOp::sum, 0);
  // ec.pg().reduce(ni_A.data(), r_ni_A.data(), lomega_npts, ReduceOp::sum, 0);

  if(ec.pg().rank() == 0) {
    const std::string lvl_str = "level" + std::to_string(level);
    sys_data.results["output"]["GFCCSD"][gfcc_type][lvl_str]["omega_npts"] = lomega_npts;
    for(size_t ni = 0; ni < lomega_npts; ni++) {
      sys_data.results["output"]["GFCCSD"][gfcc_type][lvl_str][std::to_string(ni)]["omega"] =
        ni_w[ni];
      sys_data.results["output"]["GFCCSD"][gfcc_type][lvl_str][std::to_string(ni)]["A_a"] =
        ni_A[ni];
    }
    write_json_data(sys_data, "GFCCSD");
  }
}

void write_string_to_disk(ExecutionContext& ec, const std::string& tstring,
                          const std::string& filename) {
  int  tstring_len = tstring.length();
  auto rank        = ec.pg().rank().value();

  std::vector<int> recvcounts;
  auto             size = ec.pg().size().value();

  if(rank == 0) recvcounts.resize(size, 0);

  ec.pg().gather(&tstring_len, recvcounts.data(), 0);

  /*
   * Figure out the total length of string,
   * and displacements for each rank
   */

  int              totlen = 0;
  std::vector<int> displs;
  char*            combined_string = nullptr;

  if(rank == 0) {
    displs.resize(size, 0);

    displs[0] = 0;
    totlen += recvcounts[0] + 1;

    for(int i = 1; i < size; i++) {
      totlen += recvcounts[i] + 1; /* plus one for space or \0 after words */
      displs[i] = displs[i - 1] + recvcounts[i - 1] + 1;
    }

    /* allocate string, pre-fill with spaces and null terminator */
    combined_string = new char[totlen];
    for(int i = 0; i < totlen - 1; i++) combined_string[i] = ' ';
    combined_string[totlen - 1] = '\0';
  }

  // Gather strings from all ranks in pg
  ec.pg().gatherv(tstring.c_str(), tstring_len, combined_string, &recvcounts[0], &displs[0], 0);

  if(rank == 0) {
    // cout << combined_string << endl;
    std::ofstream out(filename, std::ios::out);
    if(!out) cerr << "Error opening file " << filename << endl;
    out << combined_string << std::endl;
    out.close();
    delete combined_string;
  }
}

template<typename T>
void gfccsd_driver_ip_a(
  ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm& subcomm, const TiledIndexSpace& MO,
  Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
  Tensor<T>& f1, Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
  Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
  Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
  Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
  Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
  Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
  Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
  Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba, Tensor<T>& v2ijab_aaaa,
  Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, std::vector<T>& p_evl_sorted_occ,
  std::vector<T>& p_evl_sorted_virt, long int total_orbitals, const TAMM_SIZE nocc,
  const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& unit_tis, string files_prefix,
  string levelstr, int noa) {
  using ComplexTensor  = Tensor<std::complex<T>>;
  using VComplexTensor = std::vector<Tensor<std::complex<T>>>;
  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(obtiles, otiles)};
  v_beta  = {MO("virt"), range(vbtiles, vtiles)};

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
    if(subcomm != MPI_COMM_NULL) {
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
  auto nranks     = gec.pg().size().value();
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int       world_size;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Comm_size(world_comm, &world_size);
  MPI_Comm gf_comm;

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
  // if(num_pi_remain == 0) num_pi_remain = 1;
  int        subranks = std::floor(nranks / num_pi_remain);
  const bool no_pg    = (subranks == 0 || subranks == 1);
  if(no_pg) subranks = nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  // Figure out how many orbitals in pi_tbp can be processed with subranks
  // TODO: gf_nprocs_pi must be a multiple of total #ranks for best performance.
  size_t num_oi_can_bp = std::ceil(nranks / (1.0 * subranks));
  if(num_pi_remain < num_oi_can_bp) {
    num_oi_can_bp = num_pi_remain;
    subranks      = std::floor(nranks / num_pi_remain);
    if(no_pg) subranks = nranks;
  }

  if(rank == 0) {
    cout << "Total number of process groups = " << num_oi_can_bp << endl;
    cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
         << num_oi_can_bp << endl;
    cout << "No of processes used to compute each orbital = " << subranks << endl;
    // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  int color = 0;
  if(subranks > 1) color = world_rank / subranks;

  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (alpha)        //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
  ProcGroup        pg = ProcGroup::create_coll(gf_comm);
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

  int root_ppi = -1;
  MPI_Comm_rank(ec.pg().comm(), &root_ppi);
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
        cout << "Process group " << pg_id << " is executing orbital " << pi << endl;

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

        const TAMM_SIZE         size = B1.block_size(blockid);
        std::vector<TensorType> buf(size);
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
        int64_t gmres_hist = ngmres;

        gf_iter++;

        auto gf_gmres_0 = std::chrono::high_resolution_clock::now();

        ComplexTensor  r1_a{o_alpha};
        ComplexTensor  r2_aaa{v_alpha, o_alpha, o_alpha};
        ComplexTensor  r2_bab{v_beta, o_alpha, o_beta};
        VComplexTensor Q1_a;
        VComplexTensor Q2_aaa;
        VComplexTensor Q2_bab;

        gfccsd_x1_a(sch, MO, Hx1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_a, x2_aaa, x2_bab, f1,
                    ix2_2_a, ix1_1_1_a, ix1_1_1_b, ix2_6_3_aaaa, ix2_6_3_abab, unit_tis, false);

        gfccsd_x2_a(sch, MO, Hx2_aaa, Hx2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_a, x2_aaa,
                    x2_bab, f1, ix2_1_aaaa, ix2_1_abab, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b,
                    ix2_4_aaaa, ix2_4_abab, ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb,
                    ix2_5_baab, ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                    ix2_6_3_bbbb, ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, unit_tis,
                    false);

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
          cout << "----------------" << endl;
          cout << "  #iter " << gf_iter << ", T(x_update contraction): " << std::fixed
               << std::setprecision(6) << gftime << endl;
          cout << std::fixed << std::setprecision(2) << "  w,oi (" << gfo.str() << "," << pi
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

          gfccsd_x1_a(sch, MO, Hx1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k], Q2_aaa[k],
                      Q2_bab[k], f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b, ix2_6_3_aaaa, ix2_6_3_abab,
                      unit_tis, false);

          gfccsd_x2_a(sch, MO, Hx2_aaa, Hx2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k],
                      Q2_aaa[k], Q2_bab[k], f1, ix2_1_aaaa, ix2_1_abab, ix2_2_a, ix2_2_b, ix2_3_a,
                      ix2_3_b, ix2_4_aaaa, ix2_4_abab, ix2_5_aaaa, ix2_5_abba, ix2_5_abab,
                      ix2_5_bbbb, ix2_5_baab, ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba,
                      ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab,
                      v2ijab_bbbb, unit_tis, false);

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
            cout << "    k: " << k << ", T(gfcc contraction): " << std::fixed
                 << std::setprecision(6) << gftime << endl;

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
          if(root_ppi == 0 && debug) cout << "    k: " << k << ", T(Arnoldi): " << gftime << endl;

          CMatrix Hsub = H.block(0, 0, k + 2, k + 1);
          CMatrix bsub = b.block(0, 0, k + 2, 1);

          // apply givens rotation for complex tensors (IMPORTANT: this is different from the real
          // case)
          for(auto i = 0; i < k; i++) {
            auto temp   = cn(i, 0) * H(i, k) + sn(i, 0) * H(i + 1, k);
            H(i + 1, k) = -std::conj(sn(i, 0)) * H(i, k) + cn(i, 0) * H(i + 1, k);
            H(i, k)     = temp;
          }

          std::complex<double> scr1   = H(k, k);
          std::complex<double> scr2   = H(k + 1, k);
          std::complex<double> s_scr  = std::complex<double>(0.0, 0.0);
          T                    cnk0_r = cn(k, 0).real();
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
            cout << "    k: " << k << ", Checking if H(k+1,k) is zero: " << std::fixed
                 << std::setprecision(6) << -std::conj(sn(k, 0)) * H(k, k) + cn(k, 0) * H(k + 1, k)
                 << endl; //"," << H(k,k) << "," << H(k+1,k) << endl;
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
            cout << "    k: " << k << ", T(Givens rotation): " << gftime
                 << ", error: " << std::abs(b(k + 1, 0)) << endl;
            cout << "    ----------" << endl;
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
          cout << "  pi: " << pi << ", k: " << k << ", gmres_hist: " << gmres_hist
               << ", #iter: " << gf_iter << ", T(micro_tot): " << gftime << endl;

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
    cout << "Total orbitals executed by process group " << pg_id << " = " << total_pi_pg << endl;
    cout << "  --> Total R-GF-CCSD Time = " << time << " secs" << endl;
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
  MPI_Comm_free(&gf_comm);
}

#if 0
template<typename T>
void gfccsd_driver_ip_b(
  ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm& subcomm, const TiledIndexSpace& MO,
  Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
  Tensor<T>& f1, Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
  Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
  Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
  Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
  Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
  Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
  Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
  Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba, Tensor<T>& v2ijab_aaaa,
  Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, std::vector<T>& p_evl_sorted_occ,
  std::vector<T>& p_evl_sorted_virt, long int total_orbitals, const TAMM_SIZE nocc,
  const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& unit_tis, string files_prefix,
  string levelstr, int noa, int nob) {
  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(obtiles, otiles)};
  v_beta  = {MO("virt"), range(vbtiles, vtiles)};

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
    gfp << std::endl
        << std::string(55, '-') << std::endl
        << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_bbb{v_beta, o_beta, o_beta};
  ComplexTensor dtmp_aba{v_alpha, o_beta, o_alpha};
  ComplexTensor::allocate(&gec, dtmp_bbb, dtmp_aba);

  // double au2ev = 27.2113961;

  std::string dtmp_bbb_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_bbb.l" + levelstr;
  std::string dtmp_aba_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_aba.l" + levelstr;

  if(fs::exists(dtmp_bbb_file) && fs::exists(dtmp_aba_file)) {
    read_from_disk(dtmp_bbb, dtmp_bbb_file);
    read_from_disk(dtmp_aba, dtmp_aba_file);
  }
  else {
    ComplexTensor DEArr_IP{V, O, O};

    double       denominator  = 0.0;
    const double lshift       = 1.00000000;
    auto         DEArr_lambda = [&](const IndexVector& bid) {
      const IndexVector            blockid = internal::translate_blockid(bid, DEArr_IP());
      const TAMM_SIZE              size    = DEArr_IP.block_size(blockid);
      std::vector<std::complex<T>> buf(size);

      auto   block_dims   = DEArr_IP.block_dims(blockid);
      auto   block_offset = DEArr_IP.block_offsets(blockid);
      size_t c            = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            denominator =
              gf_omega + p_evl_sorted_virt[i] - p_evl_sorted_occ[j] - p_evl_sorted_occ[k];
            if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * lshift; }
            else if(denominator > 0.0 && denominator < 1.0) { denominator += lshift; }
            buf[c] = 1.0 / std::complex<T>(denominator, -1.0 * gf_eta);
            // if(i==0&&j==0) {
            //   cout << " omega,k,Eocc,denom_inv,buf : " << gf_omega << "," << k << "," <<
            //   p_evl_sorted_occ[k] << "," << denominator << "," << buf[c] << endl;
            // }
          }
        }
      }
      DEArr_IP.put(blockid, buf);
    };

    gsch.allocate(DEArr_IP).execute();
    if(subcomm != MPI_COMM_NULL) {
      Scheduler sub_sch{sub_ec};
      sub_sch(DEArr_IP() = 0).execute();
      block_for(sub_ec, DEArr_IP(), DEArr_lambda);
      sub_sch(dtmp_bbb() = 0)(dtmp_aba() = 0)(dtmp_bbb(p1_vb, h1_ob, h2_ob) =
                                                DEArr_IP(p1_vb, h1_ob, h2_ob))(
        dtmp_aba(p1_va, h1_ob, h2_oa) = DEArr_IP(p1_va, h1_ob, h2_oa))
        .execute();
    }
    gec.pg().barrier();
    gsch.deallocate(DEArr_IP).execute();

    write_to_disk(dtmp_bbb, dtmp_bbb_file);
    write_to_disk(dtmp_aba, dtmp_aba_file);
  }

  //------------------------
  auto nranks     = GA_Nnodes();
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int       world_size;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Comm_size(world_comm, &world_size);
  MPI_Comm gf_comm;

  const size_t        num_oi           = nob;
  size_t              num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  // Check pi's already processed
  for(size_t pi = noa; pi < noa + num_oi; pi++) {
    std::string x1_b_wpi_file = files_prefix + ".x1_b.W" + gfo.str() + ".oi" + std::to_string(pi);
    std::string x2_bbb_wpi_file =
      files_prefix + ".x2_bbb.W" + gfo.str() + ".oi" + std::to_string(pi);
    std::string x2_aba_wpi_file =
      files_prefix + ".x2_aba.W" + gfo.str() + ".oi" + std::to_string(pi);

    if(fs::exists(x1_b_wpi_file) && fs::exists(x2_bbb_wpi_file) && fs::exists(x2_aba_wpi_file))
      num_pi_processed++;
    else pi_tbp.push_back(pi);
  }

  size_t num_pi_remain = num_oi - num_pi_processed;
  if(num_pi_remain == 0) {
    gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
    return;
  }
  EXPECTS(num_pi_remain == pi_tbp.size());
  // if(num_pi_remain == 0) num_pi_remain = 1;
  int        subranks = std::floor(nranks / num_pi_remain);
  const bool no_pg    = (subranks == 0 || subranks == 1);
  if(no_pg) subranks = nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  // Figure out how many orbitals in pi_tbp can be processed with subranks
  // TODO: gf_nprocs_pi must be a multiple of total #ranks for best performance.
  size_t num_oi_can_bp = std::ceil(nranks / (1.0 * subranks));
  if(num_pi_remain < num_oi_can_bp) {
    num_oi_can_bp = num_pi_remain;
    subranks      = std::floor(nranks / num_pi_remain);
    if(no_pg) subranks = nranks;
  }

  if(rank == 0) {
    cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
         << num_oi_can_bp << endl;
    cout << "No of processes used to compute each orbital = " << subranks << endl;
    // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  int color = 0;
  if(subranks > 1) color = world_rank / subranks;

  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (beta)         //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
  ProcGroup        pg = ProcGroup::create_coll(gf_comm);
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};
#else
  Scheduler&        sch = gsch;
  ExecutionContext& ec  = gec;
#endif

  int64_t root_ppi = 0;
  for(size_t pib = 0; pib < num_pi_remain; pib += num_oi_can_bp) {
    size_t piv_end = num_oi_can_bp;
    if(pib + num_oi_can_bp > num_pi_remain) piv_end = pib + num_oi_can_bp - num_pi_remain;
    for(size_t piv = 0; piv < piv_end; piv++) {
      size_t pi = pi_tbp[piv + pib];
#if GF_PGROUPS
      if((rank >= piv * subranks && rank < (piv * subranks + subranks)) || no_pg) {
        if(!no_pg) root_ppi = piv * subranks; // root of sub-group
#endif

        auto gf_t1 = std::chrono::high_resolution_clock::now();

        bool          gf_conv = false;
        ComplexTensor Minv{{O, O}, {1, 1}};
        ComplexTensor x1{O};
        Tensor<T>     B1{O};

        ComplexTensor Hx1_b{o_beta};
        ComplexTensor Hx2_bbb{v_beta, o_beta, o_beta};
        ComplexTensor Hx2_aba{v_alpha, o_beta, o_alpha};
        ComplexTensor d1_b{o_beta};
        ComplexTensor d2_bbb{v_beta, o_beta, o_beta};
        ComplexTensor d2_aba{v_alpha, o_beta, o_alpha};
        ComplexTensor Minv_b{o_beta, o_beta};
        ComplexTensor dx1_b{o_beta};
        ComplexTensor dx2_bbb{v_beta, o_beta, o_beta};
        ComplexTensor dx2_aba{v_alpha, o_beta, o_alpha};
        ComplexTensor x1_b{o_beta};
        ComplexTensor x2_bbb{v_beta, o_beta, o_beta};
        ComplexTensor x2_aba{v_alpha, o_beta, o_alpha};
        Tensor<T>     B1_b{o_beta};

        sch
          .allocate(Hx1_b, Hx2_bbb, Hx2_aba, d1_b, d2_bbb, d2_aba, dx1_b, dx2_bbb, dx2_aba, x1,
                    x1_b, x2_bbb, x2_aba, Minv, Minv_b, B1, B1_b)
          .execute();

        int    total_iter     = 0;
        double gf_t_guess     = 0.0;
        double gf_t_x1_tot    = 0.0;
        double gf_t_x2_tot    = 0.0;
        double gf_t_res_tot   = 0.0;
        double gf_t_res_tot_1 = 0.0;
        double gf_t_res_tot_2 = 0.0;
        double gf_t_upd_tot   = 0.0;
        double gf_t_dis_tot   = 0.0;

        std::string x1_b_wpi_file =
          files_prefix + ".x1_b.W" + gfo.str() + ".oi" + std::to_string(pi);
        std::string x2_bbb_wpi_file =
          files_prefix + ".x2_bbb.W" + gfo.str() + ".oi" + std::to_string(pi);
        std::string x2_aba_wpi_file =
          files_prefix + ".x2_aba.W" + gfo.str() + ".oi" + std::to_string(pi);

        if(!(fs::exists(x1_b_wpi_file) && fs::exists(x2_bbb_wpi_file) &&
             fs::exists(x2_aba_wpi_file))) {
          ComplexTensor e1_b{o_beta, diis_tis};
          ComplexTensor e2_bbb{v_beta, o_beta, o_beta, diis_tis};
          ComplexTensor e2_aba{v_alpha, o_beta, o_alpha, diis_tis};
          ComplexTensor xx1_b{o_beta, diis_tis};
          ComplexTensor xx2_bbb{v_beta, o_beta, o_beta, diis_tis};
          ComplexTensor xx2_aba{v_alpha, o_beta, o_alpha, diis_tis};

          // clang-format off
          sch
            .allocate( e1_b, e2_bbb, e2_aba,
                      xx1_b,xx2_bbb,xx2_aba)
            (x1()      = 0)
            (Minv()    = 0)
            (Hx1_b()   = 0)
            (Hx2_bbb() = 0)
            (Hx2_aba() = 0)
            (d1_b()    = 0)
            (d2_bbb()  = 0)
            (d2_aba()  = 0)
            (x1_b()    = 0)
            (x2_bbb()  = 0)
            (x2_aba()  = 0)
            (Minv_b()  = 0)
            .execute();
          // clang-format on

          LabelLoopNest loop_nest{B1().labels()};
          sch(B1() = 0).execute();

          for(const IndexVector& bid: loop_nest) {
            const IndexVector blockid = internal::translate_blockid(bid, B1());

            const TAMM_SIZE         size = B1.block_size(blockid);
            std::vector<TensorType> buf(size);
            B1.get(blockid, buf);
            auto block_dims   = B1.block_dims(blockid);
            auto block_offset = B1.block_offsets(blockid);
            auto dim          = block_dims[0];
            auto offset       = block_offset[0];
            if(pi >= offset && pi < offset + dim) buf[pi - offset] = 1.0;
            B1.put(blockid, buf);
          }

          sch(B1_b(h1_ob) = B1(h1_ob)).execute();
          ec.pg().barrier();

          auto gf_t_guess1 = std::chrono::high_resolution_clock::now();

          gf_guess_ip(ec, MO, nocc, gf_omega, gf_eta, pi, p_evl_sorted_occ, t2v2_o, x1, Minv, true);

          // clang-format off
          sch
            (x1_b(h1_ob) = x1(h1_ob))
            (Minv_b(h1_ob,h2_ob)  = Minv(h1_ob,h2_ob))
            .execute();
          // clang-format on

          auto gf_t_guess2 = std::chrono::high_resolution_clock::now();
          gf_t_guess =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_guess2 - gf_t_guess1))
              .count();

          for(size_t iter = 0; iter < gf_maxiter; iter += ndiis) {
            for(size_t micro = iter; micro < std::min(iter + ndiis, gf_maxiter); micro++) {
              total_iter    = micro;
              auto gf_t_ini = std::chrono::high_resolution_clock::now();

              gfccsd_x1_b(sch, MO, Hx1_b, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_b, x2_bbb,
                          x2_aba, f1, ix2_2_b, ix1_1_1_a, ix1_1_1_b, ix2_6_3_bbbb, ix2_6_3_baba,
                          unit_tis, false);

              auto gf_t_x1 = std::chrono::high_resolution_clock::now();
              gf_t_x1_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x1 - gf_t_ini))
                  .count();

              gfccsd_x2_b(sch, MO, Hx2_bbb, Hx2_aba, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_b,
                          x2_bbb, x2_aba, f1, ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a,
                          ix2_3_b, ix2_4_bbbb, ix2_4_abab, ix2_5_aaaa, ix2_5_abba, ix2_5_baba,
                          ix2_5_bbbb, ix2_5_baab, ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba,
                          ix2_6_3_baba, ix2_6_3_bbbb, ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab,
                          v2ijab_bbbb, unit_tis, false);

              sch.execute(); // TODO: not needed if not profiling
              auto gf_t_x2 = std::chrono::high_resolution_clock::now();
              gf_t_x2_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x2 - gf_t_x1))
                  .count();

              // clang-format off
              sch
                (d1_b() = -1.0 * Hx1_b())
                (d1_b(h1_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x1_b(h1_ob))
                (d1_b() += B1_b())
                (d2_bbb() = -1.0 * Hx2_bbb())
                (d2_bbb(p1_vb,h1_ob,h2_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_bbb(p1_vb,h1_ob,h2_ob))
                (d2_aba() = -1.0 * Hx2_aba())
                (d2_aba(p1_va,h1_ob,h2_oa) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_aba(p1_va,h1_ob,h2_oa))
                .execute();
              // clang-format on

              auto gf_t_res_1 = std::chrono::high_resolution_clock::now();
              gf_t_res_tot_1 +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_1 - gf_t_x2))
                  .count();

              auto d1_b_norm   = norm(d1_b);
              auto d2_bbb_norm = norm(d2_bbb);
              auto d2_aba_norm = norm(d2_aba);

              auto gf_t_res_2 = std::chrono::high_resolution_clock::now();
              gf_t_res_tot_2 +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_2 - gf_t_res_1))
                  .count();

              double gf_residual =
                sqrt(d1_b_norm * d1_b_norm + d2_bbb_norm * d2_bbb_norm + d2_aba_norm * d2_aba_norm)
                  .real();

              if(debug && rank == root_ppi)
                cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                     << "): #iter " << total_iter << ", residual = " << std::fixed
                     << std::setprecision(6) << gf_residual << std::endl;
              auto gf_t_res = std::chrono::high_resolution_clock::now();
              gf_t_res_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res - gf_t_x2))
                  .count();

              if(gf_residual < gf_threshold) {
                gf_conv = true;
                break;
              }

              // JACOBI
              auto gindx = micro - iter;

              TiledIndexSpace hist_tis{diis_tis, range(gindx, gindx + 1)};
              auto [dh1] = hist_tis.labels<1>("all");

              // clang-format off
              sch
                (dx1_b(h1_ob) = Minv_b(h1_ob,h2_ob) * d1_b(h2_ob))
                (dx2_bbb(p1_vb,h1_ob,h2_ob) = dtmp_bbb(p1_vb,h1_ob,h2_ob) * d2_bbb(p1_vb,h1_ob,h2_ob))
                (dx2_aba(p1_va,h1_ob,h2_oa) = dtmp_aba(p1_va,h1_ob,h2_oa) * d2_aba(p1_va,h1_ob,h2_oa))
                (x1_b(h1_ob) += gf_damping_factor * dx1_b(h1_ob))
                (x2_bbb(p1_vb,h1_ob,h2_ob) += gf_damping_factor * dx2_bbb(p1_vb,h1_ob,h2_ob))
                (x2_aba(p1_va,h1_ob,h2_oa) += gf_damping_factor * dx2_aba(p1_va,h1_ob,h2_oa))
                (e1_b(h1_ob,dh1) = d1_b(h1_ob))
                (e2_bbb(p1_vb,h1_ob,h2_ob,dh1) = d2_bbb(p1_vb,h1_ob,h2_ob))
                (e2_aba(p1_va,h1_ob,h2_oa,dh1) = d2_aba(p1_va,h1_ob,h2_oa))
                (xx1_b(h1_ob,dh1) = x1_b(h1_ob))
                (xx2_bbb(p1_vb,h1_ob,h2_ob,dh1) = x2_bbb(p1_vb,h1_ob,h2_ob))
                (xx2_aba(p1_va,h1_ob,h2_oa,dh1) = x2_aba(p1_va,h1_ob,h2_oa))
                .execute();
              // clang-format on

              auto gf_t_upd = std::chrono::high_resolution_clock::now();
              gf_t_upd_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_upd - gf_t_res))
                  .count();

            } // end micro

            if(gf_conv || iter + ndiis >= gf_maxiter) { break; }

            auto gf_t_tmp = std::chrono::high_resolution_clock::now();

            // DIIS
            ret_gf_diis_b(ec, MO, ndiis, e1_b, e2_bbb, e2_aba, xx1_b, xx2_bbb, xx2_aba, x1_b,
                          x2_bbb, x2_aba, diis_tis, unit_tis, true);

            auto gf_t_dis = std::chrono::high_resolution_clock::now();
            gf_t_dis_tot +=
              std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_dis - gf_t_tmp))
                .count();
          } // end iter

          write_to_disk(x1_b, x1_b_wpi_file);
          write_to_disk(x2_bbb, x2_bbb_wpi_file);
          write_to_disk(x2_aba, x2_aba_wpi_file);

          // sch.deallocate(e1,e2,xx1,xx2).execute();
          sch.deallocate(e1_b, e2_bbb, e2_aba, xx1_b, xx2_bbb, xx2_aba).execute();
        }
        else { gf_conv = true; }

        if(!gf_conv) { //&& rank == root_ppi)
          std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
          tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
        }

        auto   gf_t2 = std::chrono::high_resolution_clock::now();
        double gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
        if(rank == root_ppi) {
          std::string gf_stats;
          gf_stats = gfacc_str("R-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                               ") = ", std::to_string(gftime),
                               " secs, #iter = ", std::to_string(total_iter));

          // if(debug){
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
          //}
          std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
        }

        sch
          .deallocate(Hx1_b, Hx2_bbb, Hx2_aba, d1_b, d2_bbb, d2_aba, dx1_b, dx2_bbb, dx2_aba, x1,
                      x1_b, x2_bbb, x2_aba, Minv, Minv_b, B1, B1_b)
          .execute();

#if GF_PGROUPS
      }
#endif
    } // end pi batch
    ec.pg().barrier();
  } // end all remaining pi

#if GF_PGROUPS
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  pg.destroy_coll();
#endif
  gec.pg().barrier();

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total R-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
  MPI_Comm_free(&gf_comm);
} // gfccsd_driver_ip_b

template<typename T>
void gfccsd_driver_ea_a(
  ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm& subcomm, const TiledIndexSpace& MO,
  Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
  Tensor<T>& f1, Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
  Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
  Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
  Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
  Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
  Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
  Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
  Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
  Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
  Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
  Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
  Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
  Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
  std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt, long int total_orbitals,
  const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI,
  const TiledIndexSpace& unit_tis, string files_prefix, string levelstr, int nva) {
  using ComplexTensor  = Tensor<std::complex<T>>;
  using VComplexTensor = std::vector<Tensor<std::complex<T>>>;
  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(obtiles, otiles)};
  v_beta  = {MO("virt"), range(vbtiles, vtiles)};

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
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
    gfp << std::endl
        << std::string(55, '-') << std::endl
        << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_aaa{o_alpha, v_alpha, v_alpha};
  ComplexTensor dtmp_bab{o_beta, v_alpha, v_beta};
  ComplexTensor::allocate(&gec, dtmp_aaa, dtmp_bab);

  // double au2ev = 27.2113961;

  std::string dtmp_aaa_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_aaa.l" + levelstr;
  std::string dtmp_bab_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_bab.l" + levelstr;

  if(fs::exists(dtmp_aaa_file) && fs::exists(dtmp_bab_file)) {
    read_from_disk(dtmp_aaa, dtmp_aaa_file);
    read_from_disk(dtmp_bab, dtmp_bab_file);
  }
  else {
    ComplexTensor DEArr_EA{O, V, V};

    double       denominator     = 0.0;
    const double lshift          = 1.00000000;
    auto         DEArr_EA_lambda = [&](const IndexVector& bid) {
      const IndexVector            blockid = internal::translate_blockid(bid, DEArr_EA());
      const TAMM_SIZE              size    = DEArr_EA.block_size(blockid);
      std::vector<std::complex<T>> buf(size);

      auto   block_dims   = DEArr_EA.block_dims(blockid);
      auto   block_offset = DEArr_EA.block_offsets(blockid);
      size_t c            = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            denominator =
              gf_omega + p_evl_sorted_occ[i] - p_evl_sorted_virt[j] - p_evl_sorted_virt[k];
            if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * lshift; }
            else if(denominator > 0.0 && denominator < 1.0) { denominator += lshift; }
            buf[c] = 1.0 / std::complex<T>(denominator, gf_eta);
          }
        }
      }
      DEArr_EA.put(blockid, buf);
    };

    gsch.allocate(DEArr_EA).execute();
    if(subcomm != MPI_COMM_NULL) {
      Scheduler sub_sch{sub_ec};
      sub_sch(DEArr_EA() = 0).execute();
      block_for(sub_ec, DEArr_EA(), DEArr_EA_lambda);
      sub_sch(dtmp_aaa() = 0)(dtmp_bab() = 0)(dtmp_aaa(h1_oa, p1_va, p2_va) =
                                                DEArr_EA(h1_oa, p1_va, p2_va))(
        dtmp_bab(h1_ob, p1_va, p2_vb) = DEArr_EA(h1_ob, p1_va, p2_vb))
        .execute();
    }
    gec.pg().barrier();
    gsch.deallocate(DEArr_EA).execute();
    write_to_disk(dtmp_aaa, dtmp_aaa_file);
    write_to_disk(dtmp_bab, dtmp_bab_file);
  }

  //------------------------
  auto nranks     = GA_Nnodes();
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int       world_size;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Comm_size(world_comm, &world_size);
  MPI_Comm gf_comm;

  const size_t        num_oi           = nva;
  size_t              num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  // Check pi's already processed
  for(size_t pi = 0; pi < num_oi; pi++) {
    std::string y1_a_conv_wpi_file =
      files_prefix + ".y1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string y2_aaa_conv_wpi_file =
      files_prefix + ".y2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string y2_bab_conv_wpi_file =
      files_prefix + ".y2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);

    if(fs::exists(y1_a_conv_wpi_file) && fs::exists(y2_aaa_conv_wpi_file) &&
       fs::exists(y2_bab_conv_wpi_file))
      num_pi_processed++;
    else pi_tbp.push_back(pi);
  }

  size_t num_pi_remain = num_oi - num_pi_processed;
  if(num_pi_remain == 0) {
    gsch.deallocate(dtmp_aaa, dtmp_bab).execute();
    return;
  }
  EXPECTS(num_pi_remain == pi_tbp.size());
  // if(num_pi_remain == 0) num_pi_remain = 1;
  int        subranks = std::floor(nranks / num_pi_remain);
  const bool no_pg    = (subranks == 0 || subranks == 1);
  if(no_pg) subranks = nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  // Figure out how many orbitals in pi_tbp can be processed with subranks
  // TODO: gf_nprocs_pi must be a multiple of total #ranks for best performance.
  size_t num_oi_can_bp = std::ceil(nranks / (1.0 * subranks));
  if(num_pi_remain < num_oi_can_bp) {
    num_oi_can_bp = num_pi_remain;
    subranks      = std::floor(nranks / num_pi_remain);
    if(no_pg) subranks = nranks;
  }

  if(rank == 0) {
    cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
         << num_oi_can_bp << endl;
    cout << "No of processes used to compute each orbital = " << subranks << endl;
    // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  int color = 0;
  if(subranks > 1) color = world_rank / subranks;

  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (alpha)        //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
  ProcGroup        pg = ProcGroup::create_coll(gf_comm);
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};
#else
  Scheduler&        sch = gsch;
  ExecutionContext& ec  = gec;
#endif

  int64_t root_ppi = 0;
  for(size_t pib = 0; pib < num_pi_remain; pib += num_oi_can_bp) {
    size_t piv_end = num_oi_can_bp;
    if(pib + num_oi_can_bp > num_pi_remain) piv_end = pib + num_oi_can_bp - num_pi_remain;
    for(size_t piv = 0; piv < piv_end; piv++) {
      size_t pi = pi_tbp[piv + pib];
#if GF_PGROUPS
      if((rank >= piv * subranks && rank < (piv * subranks + subranks)) || no_pg) {
        if(!no_pg) root_ppi = piv * subranks; // root of sub-group
#endif

        auto gf_t1 = std::chrono::high_resolution_clock::now();

        bool          gf_conv = false;
        ComplexTensor Minv{{V, V}, {1, 1}};
        ComplexTensor y1{V};
        Tensor<T>     B1{V};

        ComplexTensor Hy1_a{v_alpha};
        ComplexTensor Hy2_aaa{o_alpha, v_alpha, v_alpha};
        ComplexTensor Hy2_bab{o_beta, v_alpha, v_beta};
        ComplexTensor y1_a{v_alpha};
        ComplexTensor y2_aaa{o_alpha, v_alpha, v_alpha};
        ComplexTensor y2_bab{o_beta, v_alpha, v_beta};
        Tensor<T>     B1_a{v_alpha};

        // if(rank==0) cout << "allocate B" << endl;
        sch.allocate(B1, B1_a).execute();
        LabelLoopNest loop_nest{B1().labels()};
        sch(B1() = 0).execute();

        for(const IndexVector& bid: loop_nest) {
          const IndexVector blockid = internal::translate_blockid(bid, B1());

          const TAMM_SIZE         size = B1.block_size(blockid);
          std::vector<TensorType> buf(size);
          B1.get(blockid, buf);
          auto block_dims   = B1.block_dims(blockid);
          auto block_offset = B1.block_offsets(blockid);
          auto dim          = block_dims[0];
          auto offset       = block_offset[0];
          if(pi >= offset && pi < offset + dim) buf[pi - offset] = 1.0;
          B1.put(blockid, buf);
        }

        sch(B1_a(p1_va) = B1(p1_va)).deallocate(B1).execute();
        ec.pg().barrier();

        sch.allocate(Hy1_a, Hy2_aaa, Hy2_bab, y1_a, y2_aaa, y2_bab).execute();

        double gf_t_guess     = 0.0;
        double gf_t_y1_tot    = 0.0;
        double gf_t_y2_tot    = 0.0;
        double gf_t_res_tot   = 0.0;
        double gf_t_res_tot_1 = 0.0;
        double gf_t_res_tot_2 = 0.0;
        double gf_t_upd_tot   = 0.0;
        double gf_t_dis_tot   = 0.0;
        size_t gf_iter        = 0;

        std::string y1_a_inter_wpi_file =
          files_prefix + ".y1_a.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string y2_aaa_inter_wpi_file =
          files_prefix + ".y2_aaa.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string y2_bab_inter_wpi_file =
          files_prefix + ".y2_bab.inter.w" + gfo.str() + ".oi" + std::to_string(pi);

        if(fs::exists(y1_a_inter_wpi_file) && fs::exists(y2_aaa_inter_wpi_file) &&
           fs::exists(y2_bab_inter_wpi_file)) {
          read_from_disk(y1_a, y1_a_inter_wpi_file);
          read_from_disk(y2_aaa, y2_aaa_inter_wpi_file);
          read_from_disk(y2_bab, y2_bab_inter_wpi_file);
        }
        else {
          // clang-format off
          sch
            .allocate(y1,Minv)
            (y1()      = 0)
            (Minv()    = 0)
            .execute();
          // clang-format on

          gf_guess_ea(ec, MO, nvir, gf_omega, gf_eta, pi, p_evl_sorted_virt, t2v2_v, y1, Minv,
                      true);

          sch(y1_a(p1_va) = y1(p1_va)).deallocate(y1, Minv).execute();
        }

        // GMRES

        ComplexTensor tmp{};
        sch.allocate(tmp).execute();

        do {
          int64_t gmres_hist = ngmres;

          gf_iter++;

          ComplexTensor  r1_a{v_alpha};
          ComplexTensor  r2_aaa{o_alpha, v_alpha, v_alpha};
          ComplexTensor  r2_bab{o_beta, v_alpha, v_beta};
          VComplexTensor Q1_a;
          VComplexTensor Q2_aaa;
          VComplexTensor Q2_bab;

          gfccsd_y1_a(sch, MO, Hy1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_a, y2_aaa, y2_bab,
                      f1, iy1_a, iy1_2_1_a, iy1_2_1_b, v2ijab_aaaa, v2ijab_abab, cholOV_a, cholOV_b,
                      cholVV_a, CI, unit_tis, false);

          gfccsd_y2_a(sch, MO, Hy2_aaa, Hy2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_a,
                      y2_aaa, y2_bab, f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b,
                      iy3_1_aaaa, iy3_1_baba, iy3_1_abba, iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_bbbb,
                      iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb,
                      iy4_1_abba, iy4_2_aaaa, iy4_2_abba, iy5_aaaa, iy5_baab, iy5_bbbb, iy5_baba,
                      iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a,
                      cholVV_b, v2ijab_aaaa, v2ijab_abab, CI, unit_tis, false);

          // clang-format off
          sch 
            .allocate(r1_a,  r2_aaa,  r2_bab)
            (r1_a()   =  1.0 * Hy1_a())
            (r2_aaa() =  1.0 * Hy2_aaa())
            (r2_bab() =  1.0 * Hy2_bab())
            (r1_a(p1_va)               -= std::complex<double>(gf_omega,gf_eta) * y1_a(p1_va))
            (r2_aaa(h1_oa,p1_va,p2_va) -= std::complex<double>(gf_omega,gf_eta) * y2_aaa(h1_oa,p1_va,p2_va))
            (r2_bab(h1_ob,p1_va,p2_vb) -= std::complex<double>(gf_omega,gf_eta) * y2_bab(h1_ob,p1_va,p2_vb))
            (r1_a() += B1_a())
            .execute();
          // clang-format on

          auto r1_a_norm   = norm(r1_a);
          auto r2_aaa_norm = norm(r2_aaa);
          auto r2_bab_norm = norm(r2_bab);

          auto gf_residual = sqrt(r1_a_norm * r1_a_norm + 0.5 * r2_aaa_norm * r2_aaa_norm +
                                  r2_bab_norm * r2_bab_norm);

          if(rank == root_ppi && debug) {
            cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                 << "): #iter " << gf_iter << "(" << gf_maxiter << "), residual = " << std::fixed
                 << std::setprecision(6) << gf_residual << std::endl;
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

          for(auto k = 0; k < gmres_hist; k++) {
            ComplexTensor q1_a{v_alpha};
            ComplexTensor q2_aaa{o_alpha, v_alpha, v_alpha};
            ComplexTensor q2_bab{o_beta, v_alpha, v_beta};

            gfccsd_y1_a(sch, MO, Hy1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k], Q2_aaa[k],
                        Q2_bab[k], f1, iy1_a, iy1_2_1_a, iy1_2_1_b, v2ijab_aaaa, v2ijab_abab,
                        cholOV_a, cholOV_b, cholVV_a, CI, unit_tis, false);

            gfccsd_y2_a(sch, MO, Hy2_aaa, Hy2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k],
                        Q2_aaa[k], Q2_bab[k], f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a,
                        iy2_b, iy3_1_aaaa, iy3_1_baba, iy3_1_abba, iy3_1_2_a, iy3_1_2_b, iy3_aaaa,
                        iy3_bbbb, iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab, iy4_1_baba,
                        iy4_1_bbbb, iy4_1_abba, iy4_2_aaaa, iy4_2_abba, iy5_aaaa, iy5_baab,
                        iy5_bbbb, iy5_baba, iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b, cholOV_a,
                        cholOV_b, cholVV_a, cholVV_b, v2ijab_aaaa, v2ijab_abab, CI, unit_tis,
                        false);

            // clang-format off
            sch 
              .allocate(q1_a,q2_aaa,q2_bab)
              (q1_a()   =  -1.0 * Hy1_a())
              (q2_aaa() =  -1.0 * Hy2_aaa())
              (q2_bab() =  -1.0 * Hy2_bab())
              (q1_a(p1_va)               += std::complex<double>(gf_omega,gf_eta) * Q1_a[k](p1_va))
              (q2_aaa(h1_oa,p1_va,p2_va) += std::complex<double>(gf_omega,gf_eta) * Q2_aaa[k](h1_oa,p1_va,p2_va))
              (q2_bab(h1_ob,p1_va,p2_vb) += std::complex<double>(gf_omega,gf_eta) * Q2_bab[k](h1_ob,p1_va,p2_vb))
              .execute();
            // clang-format on

            // Arnoldi iteration or G-S orthogonalization
            for(auto j = 0; j <= k; j++) {
              auto conj_a   = tamm::conj(Q1_a[j]);
              auto conj_aaa = tamm::conj(Q2_aaa[j]);
              auto conj_bab = tamm::conj(Q2_bab[j]);

              // clang-format off
              sch
                (tmp()  = 1.0 * conj_a(p1_va) * q1_a(p1_va))
                (tmp() += 0.5 * conj_aaa(h1_oa,p1_va,p2_va) * q2_aaa(h1_oa,p1_va,p2_va))
                (tmp() += 1.0 * conj_bab(h1_ob,p1_va,p2_vb) * q2_bab(h1_ob,p1_va,p2_vb))
                (q1_a()   -= tmp() * Q1_a[j]())
                (q2_aaa() -= tmp() * Q2_aaa[j]())
                (q2_bab() -= tmp() * Q2_bab[j]())
                .deallocate(conj_a,conj_aaa,conj_bab)
                .execute();
              // clang-format on

              H(j, k) = get_scalar(tmp);
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

            CMatrix Hsub = H.block(0, 0, k + 2, k + 1);
            CMatrix bsub = b.block(0, 0, k + 2, 1);

            // apply givens rotation
            for(auto i = 0; i < k; i++) {
              auto temp   = cn(i, 0) * H(i, k) + sn(i, 0) * H(i + 1, k);
              H(i + 1, k) = -sn(i, 0) * H(i, k) + cn(i, 0) * H(i + 1, k);
              H(i, k)     = temp;
            }

            // if(std::abs(H(k,k))<1e-16){
            //   cn(k,0) = std::complex<double>(0,0);
            //   sn(k,0) = std::complex<double>(1,0);
            // }
            // else{
            auto t   = sqrt(H(k, k) * H(k, k) + H(k + 1, k) * H(k + 1, k));
            cn(k, 0) = abs(H(k, k)) / t;
            sn(k, 0) = cn(k, 0) * H(k + 1, k) / H(k, k);
            // }

            H(k, k)     = cn(k, 0) * H(k, k) + sn(k, 0) * H(k + 1, k);
            H(k + 1, k) = std::complex<double>(0, 0);

            b(k + 1, 0) = -sn(k, 0) * b(k, 0);
            b(k, 0)     = cn(k, 0) * b(k, 0);

            if(rank == root_ppi && debug)
              cout << "k: " << k << ", error: " << std::abs(b(k + 1, 0)) << endl;

            // normalization
            if(std::abs(b(k + 1, 0)) > 1e-2) {
              tamm::scale_ip(q1_a, scaling);
              tamm::scale_ip(q2_aaa, scaling);
              tamm::scale_ip(q2_bab, scaling);
              Q1_a.push_back(q1_a);
              Q2_aaa.push_back(q2_aaa);
              Q2_bab.push_back(q2_bab);
            }
            else {
              gmres_hist = k + 1;
              sch.deallocate(q1_a, q2_aaa, q2_bab).execute();
              break;
            }
          } // k loop

          // solve a least square problem in the subspace
          CMatrix Hsub = H.block(0, 0, gmres_hist, gmres_hist);
          CMatrix bsub = b.block(0, 0, gmres_hist, 1);
          CMatrix y    = Hsub.householderQr().solve(bsub);

          if(rank == 0) cout << "residual: " << (bsub - Hsub * y).norm() << endl;

          for(auto i = 0; i < gmres_hist; i++) {
            // clang-format off
            sch
              (y1_a(p1_va)               += y(i,0) * Q1_a[i](p1_va))
              (y2_aaa(h1_oa,p1_va,p2_va) += y(i,0) * Q2_aaa[i](h1_oa,p1_va,p2_va))
              (y2_bab(h1_ob,p1_va,p2_vb) += y(i,0) * Q2_bab[i](h1_ob,p1_va,p2_vb));
            // clang-format on
          }
          sch.execute();

          write_to_disk(y1_a, y1_a_inter_wpi_file);
          write_to_disk(y2_aaa, y2_aaa_inter_wpi_file);
          write_to_disk(y2_bab, y2_bab_inter_wpi_file);

          free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
          Q1_a.clear();
          Q2_aaa.clear();
          Q2_bab.clear();

        } while(true);

        // deallocate memory
        sch.deallocate(tmp).execute();

        if(gf_conv) {
          std::string y1_a_conv_wpi_file =
            files_prefix + ".y1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
          std::string y2_aaa_conv_wpi_file =
            files_prefix + ".y2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
          std::string y2_bab_conv_wpi_file =
            files_prefix + ".y2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);
          write_to_disk(y1_a, y1_a_conv_wpi_file);
          write_to_disk(y2_aaa, y2_aaa_conv_wpi_file);
          write_to_disk(y2_bab, y2_bab_conv_wpi_file);
          fs::remove(y1_a_inter_wpi_file);
          fs::remove(y2_aaa_inter_wpi_file);
          fs::remove(y2_bab_inter_wpi_file);
        }

        if(!gf_conv) { // && rank == root_ppi
          std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
          tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
        }

        auto   gf_t2 = std::chrono::high_resolution_clock::now();
        double gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
        if(rank == root_ppi) {
          std::string gf_stats;
          gf_stats = gfacc_str("A-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                               ") = ", std::to_string(gftime),
                               " secs, #iter = ", std::to_string(gf_iter));

          // if(debug){
          gf_stats += gfacc_str("|----------initial guess  : ", std::to_string(gf_t_guess));
          gf_stats += gfacc_str("|----------y1 contraction : ", std::to_string(gf_t_y1_tot));
          gf_stats += gfacc_str("|----------y2 contraction : ", std::to_string(gf_t_y2_tot));
          gf_stats += gfacc_str("|----------computing res. : ", std::to_string(gf_t_res_tot));
          gf_stats +=
            gfacc_str("           |----------misc. contr. : ", std::to_string(gf_t_res_tot_1));
          gf_stats +=
            gfacc_str("           |----------compt. norm  : ", std::to_string(gf_t_res_tot_2));
          gf_stats += gfacc_str("|----------updating y1/y2 : ", std::to_string(gf_t_upd_tot));
          gf_stats += gfacc_str("|----------diis update    : ", std::to_string(gf_t_dis_tot));
          //}
          std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
        }

        sch.deallocate(Hy1_a, Hy2_aaa, Hy2_bab, y1_a, y2_aaa, y2_bab, B1_a).execute();

#if GF_PGROUPS
      }
#endif
    } // end pi batch
    ec.pg().barrier();
  } // end all remaining pi

#if GF_PGROUPS
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  pg.destroy_coll();
#endif
  gec.pg().barrier();

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total A-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  gsch.deallocate(dtmp_aaa, dtmp_bab).execute();
  MPI_Comm_free(&gf_comm);
} // gfccsd_driver_ea_a

template<typename T>
void gfccsd_driver_ea_b(
  ExecutionContext& gec, ExecutionContext& sub_ec, MPI_Comm& subcomm, const TiledIndexSpace& MO,
  Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
  Tensor<T>& f1, Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
  Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
  Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
  Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
  Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
  Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
  Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
  Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
  Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
  Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
  Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
  Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
  Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
  std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt, long int total_orbitals,
  const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI,
  const TiledIndexSpace& unit_tis, string files_prefix, string levelstr, int nva, int nvb) {
  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");
  // auto [u1] = unit_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(obtiles, otiles)};
  v_beta  = {MO("virt"), range(vbtiles, vtiles)};

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
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
    gfp << std::endl
        << std::string(55, '-') << std::endl
        << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
    std::cout << gfp.str() << std::flush;
  }

  ComplexTensor dtmp_bbb{o_beta, v_beta, v_beta};
  ComplexTensor dtmp_aba{o_alpha, v_beta, v_alpha};
  ComplexTensor::allocate(&gec, dtmp_bbb, dtmp_aba);

  // double au2ev = 27.2113961;

  std::string dtmp_bbb_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_bbb.l" + levelstr;
  std::string dtmp_aba_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_aba.l" + levelstr;

  if(fs::exists(dtmp_bbb_file) && fs::exists(dtmp_aba_file)) {
    read_from_disk(dtmp_bbb, dtmp_bbb_file);
    read_from_disk(dtmp_aba, dtmp_aba_file);
  }
  else {
    ComplexTensor DEArr_EA{O, V, V};

    double       denominator     = 0.0;
    const double lshift          = 1.00000000;
    auto         DEArr_EA_lambda = [&](const IndexVector& bid) {
      const IndexVector            blockid = internal::translate_blockid(bid, DEArr_EA());
      const TAMM_SIZE              size    = DEArr_EA.block_size(blockid);
      std::vector<std::complex<T>> buf(size);

      auto   block_dims   = DEArr_EA.block_dims(blockid);
      auto   block_offset = DEArr_EA.block_offsets(blockid);
      size_t c            = 0;
      for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
        for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
          for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
            denominator =
              gf_omega + p_evl_sorted_occ[i] - p_evl_sorted_virt[j] - p_evl_sorted_virt[k];
            if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * lshift; }
            else if(denominator > 0.0 && denominator < 1.0) { denominator += lshift; }
            buf[c] = 1.0 / std::complex<T>(denominator, gf_eta);
          }
        }
      }
      DEArr_EA.put(blockid, buf);
    };

    gsch.allocate(DEArr_EA).execute();
    if(subcomm != MPI_COMM_NULL) {
      Scheduler sub_sch{sub_ec};
      sub_sch(DEArr_EA() = 0).execute();
      block_for(sub_ec, DEArr_EA(), DEArr_EA_lambda);
      // clang-format off
      sub_sch      
        (dtmp_bbb() = 0)
        (dtmp_aba() = 0)
        (dtmp_bbb(h1_ob,p1_vb,p2_vb) = DEArr_EA(h1_ob,p1_vb,p2_vb))
        (dtmp_aba(h1_oa,p1_vb,p2_va) = DEArr_EA(h1_oa,p1_vb,p2_va))
        .execute();
      // clang-format on
    }
    gec.pg().barrier();
    gsch.deallocate(DEArr_EA).execute();

    write_to_disk(dtmp_bbb, dtmp_bbb_file);
    write_to_disk(dtmp_aba, dtmp_aba_file);
  }

  //------------------------
  auto nranks     = GA_Nnodes();
  auto world_comm = gec.pg().comm();
  auto world_rank = gec.pg().rank().value();

  MPI_Group world_group;
  int       world_size;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Comm_size(world_comm, &world_size);
  MPI_Comm gf_comm;

  const size_t        num_oi           = nvb;
  size_t              num_pi_processed = 0;
  std::vector<size_t> pi_tbp;
  // Check pi's already processed
  for(size_t pi = nva; pi < nva + num_oi; pi++) {
    std::string y1_b_wpi_file = files_prefix + ".y1_b.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string y2_bbb_wpi_file =
      files_prefix + ".y2_bbb.w" + gfo.str() + ".oi" + std::to_string(pi);
    std::string y2_aba_wpi_file =
      files_prefix + ".y2_aba.w" + gfo.str() + ".oi" + std::to_string(pi);

    if(fs::exists(y1_b_wpi_file) && fs::exists(y2_bbb_wpi_file) && fs::exists(y2_aba_wpi_file))
      num_pi_processed++;
    else pi_tbp.push_back(pi);
  }

  size_t num_pi_remain = num_oi - num_pi_processed;
  if(num_pi_remain == 0) {
    gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
    return;
  }
  EXPECTS(num_pi_remain == pi_tbp.size());
  // if(num_pi_remain == 0) num_pi_remain = 1;
  int        subranks = std::floor(nranks / num_pi_remain);
  const bool no_pg    = (subranks == 0 || subranks == 1);
  if(no_pg) subranks = nranks;
  if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;

  // Figure out how many orbitals in pi_tbp can be processed with subranks
  // TODO: gf_nprocs_pi must be a multiple of total #ranks for best performance.
  size_t num_oi_can_bp = std::ceil(nranks / (1.0 * subranks));
  if(num_pi_remain < num_oi_can_bp) {
    num_oi_can_bp = num_pi_remain;
    subranks      = std::floor(nranks / num_pi_remain);
    if(no_pg) subranks = nranks;
  }

  if(rank == 0) {
    cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
         << num_oi_can_bp << endl;
    cout << "No of processes used to compute each orbital = " << subranks << endl;
    // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
  }

  int color = 0;
  if(subranks > 1) color = world_rank / subranks;

  MPI_Comm_split(world_comm, color, world_rank, &gf_comm);

  ///////////////////////////
  //                       //
  //  MAIN ITERATION LOOP  //
  //        (beta)        //
  ///////////////////////////
  auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
  ProcGroup        pg = ProcGroup::create_coll(gf_comm);
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  Scheduler        sch{ec};
#else
  Scheduler&        sch = gsch;
  ExecutionContext& ec  = gec;
#endif

  int64_t root_ppi = 0;
  for(size_t pib = 0; pib < num_pi_remain; pib += num_oi_can_bp) {
    size_t piv_end = num_oi_can_bp;
    if(pib + num_oi_can_bp > num_pi_remain) piv_end = pib + num_oi_can_bp - num_pi_remain;
    for(size_t piv = 0; piv < piv_end; piv++) {
      size_t pi = pi_tbp[piv + pib];
#if GF_PGROUPS
      if((rank >= piv * subranks && rank < (piv * subranks + subranks)) || no_pg) {
        if(!no_pg) root_ppi = piv * subranks; // root of sub-group
#endif

        auto gf_t1 = std::chrono::high_resolution_clock::now();

        bool          gf_conv = false;
        ComplexTensor Minv{{V, V}, {1, 1}};
        ComplexTensor y1{V};
        Tensor<T>     B1{V};

        ComplexTensor Hy1_b{v_beta};
        ComplexTensor Hy2_bbb{o_beta, v_beta, v_beta};
        ComplexTensor Hy2_aba{o_alpha, v_beta, v_alpha};
        ComplexTensor d1_b{v_beta};
        ComplexTensor d2_bbb{o_beta, v_beta, v_beta};
        ComplexTensor d2_aba{o_alpha, v_beta, v_alpha};
        ComplexTensor Minv_b{v_beta, v_beta};
        ComplexTensor dy1_b{v_beta};
        ComplexTensor dy2_bbb{o_beta, v_beta, v_beta};
        ComplexTensor dy2_aba{o_alpha, v_beta, v_alpha};
        ComplexTensor y1_b{v_beta};
        ComplexTensor y2_bbb{o_beta, v_beta, v_beta};
        ComplexTensor y2_aba{o_alpha, v_beta, v_alpha};
        Tensor<T>     B1_b{v_beta};

        sch
          .allocate(Hy1_b, Hy2_bbb, Hy2_aba, d1_b, d2_bbb, d2_aba, dy1_b, dy2_bbb, dy2_aba, y1,
                    y1_b, y2_bbb, y2_aba, Minv, Minv_b, B1, B1_b)
          .execute();

        int    total_iter     = 0;
        double gf_t_guess     = 0.0;
        double gf_t_y1_tot    = 0.0;
        double gf_t_y2_tot    = 0.0;
        double gf_t_res_tot   = 0.0;
        double gf_t_res_tot_1 = 0.0;
        double gf_t_res_tot_2 = 0.0;
        double gf_t_upd_tot   = 0.0;
        double gf_t_dis_tot   = 0.0;

        std::string y1_b_wpi_file =
          files_prefix + ".y1_b.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string y2_bbb_wpi_file =
          files_prefix + ".y2_bbb.w" + gfo.str() + ".oi" + std::to_string(pi);
        std::string y2_aba_wpi_file =
          files_prefix + ".y2_aba.w" + gfo.str() + ".oi" + std::to_string(pi);

        if(!(fs::exists(y1_b_wpi_file) && fs::exists(y2_bbb_wpi_file) &&
             fs::exists(y2_aba_wpi_file))) {
          ComplexTensor e1_b{v_beta, diis_tis};
          ComplexTensor e2_bbb{o_beta, v_beta, v_beta, diis_tis};
          ComplexTensor e2_aba{o_alpha, v_beta, v_alpha, diis_tis};
          ComplexTensor yy1_b{v_beta, diis_tis};
          ComplexTensor yy2_bbb{o_beta, v_beta, v_beta, diis_tis};
          ComplexTensor yy2_aba{o_alpha, v_beta, v_alpha, diis_tis};

          // clang-format off
          sch
            .allocate(e1_b, e2_bbb, e2_aba, 
                    yy1_b,yy2_bbb,yy2_aba)
            (y1()      = 0)
            (Minv()    = 0)
            (Hy1_b()   = 0)
            (Hy2_bbb() = 0)
            (Hy2_aba() = 0)
            (d1_b()    = 0)
            (d2_bbb()  = 0)
            (d2_aba()  = 0)
            (y1_b()    = 0)
            (y2_bbb()  = 0)
            (y2_aba()  = 0)
            (Minv_b()  = 0)
            .execute();
          // clang-format on

          LabelLoopNest loop_nest{B1().labels()};
          sch(B1() = 0).execute();

          for(const IndexVector& bid: loop_nest) {
            const IndexVector blockid = internal::translate_blockid(bid, B1());

            const TAMM_SIZE         size = B1.block_size(blockid);
            std::vector<TensorType> buf(size);
            B1.get(blockid, buf);
            auto block_dims   = B1.block_dims(blockid);
            auto block_offset = B1.block_offsets(blockid);
            auto dim          = block_dims[0];
            auto offset       = block_offset[0];
            if(pi >= offset && pi < offset + dim) buf[pi - offset] = 1.0;
            B1.put(blockid, buf);
          }

          sch(B1_b(p1_vb) = B1(p1_vb)).execute();
          ec.pg().barrier();

          auto gf_t_guess1 = std::chrono::high_resolution_clock::now();

          gf_guess_ea(ec, MO, nvir, gf_omega, gf_eta, pi, p_evl_sorted_virt, t2v2_v, y1, Minv,
                      true);

          // clang-format off
          sch
            (y1_b(p1_vb) = y1(p1_vb))
            (Minv_b(p1_vb,p2_vb)  = Minv(p1_vb,p2_vb))
            .execute();
          // clang-format on

          auto gf_t_guess2 = std::chrono::high_resolution_clock::now();
          gf_t_guess =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_guess2 - gf_t_guess1))
              .count();

          for(size_t iter = 0; iter < gf_maxiter; iter += ndiis) {
            for(size_t micro = iter; micro < std::min(iter + ndiis, gf_maxiter); micro++) {
              total_iter    = micro;
              auto gf_t_ini = std::chrono::high_resolution_clock::now();

              gfccsd_y1_b(sch, MO, Hy1_b, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_b, y2_bbb,
                          y2_aba, f1, iy1_b, iy1_2_1_a, iy1_2_1_b, v2ijab_bbbb, v2ijab_abab,
                          cholOV_a, cholOV_b, cholVV_b, CI, unit_tis, false);

              auto gf_t_y1 = std::chrono::high_resolution_clock::now();
              gf_t_y1_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_y1 - gf_t_ini))
                  .count();

              gfccsd_y2_b(sch, MO, Hy2_bbb, Hy2_aba, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_b,
                          y2_bbb, y2_aba, f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b,
                          iy3_1_bbbb, iy3_1_abab, iy3_1_baab, iy3_1_2_a, iy3_1_2_b, iy3_aaaa,
                          iy3_bbbb, iy3_abab, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab,
                          iy4_1_bbbb, iy4_1_abba, iy4_1_abab, iy4_2_bbbb, iy4_2_baab, iy5_aaaa,
                          iy5_baab, iy5_bbbb, iy5_abab, iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b,
                          cholOV_a, cholOV_b, cholVV_a, cholVV_b, v2ijab_bbbb, v2ijab_abab, CI,
                          unit_tis, false);

              sch.execute(); // TODO: not needed if not profiling
              auto gf_t_y2 = std::chrono::high_resolution_clock::now();
              gf_t_y2_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_y2 - gf_t_y1))
                  .count();

              // clang-format off
              sch
                (d1_b() =  1.0 * Hy1_b())
                (d1_b(p1_vb) -= std::complex<double>(gf_omega,gf_eta) * y1_b(p1_vb))
                (d1_b() += B1_b())
                (d2_bbb() =  1.0 * Hy2_bbb())
                (d2_bbb(h1_ob,p1_vb,p2_vb) -= std::complex<double>(gf_omega,gf_eta) * y2_bbb(h1_ob,p1_vb,p2_vb))
                (d2_aba() =  1.0 * Hy2_aba())
                (d2_aba(h1_oa,p1_vb,p2_va) -= std::complex<double>(gf_omega,gf_eta) * y2_aba(h1_oa,p1_vb,p2_va))
                .execute();
              // clang-format on

              auto gf_t_res_1 = std::chrono::high_resolution_clock::now();
              gf_t_res_tot_1 +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_1 - gf_t_y2))
                  .count();

              auto d1_b_norm   = norm(d1_b);
              auto d2_bbb_norm = norm(d2_bbb);
              auto d2_aba_norm = norm(d2_aba);

              auto gf_t_res_2 = std::chrono::high_resolution_clock::now();
              gf_t_res_tot_2 +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_2 - gf_t_res_1))
                  .count();

              double gf_residual = sqrt(d1_b_norm * d1_b_norm + 0.5 * d2_bbb_norm * d2_bbb_norm +
                                        d2_aba_norm * d2_aba_norm)
                                     .real();

              if(debug && rank == root_ppi)
                cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                     << "): #iter " << total_iter << ", residual = " << std::fixed
                     << std::setprecision(6) << gf_residual << std::endl;
              auto gf_t_res = std::chrono::high_resolution_clock::now();
              gf_t_res_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res - gf_t_y2))
                  .count();

              if(gf_residual < gf_threshold) {
                gf_conv = true;
                break;
              }

              // JACOBI
              auto gindx = micro - iter;

              TiledIndexSpace hist_tis{diis_tis, range(gindx, gindx + 1)};
              auto [dh1] = hist_tis.labels<1>("all");

              // clang-format off
              sch
                (dy1_b(p1_vb) = Minv_b(p1_vb,p2_vb) * d1_b(p2_vb))
                (dy2_bbb(h1_ob,p1_vb,p2_vb) = dtmp_bbb(h1_ob,p1_vb,p2_vb) * d2_bbb(h1_ob,p1_vb,p2_vb))
                (dy2_aba(h1_oa,p1_vb,p2_va) = dtmp_aba(h1_oa,p1_vb,p2_va) * d2_aba(h1_oa,p1_vb,p2_va))
                (y1_b(p1_vb) += gf_damping_factor * dy1_b(p1_vb))
                (y2_bbb(h1_ob,p1_vb,p2_vb) += gf_damping_factor * dy2_bbb(h1_ob,p1_vb,p2_vb))
                (y2_aba(h1_oa,p1_vb,p2_va) += gf_damping_factor * dy2_aba(h1_oa,p1_vb,p2_va))
                (e1_b(p1_vb,dh1) = d1_b(p1_vb))
                (e2_bbb(h1_ob,p1_vb,p2_vb,dh1) = d2_bbb(h1_ob,p1_vb,p2_vb))
                (e2_aba(h1_oa,p1_vb,p2_va,dh1) = d2_aba(h1_oa,p1_vb,p2_va))
                (yy1_b(p1_vb,dh1) = y1_b(p1_vb))
                (yy2_bbb(h1_ob,p1_vb,p2_vb,dh1) = y2_bbb(h1_ob,p1_vb,p2_vb))
                (yy2_aba(h1_oa,p1_vb,p2_va,dh1) = y2_aba(h1_oa,p1_vb,p2_va))
                .execute();
              // clang-format on

              auto gf_t_upd = std::chrono::high_resolution_clock::now();
              gf_t_upd_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_upd - gf_t_res))
                  .count();

            } // end micro

            if(gf_conv || iter + ndiis >= gf_maxiter) { break; }

            auto gf_t_tmp = std::chrono::high_resolution_clock::now();

            // DIIS
            adv_gf_diis_b(ec, MO, ndiis, e1_b, e2_bbb, e2_aba, yy1_b, yy2_bbb, yy2_aba, y1_b,
                          y2_bbb, y2_aba, diis_tis, unit_tis, true);

            auto gf_t_dis = std::chrono::high_resolution_clock::now();
            gf_t_dis_tot +=
              std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_dis - gf_t_tmp))
                .count();
          } // end iter

          write_to_disk(y1_b, y1_b_wpi_file);
          write_to_disk(y2_bbb, y2_bbb_wpi_file);
          write_to_disk(y2_aba, y2_aba_wpi_file);

          sch.deallocate(e1_b, e2_bbb, e2_aba, yy1_b, yy2_bbb, yy2_aba).execute();
        }
        else { gf_conv = true; }

        if(!gf_conv) { // && rank == root_ppi
          std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
          tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
        }

        auto   gf_t2 = std::chrono::high_resolution_clock::now();
        double gftime =
          std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
        if(rank == root_ppi) {
          std::string gf_stats;
          gf_stats = gfacc_str("A-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                               ") = ", std::to_string(gftime),
                               " secs, #iter = ", std::to_string(total_iter));

          // if(debug){
          gf_stats += gfacc_str("|----------initial guess  : ", std::to_string(gf_t_guess));
          gf_stats += gfacc_str("|----------y1 contraction : ", std::to_string(gf_t_y1_tot));
          gf_stats += gfacc_str("|----------y2 contraction : ", std::to_string(gf_t_y2_tot));
          gf_stats += gfacc_str("|----------computing res. : ", std::to_string(gf_t_res_tot));
          gf_stats +=
            gfacc_str("           |----------misc. contr. : ", std::to_string(gf_t_res_tot_1));
          gf_stats +=
            gfacc_str("           |----------compt. norm  : ", std::to_string(gf_t_res_tot_2));
          gf_stats += gfacc_str("|----------updating y1/y2 : ", std::to_string(gf_t_upd_tot));
          gf_stats += gfacc_str("|----------diis update    : ", std::to_string(gf_t_dis_tot));
          //}
          std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
        }

        sch
          .deallocate(Hy1_b, Hy2_bbb, Hy2_aba, d1_b, d2_bbb, d2_aba, dy1_b, dy2_bbb, dy2_aba, y1_b,
                      y2_bbb, y2_aba, y1, Minv, Minv_b, B1, B1_b)
          .execute();

#if GF_PGROUPS
      }
#endif
    } // end pi batch
    ec.pg().barrier();
  } // end all remaining pi

#if GF_PGROUPS
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  pg.destroy_coll();
#endif
  gec.pg().barrier();

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << "Total A-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;
  }

  gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
  MPI_Comm_free(&gf_comm);
}
#endif // gfccsd_driver_ea_b

////////////////////main entry point///////////////////////////
void gfccsd_driver(std::string filename, OptionsMap options_map) {
  using T = double;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank = ec.pg().rank();

  ProcGroup        pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
  ExecutionContext ec_l{pg_l, DistributionKind::nw, MemoryManagerKind::local};

  auto restart_time_start = std::chrono::high_resolution_clock::now();

  auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt,
        AO_tis, scf_conv] = hartree_fock_driver<T>(ec, filename, options_map);

  int nsranks = sys_data.nbf / 15;
  if(nsranks < 1) nsranks = 1;
  int ga_cnn = GA_Cluster_nnodes();
  if(nsranks > ga_cnn) nsranks = ga_cnn;
  nsranks = nsranks * GA_Cluster_nprocs(0);
  int subranks[nsranks];
  for(int i = 0; i < nsranks; i++) subranks[i] = i;
  auto      world_comm = ec.pg().comm();
  MPI_Group world_group;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Group subgroup;
  MPI_Group_incl(world_group, nsranks, subranks, &subgroup);
  MPI_Comm subcomm;
  MPI_Comm_create(world_comm, subgroup, &subcomm);

  ProcGroup         sub_pg;
  ExecutionContext* sub_ec = nullptr;

  if(subcomm != MPI_COMM_NULL) {
    sub_pg = ProcGroup::create_coll(subcomm);
    sub_ec = new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
  }

  Scheduler sub_sch{*sub_ec};

  // force writet on
  sys_data.options_map.ccsd_options.writet       = true;
  sys_data.options_map.ccsd_options.computeTData = true;

  CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;
  debug                     = ccsd_options.debug;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = setupMOIS(sys_data);

  std::string out_fp       = sys_data.output_file_prefix + "." + ccsd_options.basis;
  std::string files_dir    = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string t1file       = files_prefix + ".t1amp";
  std::string t2file       = files_prefix + ".t2amp";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";
  std::string ccsdstatus   = files_prefix + ".ccsdstatus";

  const bool is_rhf = sys_data.is_restricted;

  bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                              fs::exists(f1file) && fs::exists(v2file)));

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cd_svd_driver<T>(sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells,
                     shell_tile_map, ccsd_restart, cholfile);
  free_tensors(lcao);
  total_orbitals = sys_data.nmo;

  // if(ccsd_options.writev) ccsd_options.writet = true;

  TiledIndexSpace N = MO("all");

  std::vector<T>         p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  if(is_rhf)
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors_cs(
      ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);
  else
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors(
      ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

  if(ccsd_restart) {
    read_from_disk(d_f1, f1file);
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    read_from_disk(cholVpr, v2file);
    ec.pg().barrier();
    p_evl_sorted = tamm::diagonal(d_f1);
  }

  else if(ccsd_options.writet) {
    // fs::remove_all(files_dir);
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);

    write_to_disk(d_f1, f1file);
    write_to_disk(cholVpr, v2file);

    if(rank == 0) {
      std::ofstream out(cholfile, std::ios::out);
      if(!out) cerr << "Error opening file " << cholfile << endl;
      out << chol_count << std::endl;
      out.close();
    }
  }

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

  // std::string fullV2file = files_prefix+".fullV2";
  // t1file = files_prefix+".fullT1amp";
  // t2file = files_prefix+".fullT2amp";

  bool computeTData = true;
  // if(ccsd_options.writev)
  //     computeTData = computeTData && !fs::exists(fullV2file)
  //             && !fs::exists(t1file) && !fs::exists(t2file);

  if(computeTData && is_rhf) setup_full_t1t2(ec, MO, dt1_full, dt2_full);

  double residual = 0, corr_energy = 0;

  if(is_rhf) {
    if(ccsd_restart) {
      if(subcomm != MPI_COMM_NULL) {
        const int ppn = GA_Cluster_nprocs(0);
        if(rank == 0)
          std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                    << std::endl;
        std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
          sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
      }
      ec.pg().barrier();
    }
    else {
      std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
        sys_data, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
        p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
    }
  }
  else {
    if(ccsd_restart) {
      if(subcomm != MPI_COMM_NULL) {
        const int ppn = GA_Cluster_nprocs(0);
        if(rank == 0)
          std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                    << std::endl;
        std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
          sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
      }
      ec.pg().barrier();
    }
    else {
      std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
        sys_data, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
        p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
    }
  }

  if(computeTData && is_rhf) {
    // if(ccsd_options.writev) {
    //     write_to_disk(dt1_full,t1file);
    //     write_to_disk(dt2_full,t2file);
    //     free_tensors(dt1_full, dt2_full);
    // }
    free_tensors(d_t1, d_t2); // free t1_aa, t2_abab
    d_t1 = dt1_full;          // GFCC uses full T1,T2
    d_t2 = dt2_full;          // GFCC uses full T1,T2
  }

  ccsd_stats(ec, hf_energy, residual, corr_energy, ccsd_options.threshold);

  if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    if(rank == 0) {
      std::ofstream out(ccsdstatus, std::ios::out);
      if(!out) cerr << "Error opening file " << ccsdstatus << endl;
      out << 1 << std::endl;
      out.close();
    }
  }

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    if(is_rhf)
      std::cout << std::endl
                << "Time taken for Closed Shell Cholesky CCSD: " << std::fixed
                << std::setprecision(2) << ccsd_time << " secs" << std::endl;
    else
      std::cout << std::endl
                << "Time taken for Open Shell Cholesky CCSD: " << std::fixed << std::setprecision(2)
                << ccsd_time << " secs" << std::endl;
  }

  cc_print(sys_data, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }

  ec.flush_and_sync();

  //////////////////////////
  //                      //
  // Start GFCCSD Routine //
  //                      //
  //////////////////////////
  cc_t1 = std::chrono::high_resolution_clock::now();

  const TAMM_SIZE nocc = sys_data.nocc;
  const TAMM_SIZE nvir = sys_data.nvir;
  const TAMM_SIZE noa  = sys_data.n_occ_alpha;
  const TAMM_SIZE nob  = sys_data.n_occ_beta;
  const TAMM_SIZE nva  = sys_data.n_vir_alpha;
  const TAMM_SIZE nvb  = sys_data.n_vir_beta;

  ndiis                = ccsd_options.gf_ndiis;
  ngmres               = ccsd_options.gf_ngmres;
  gf_eta               = ccsd_options.gf_eta;
  gf_profile           = ccsd_options.gf_profile;
  gf_maxiter           = ccsd_options.gf_maxiter;
  gf_threshold         = ccsd_options.gf_threshold;
  gf_lshift            = ccsd_options.gf_lshift;
  gf_preconditioning   = ccsd_options.gf_preconditioning;
  omega_min_ip         = ccsd_options.gf_omega_min_ip;
  omega_max_ip         = ccsd_options.gf_omega_max_ip;
  lomega_min_ip        = ccsd_options.gf_omega_min_ip_e;
  lomega_max_ip        = ccsd_options.gf_omega_max_ip_e;
  omega_min_ea         = ccsd_options.gf_omega_min_ea;
  omega_max_ea         = ccsd_options.gf_omega_max_ea;
  lomega_min_ea        = ccsd_options.gf_omega_min_ea_e;
  lomega_max_ea        = ccsd_options.gf_omega_max_ea_e;
  omega_delta          = ccsd_options.gf_omega_delta;
  omega_delta_e        = ccsd_options.gf_omega_delta_e;
  gf_nprocs_poi        = ccsd_options.gf_nprocs_poi;
  gf_orbitals          = ccsd_options.gf_orbitals;
  gf_damping_factor    = ccsd_options.gf_damping_factor;
  gf_extrapolate_level = ccsd_options.gf_extrapolate_level;
  gf_analyze_level     = ccsd_options.gf_analyze_level;
  gf_analyze_num_omega = ccsd_options.gf_analyze_num_omega;
  omega_npts_ip        = std::ceil((omega_max_ip - omega_min_ip) / omega_delta + 1);
  lomega_npts_ip       = std::ceil((lomega_max_ip - lomega_min_ip) / omega_delta_e + 1);
  omega_npts_ea        = std::ceil((omega_max_ea - omega_min_ea) / omega_delta + 1);
  lomega_npts_ea       = std::ceil((lomega_max_ea - lomega_min_ea) / omega_delta_e + 1);

  if(ec.pg().size() < gf_nprocs_poi)
    tamm_terminate(
      "ERROR: gf_nprocs_poi cannot be greater than total number of mpi ranks provided");

  const int gf_p_oi = ccsd_options.gf_p_oi_range;

  if(gf_p_oi == 1) p_oi = nocc;
  else p_oi = nocc + nvir;

  int level = 1;

  if(rank == 0) ccsd_options.print();

  using ComplexTensor = Tensor<std::complex<T>>;

  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(obtiles, otiles)};
  v_beta  = {MO("virt"), range(vbtiles, vtiles)};

  auto [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = MO.labels<10>("virt");
  auto [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10] = MO.labels<10>("occ");

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h5_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] =
    o_alpha.labels<10>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h5_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] =
    o_beta.labels<10>("all");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va, p7_va, p8_va, p9_va, p10_va] =
    v_alpha.labels<10>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p5_vb, p6_vb, p7_vb, p8_vb, p9_vb, p10_vb] =
    v_beta.labels<10>("all");

  Scheduler sch{ec};

  if(rank == 0) cout << endl << "#occupied, #virtual = " << nocc << ", " << nvir << endl;
  std::vector<T> p_evl_sorted_occ(nocc);
  std::vector<T> p_evl_sorted_virt(nvir);
  std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + nocc, p_evl_sorted_occ.begin());
  std::copy(p_evl_sorted.begin() + nocc, p_evl_sorted.end(), p_evl_sorted_virt.begin());

  // START SF2

  std::vector<T> omega_space_ip;
  std::vector<T> omega_space_ea;

  if(ccsd_options.gf_ip) {
    for(int64_t ni = 0; ni < omega_npts_ip; ni++) {
      T omega_tmp = omega_min_ip + ni * omega_delta;
      omega_space_ip.push_back(omega_tmp);
    }
    if(rank == 0)
      cout << "Freq. space (before doing MOR): " << std::fixed << std::setprecision(2)
           << omega_space_ip << endl;
  }
  if(ccsd_options.gf_ea) {
    for(int64_t ni = 0; ni < omega_npts_ea; ni++) {
      T omega_tmp = omega_min_ea + ni * omega_delta;
      omega_space_ea.push_back(omega_tmp);
    }
    if(rank == 0)
      cout << "Freq. space (before doing MOR): " << std::fixed << std::setprecision(2)
           << omega_space_ea << endl;
  }

  //#define MOR 1
  //#if MOR

  auto   restart_time_end   = std::chrono::high_resolution_clock::now();
  double total_restart_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                (restart_time_end - restart_time_start))
                                .count();
  if(rank == 0)
    std::cout << std::endl
              << "GFCC: Time taken pre-restart: " << total_restart_time << " secs" << std::endl;

  using Complex2DMatrix =
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  auto inter_read_start = std::chrono::high_resolution_clock::now();

  size_t nptsi = 0;

  std::vector<bool> omega_ip_conv_a(omega_npts_ip, false);
  std::vector<bool> omega_ip_conv_b(omega_npts_ip, false);
  std::vector<bool> omega_ea_conv_a(omega_npts_ea, false);
  std::vector<bool> omega_ea_conv_b(omega_npts_ea, false);
  std::vector<T>    omega_ip_A0(omega_npts_ip, UINT_MAX);
  std::vector<T>    omega_ea_A0(omega_npts_ea, UINT_MAX);

  std::vector<T> omega_extra;
  std::vector<T> omega_extra_finished;

  ///////////////////////////
  //                       //
  // Compute intermediates //
  //                       //
  ///////////////////////////

  if(!fs::exists(files_dir)) { fs::create_directories(files_dir); }

  Tensor<T> d_t1_a{v_alpha, o_alpha};
  Tensor<T> d_t1_b{v_beta, o_beta};
  Tensor<T> d_t2_aaaa{v_alpha, v_alpha, o_alpha, o_alpha};
  Tensor<T> d_t2_bbbb{v_beta, v_beta, o_beta, o_beta};
  Tensor<T> d_t2_abab{v_alpha, v_beta, o_alpha, o_beta};
  Tensor<T> cholOO_a{o_alpha, o_alpha, CI};
  Tensor<T> cholOO_b{o_beta, o_beta, CI};
  Tensor<T> cholOV_a{o_alpha, v_alpha, CI};
  Tensor<T> cholOV_b{o_beta, v_beta, CI};
  Tensor<T> cholVV_a{v_alpha, v_alpha, CI};
  Tensor<T> cholVV_b{v_beta, v_beta, CI};
  Tensor<T> v2ijab_aaaa{o_alpha, o_alpha, v_alpha, v_alpha};
  Tensor<T> v2ijab_bbbb{o_beta, o_beta, v_beta, v_beta};
  Tensor<T> v2ijab_abab{o_alpha, o_beta, v_alpha, v_beta};

  Tensor<T> v2ijab{{O, O, V, V}, {2, 2}};
  Tensor<T> v2ijka{{O, O, O, V}, {2, 2}};
  Tensor<T> v2ijkl{{O, O, O, O}, {2, 2}};
  Tensor<T> v2iajb{{O, V, O, V}, {2, 2}};
  Tensor<T> v2iabc{{O, V, V, V}, {2, 2}};

  std::string d_t1_a_file      = files_prefix + ".d_t1_a";
  std::string d_t1_b_file      = files_prefix + ".d_t1_b";
  std::string d_t2_aaaa_file   = files_prefix + ".d_t2_aaaa";
  std::string d_t2_bbbb_file   = files_prefix + ".d_t2_bbbb";
  std::string d_t2_abab_file   = files_prefix + ".d_t2_abab";
  std::string cholOO_a_file    = files_prefix + ".cholOO_a";
  std::string cholOO_b_file    = files_prefix + ".cholOO_b";
  std::string cholOV_a_file    = files_prefix + ".cholOV_a";
  std::string cholOV_b_file    = files_prefix + ".cholOV_b";
  std::string cholVV_a_file    = files_prefix + ".cholVV_a";
  std::string cholVV_b_file    = files_prefix + ".cholVV_b";
  std::string v2ijab_aaaa_file = files_prefix + ".v2ijab_aaaa";
  std::string v2ijab_bbbb_file = files_prefix + ".v2ijab_bbbb";
  std::string v2ijab_abab_file = files_prefix + ".v2ijab_abab";

  sch
    .allocate(d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, cholOO_a, cholOO_b, cholOV_a,
              cholOV_b, cholVV_a, cholVV_b, v2ijab_aaaa, v2ijab_bbbb, v2ijab_abab, v2ijab, v2ijka,
              v2iajb)
    .execute();

  auto gfst_start = std::chrono::high_resolution_clock::now();

  // clang-format off
  sch ( v2ijka(h1,h2,h3,p1)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
      ( v2ijka(h1,h2,h3,p1)     +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) )
      ( v2iajb(h1,p1,h2,p2)      =   1.0 * cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
      ( v2iajb(h1,p1,h2,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )
      ( v2ijab(h1,h2,p1,p2)      =   1.0 * cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
      ( v2ijab(h1,h2,p1,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) );
  // clang-format on

  sch.execute(sch.ec().exhw());

  auto   gfst_end = std::chrono::high_resolution_clock::now();
  double gfcc_restart_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((gfst_end - gfst_start)).count();
  if(rank == 0)
    std::cout << std::endl
              << " -- GFCC: Time to compute v2 blocks: " << gfcc_restart_time << " secs"
              << std::endl;

  gfst_start           = std::chrono::high_resolution_clock::now();
  std::string rw_inter = "computing and writing";

  if(fs::exists(d_t1_a_file) && fs::exists(d_t1_b_file) && fs::exists(d_t2_aaaa_file) &&
     fs::exists(d_t2_bbbb_file) && fs::exists(d_t2_abab_file) && fs::exists(cholOO_a_file) &&
     fs::exists(cholOO_b_file) && fs::exists(cholOV_a_file) && fs::exists(cholOV_b_file) &&
     fs::exists(cholVV_a_file) && fs::exists(cholVV_b_file) && fs::exists(v2ijab_aaaa_file) &&
     fs::exists(v2ijab_bbbb_file) && fs::exists(v2ijab_abab_file) && ccsd_options.gf_restart) {
    //  read_from_disk(d_t1_a,d_t1_a_file);
    //  read_from_disk(d_t1_b,d_t1_b_file);
    //  read_from_disk(d_t2_aaaa,d_t2_aaaa_file);
    //  read_from_disk(d_t2_bbbb,d_t2_bbbb_file);
    //  read_from_disk(d_t2_abab,d_t2_abab_file);
    //  read_from_disk(cholOO_a,cholOO_a_file);
    //  read_from_disk(cholOO_b,cholOO_b_file);
    //  read_from_disk(cholOV_a,cholOV_a_file);
    //  read_from_disk(cholOV_b,cholOV_b_file);
    //  read_from_disk(cholVV_a,cholVV_a_file);
    //  read_from_disk(cholVV_b,cholVV_b_file);
    //  read_from_disk(v2ijab_aaaa,v2ijab_aaaa_file);
    //  read_from_disk(v2ijab_bbbb,v2ijab_bbbb_file);
    //  read_from_disk(v2ijab_abab,v2ijab_abab_file);
    rw_inter                          = "reading";
    std::vector<Tensor<T>>   rtensors = {d_t1_a,   d_t1_b,      d_t2_aaaa,   d_t2_bbbb,  d_t2_abab,
                                         cholOO_a, cholOO_b,    cholOV_a,    cholOV_b,   cholVV_a,
                                         cholVV_b, v2ijab_aaaa, v2ijab_bbbb, v2ijab_abab};
    std::vector<std::string> rtfnames = {
      d_t1_a_file,   d_t1_b_file,      d_t2_aaaa_file,   d_t2_bbbb_file,  d_t2_abab_file,
      cholOO_a_file, cholOO_b_file,    cholOV_a_file,    cholOV_b_file,   cholVV_a_file,
      cholVV_b_file, v2ijab_aaaa_file, v2ijab_bbbb_file, v2ijab_abab_file};
    read_from_disk_group(ec, rtensors, rtfnames);
  }
  else {
    // clang-format off

    #if GF_IN_SG
    if(subcomm != MPI_COMM_NULL) {
      sub_sch
    #else
      sch
    #endif
        // ( d_t2() = 0 ) // CCS
        ( d_t1_a(p1_va,h3_oa)  =  1.0 * d_t1(p1_va,h3_oa)                            )
        ( d_t1_b(p1_vb,h3_ob)  =  1.0 * d_t1(p1_vb,h3_ob)                            )
        ( d_t2_aaaa(p1_va,p2_va,h3_oa,h4_oa)  =  1.0 * d_t2(p1_va,p2_va,h3_oa,h4_oa) )
        ( d_t2_abab(p1_va,p2_vb,h3_oa,h4_ob)  =  1.0 * d_t2(p1_va,p2_vb,h3_oa,h4_ob) )
        ( d_t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob)  =  1.0 * d_t2(p1_vb,p2_vb,h3_ob,h4_ob) )
        ( cholOO_a(h1_oa,h2_oa,cind)  =  1.0 * cholVpr(h1_oa,h2_oa,cind)             )
        ( cholOO_b(h1_ob,h2_ob,cind)  =  1.0 * cholVpr(h1_ob,h2_ob,cind)             )
        ( cholOV_a(h1_oa,p1_va,cind)  =  1.0 * cholVpr(h1_oa,p1_va,cind)             )
        ( cholOV_b(h1_ob,p1_vb,cind)  =  1.0 * cholVpr(h1_ob,p1_vb,cind)             )
        ( cholVV_a(p1_va,p2_va,cind)  =  1.0 * cholVpr(p1_va,p2_va,cind)             )
        ( cholVV_b(p1_vb,p2_vb,cind)  =  1.0 * cholVpr(p1_vb,p2_vb,cind)             )
        ( v2ijab_aaaa(h1_oa,h2_oa,p1_va,p2_va)  =  1.0 * v2ijab(h1_oa,h2_oa,p1_va,p2_va) )
        ( v2ijab_abab(h1_oa,h2_ob,p1_va,p2_vb)  =  1.0 * v2ijab(h1_oa,h2_ob,p1_va,p2_vb) )
        ( v2ijab_bbbb(h1_ob,h2_ob,p1_vb,p2_vb)  =  1.0 * v2ijab(h1_ob,h2_ob,p1_vb,p2_vb) )
        .execute();
    #if GF_IN_SG
    }
    #endif
    // clang-format on

    // write_to_disk(d_t1_a,d_t1_a_file);
    // write_to_disk(d_t1_b,d_t1_b_file);
    // write_to_disk(d_t2_aaaa,d_t2_aaaa_file);
    // write_to_disk(d_t2_bbbb,d_t2_bbbb_file);
    // write_to_disk(d_t2_abab,d_t2_abab_file);
    // write_to_disk(cholOO_a,cholOO_a_file);
    // write_to_disk(cholOO_b,cholOO_b_file);
    // write_to_disk(cholOV_a,cholOV_a_file);
    // write_to_disk(cholOV_b,cholOV_b_file);
    // write_to_disk(cholVV_a,cholVV_a_file);
    // write_to_disk(cholVV_b,cholVV_b_file);
    // write_to_disk(v2ijab_aaaa,v2ijab_aaaa_file);
    // write_to_disk(v2ijab_bbbb,v2ijab_bbbb_file);
    // write_to_disk(v2ijab_abab,v2ijab_abab_file);
    std::vector<Tensor<T>>   rtensors = {d_t1_a,   d_t1_b,      d_t2_aaaa,   d_t2_bbbb,  d_t2_abab,
                                         cholOO_a, cholOO_b,    cholOV_a,    cholOV_b,   cholVV_a,
                                         cholVV_b, v2ijab_aaaa, v2ijab_bbbb, v2ijab_abab};
    std::vector<std::string> rtfnames = {
      d_t1_a_file,   d_t1_b_file,      d_t2_aaaa_file,   d_t2_bbbb_file,  d_t2_abab_file,
      cholOO_a_file, cholOO_b_file,    cholOV_a_file,    cholOV_b_file,   cholVV_a_file,
      cholVV_b_file, v2ijab_aaaa_file, v2ijab_bbbb_file, v2ijab_abab_file};
    write_to_disk_group(ec, rtensors, rtfnames);
  }

  ec.pg().barrier();

  gfst_end = std::chrono::high_resolution_clock::now();
  gfcc_restart_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((gfst_end - gfst_start)).count();
  if(rank == 0)
    std::cout << " -- GFCC: Time for " << rw_inter
              << " spin-explicit t1,t2,v2 tensors: " << gfcc_restart_time << " secs" << std::endl;

  gfst_start = std::chrono::high_resolution_clock::now();
  rw_inter   = "computing and writing";

  if(ccsd_options.gf_ip) {
    std::string t2v2_o_file       = files_prefix + ".t2v2_o";
    std::string lt12_o_a_file     = files_prefix + ".lt12_o_a";
    std::string lt12_o_b_file     = files_prefix + ".lt12_o_b";
    std::string ix1_1_1_a_file    = files_prefix + ".ix1_1_1_a";    //
    std::string ix1_1_1_b_file    = files_prefix + ".ix1_1_1_b";    //
    std::string ix2_1_aaaa_file   = files_prefix + ".ix2_1_aaaa";   //
    std::string ix2_1_abab_file   = files_prefix + ".ix2_1_abab";   //
    std::string ix2_1_bbbb_file   = files_prefix + ".ix2_1_bbbb";   //
    std::string ix2_1_baba_file   = files_prefix + ".ix2_1_baba";   //
    std::string ix2_2_a_file      = files_prefix + ".ix2_2_a";      //
    std::string ix2_2_b_file      = files_prefix + ".ix2_2_b";      //
    std::string ix2_3_a_file      = files_prefix + ".ix2_3_a";      //
    std::string ix2_3_b_file      = files_prefix + ".ix2_3_b";      //
    std::string ix2_4_aaaa_file   = files_prefix + ".ix2_4_aaaa";   //
    std::string ix2_4_abab_file   = files_prefix + ".ix2_4_abab";   //
    std::string ix2_4_bbbb_file   = files_prefix + ".ix2_4_bbbb";   //
    std::string ix2_5_aaaa_file   = files_prefix + ".ix2_5_aaaa";   //
    std::string ix2_5_abba_file   = files_prefix + ".ix2_5_abba";   //
    std::string ix2_5_abab_file   = files_prefix + ".ix2_5_abab";   //
    std::string ix2_5_bbbb_file   = files_prefix + ".ix2_5_bbbb";   //
    std::string ix2_5_baab_file   = files_prefix + ".ix2_5_baab";   //
    std::string ix2_5_baba_file   = files_prefix + ".ix2_5_baba";   //
    std::string ix2_6_2_a_file    = files_prefix + ".ix2_6_2_a";    //
    std::string ix2_6_2_b_file    = files_prefix + ".ix2_6_2_b";    //
    std::string ix2_6_3_aaaa_file = files_prefix + ".ix2_6_3_aaaa"; //
    std::string ix2_6_3_abba_file = files_prefix + ".ix2_6_3_abba"; //
    std::string ix2_6_3_abab_file = files_prefix + ".ix2_6_3_abab"; //
    std::string ix2_6_3_baab_file = files_prefix + ".ix2_6_3_baab"; //
    std::string ix2_6_3_bbbb_file = files_prefix + ".ix2_6_3_bbbb"; //
    std::string ix2_6_3_baba_file = files_prefix + ".ix2_6_3_baba"; //

    t2v2_o       = Tensor<T>{{O, O}, {1, 1}};
    lt12_o_a     = Tensor<T>{o_alpha, o_alpha};
    lt12_o_b     = Tensor<T>{o_beta, o_beta};
    ix1_1_1_a    = Tensor<T>{o_alpha, v_alpha};
    ix1_1_1_b    = Tensor<T>{o_beta, v_beta};
    ix2_1_aaaa   = Tensor<T>{o_alpha, v_alpha, o_alpha, o_alpha};
    ix2_1_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, o_beta};
    ix2_1_abab   = Tensor<T>{o_alpha, v_beta, o_alpha, o_beta};
    ix2_1_baba   = Tensor<T>{o_beta, v_alpha, o_beta, o_alpha};
    ix2_2_a      = Tensor<T>{o_alpha, o_alpha};
    ix2_2_b      = Tensor<T>{o_beta, o_beta};
    ix2_3_a      = Tensor<T>{v_alpha, v_alpha};
    ix2_3_b      = Tensor<T>{v_beta, v_beta};
    ix2_4_aaaa   = Tensor<T>{o_alpha, o_alpha, o_alpha, o_alpha};
    ix2_4_abab   = Tensor<T>{o_alpha, o_beta, o_alpha, o_beta};
    ix2_4_bbbb   = Tensor<T>{o_beta, o_beta, o_beta, o_beta};
    ix2_5_aaaa   = Tensor<T>{o_alpha, v_alpha, o_alpha, v_alpha};
    ix2_5_abba   = Tensor<T>{o_alpha, v_beta, o_beta, v_alpha};
    ix2_5_abab   = Tensor<T>{o_alpha, v_beta, o_alpha, v_beta};
    ix2_5_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, v_beta};
    ix2_5_baab   = Tensor<T>{o_beta, v_alpha, o_alpha, v_beta};
    ix2_5_baba   = Tensor<T>{o_beta, v_alpha, o_beta, v_alpha};
    ix2_6_2_a    = Tensor<T>{o_alpha, v_alpha};
    ix2_6_2_b    = Tensor<T>{o_beta, v_beta};
    ix2_6_3_aaaa = Tensor<T>{o_alpha, o_alpha, o_alpha, v_alpha};
    ix2_6_3_abba = Tensor<T>{o_alpha, o_beta, o_beta, v_alpha};
    ix2_6_3_abab = Tensor<T>{o_alpha, o_beta, o_alpha, v_beta};
    ix2_6_3_baab = Tensor<T>{o_beta, o_alpha, o_alpha, v_beta};
    ix2_6_3_bbbb = Tensor<T>{o_beta, o_beta, o_beta, v_beta};
    ix2_6_3_baba = Tensor<T>{o_beta, o_alpha, o_beta, v_alpha};

    Tensor<T> lt12_o{{O, O}, {1, 1}};
    Tensor<T> ix1_1_1{{O, V}, {1, 1}};
    Tensor<T> ix2_1_1{{O, V, O, V}, {2, 2}};
    Tensor<T> ix2_1_3{{O, O, O, V}, {2, 2}};
    Tensor<T> ix2_1_temp{{O, V, O, O}, {2, 2}};
    Tensor<T> ix2_1{{O, V, O, O}, {2, 2}};
    Tensor<T> ix2_2{{O, O}, {1, 1}};
    Tensor<T> ix2_3{{V, V}, {1, 1}};
    Tensor<T> ix2_4_1{{O, O, O, V}, {2, 2}};
    Tensor<T> ix2_4_temp{{O, O, O, O}, {2, 2}};
    Tensor<T> ix2_4{{O, O, O, O}, {2, 2}};
    Tensor<T> ix2_5{{O, V, O, V}, {2, 2}};
    Tensor<T> ix2_6_2{{O, V}, {1, 1}};
    Tensor<T> ix2_6_3{{O, O, O, V}, {2, 2}};

    sch
      .allocate(t2v2_o, lt12_o_a, lt12_o_b, ix1_1_1_a, ix1_1_1_b, ix2_1_aaaa, ix2_1_abab,
                ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab,
                ix2_4_bbbb, ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb,
                ix2_6_3_baab, ix2_6_3_baba)
      .execute();

    if(fs::exists(t2v2_o_file) && fs::exists(lt12_o_a_file) && fs::exists(lt12_o_b_file) &&
       fs::exists(ix1_1_1_a_file) && fs::exists(ix1_1_1_b_file) && fs::exists(ix2_1_aaaa_file) &&
       fs::exists(ix2_1_bbbb_file) && fs::exists(ix2_1_abab_file) && fs::exists(ix2_1_baba_file) &&
       fs::exists(ix2_2_a_file) && fs::exists(ix2_2_b_file) && fs::exists(ix2_3_a_file) &&
       fs::exists(ix2_3_b_file) && fs::exists(ix2_4_aaaa_file) && fs::exists(ix2_4_abab_file) &&
       fs::exists(ix2_4_bbbb_file) && fs::exists(ix2_5_aaaa_file) && fs::exists(ix2_5_abba_file) &&
       fs::exists(ix2_5_abab_file) && fs::exists(ix2_5_bbbb_file) && fs::exists(ix2_5_baab_file) &&
       fs::exists(ix2_5_baba_file) && fs::exists(ix2_6_2_a_file) && fs::exists(ix2_6_2_b_file) &&
       fs::exists(ix2_6_3_aaaa_file) && fs::exists(ix2_6_3_abba_file) &&
       fs::exists(ix2_6_3_abab_file) && fs::exists(ix2_6_3_bbbb_file) &&
       fs::exists(ix2_6_3_baab_file) && fs::exists(ix2_6_3_baba_file) && ccsd_options.gf_restart) {
      // read_from_disk(t2v2_o,t2v2_o_file);
      // read_from_disk(lt12_o_a,lt12_o_a_file);
      // read_from_disk(lt12_o_b,lt12_o_b_file);
      // read_from_disk(ix1_1_1_a,ix1_1_1_a_file);
      // read_from_disk(ix1_1_1_b,ix1_1_1_b_file);
      // read_from_disk(ix2_1_aaaa,ix2_1_aaaa_file);
      // read_from_disk(ix2_1_bbbb,ix2_1_bbbb_file);
      // read_from_disk(ix2_1_abab,ix2_1_abab_file);
      // read_from_disk(ix2_1_baba,ix2_1_baba_file);
      // read_from_disk(ix2_2_a,ix2_2_a_file);
      // read_from_disk(ix2_2_b,ix2_2_b_file);
      // read_from_disk(ix2_3_a,ix2_3_a_file);
      // read_from_disk(ix2_3_b,ix2_3_b_file);
      // read_from_disk(ix2_4_aaaa,ix2_4_aaaa_file);
      // read_from_disk(ix2_4_abab,ix2_4_abab_file);
      // read_from_disk(ix2_4_bbbb,ix2_4_bbbb_file);
      // read_from_disk(ix2_5_aaaa,ix2_5_aaaa_file);
      // read_from_disk(ix2_5_abba,ix2_5_abba_file);
      // read_from_disk(ix2_5_abab,ix2_5_abab_file);
      // read_from_disk(ix2_5_bbbb,ix2_5_bbbb_file);
      // read_from_disk(ix2_5_baab,ix2_5_baab_file);
      // read_from_disk(ix2_5_baba,ix2_5_baba_file);
      // read_from_disk(ix2_6_2_a,ix2_6_2_a_file);
      // read_from_disk(ix2_6_2_b,ix2_6_2_b_file);
      // read_from_disk(ix2_6_3_aaaa,ix2_6_3_aaaa_file);
      // read_from_disk(ix2_6_3_abba,ix2_6_3_abba_file);
      // read_from_disk(ix2_6_3_abab,ix2_6_3_abab_file);
      // read_from_disk(ix2_6_3_bbbb,ix2_6_3_bbbb_file);
      // read_from_disk(ix2_6_3_baab,ix2_6_3_baab_file);
      // read_from_disk(ix2_6_3_baba,ix2_6_3_baba_file);
      rw_inter                        = "reading";
      std::vector<Tensor<T>> rtensors = {
        t2v2_o,       lt12_o_a,     lt12_o_b,     ix1_1_1_a,    ix1_1_1_b,    ix2_1_aaaa,
        ix2_1_bbbb,   ix2_1_abab,   ix2_1_baba,   ix2_2_a,      ix2_2_b,      ix2_3_a,
        ix2_3_b,      ix2_4_aaaa,   ix2_4_abab,   ix2_4_bbbb,   ix2_5_aaaa,   ix2_5_abba,
        ix2_5_abab,   ix2_5_bbbb,   ix2_5_baab,   ix2_5_baba,   ix2_6_2_a,    ix2_6_2_b,
        ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba};
      std::vector<std::string> rtfnames = {
        t2v2_o_file,       lt12_o_a_file,     lt12_o_b_file,     ix1_1_1_a_file,
        ix1_1_1_b_file,    ix2_1_aaaa_file,   ix2_1_bbbb_file,   ix2_1_abab_file,
        ix2_1_baba_file,   ix2_2_a_file,      ix2_2_b_file,      ix2_3_a_file,
        ix2_3_b_file,      ix2_4_aaaa_file,   ix2_4_abab_file,   ix2_4_bbbb_file,
        ix2_5_aaaa_file,   ix2_5_abba_file,   ix2_5_abab_file,   ix2_5_bbbb_file,
        ix2_5_baab_file,   ix2_5_baba_file,   ix2_6_2_a_file,    ix2_6_2_b_file,
        ix2_6_3_aaaa_file, ix2_6_3_abba_file, ix2_6_3_abab_file, ix2_6_3_bbbb_file,
        ix2_6_3_baab_file, ix2_6_3_baba_file};
      read_from_disk_group(ec, rtensors, rtfnames);
    }
    else {
      // clang-format off
    
      #if GF_IN_SG
      if(subcomm != MPI_COMM_NULL) {
        sub_sch
      #else
        sch
      #endif
          .allocate(lt12_o,
                    ix1_1_1,
                    ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                    ix2_2,
                    ix2_3,
                    ix2_4_1,ix2_4_temp,ix2_4,
                    ix2_5,
                    ix2_6_2,ix2_6_3,
                    v2ijkl,v2iabc)
          ( v2ijkl(h1,h2,h3,h4)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
          ( v2ijkl(h1,h2,h3,h4)     +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) )
          ( v2iabc(h1,p1,p2,p3)      =   1.0 * cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
          ( v2iabc(h1,p1,p2,p3)     +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) )
          ( t2v2_o(h1,h2)            =   0.5 * d_t2(p1,p2,h1,h3) * v2ijab(h3,h2,p1,p2)   )
          ( lt12_o(h1_oa,h3_oa)      =   0.5 * d_t2(p1_va,p2_va,h1_oa,h2_oa) * d_t2(p1_va,p2_va,h3_oa,h2_oa) )
          ( lt12_o(h1_oa,h3_oa)     +=   1.0 * d_t2(p1_va,p2_vb,h1_oa,h2_ob) * d_t2(p1_va,p2_vb,h3_oa,h2_ob) )
          ( lt12_o(h1_oa,h2_oa)     +=   1.0 * d_t1(p1_va,h1_oa) * d_t1(p1_va,h2_oa)                         )
          ( lt12_o(h1_ob,h3_ob)      =   0.5 * d_t2(p1_vb,p2_vb,h1_ob,h2_ob) * d_t2(p1_vb,p2_vb,h3_ob,h2_ob) )
          ( lt12_o(h1_ob,h3_ob)     +=   1.0 * d_t2(p2_va,p1_vb,h2_oa,h1_ob) * d_t2(p2_va,p1_vb,h2_oa,h3_ob) )
          ( lt12_o(h1_ob,h2_ob)     +=   1.0 * d_t1(p1_vb,h1_ob) * d_t1(p1_vb,h2_ob)                         )
          (   ix1_1_1(h6,p7)         =   1.0 * d_f1(h6,p7)                               )
          (   ix1_1_1(h6,p7)        +=   1.0 * d_t1(p4,h5) * v2ijab(h5,h6,p4,p7)         )
          ( ix2_1(h9,p3,h1,h2)       =   1.0 * v2ijka(h1,h2,h9,p3)                       )
          (   ix2_1_1(h9,p3,h1,p5)   =   1.0 * v2iajb(h9,p3,h1,p5)                       )
          ( ix2_5(h7,p3,h1,p8)       =   1.0 * d_t1(p5,h1) * v2iabc(h7,p3,p5,p8)         ) //O2V3
          (   ix2_1_1(h9,p3,h1,p5)  +=  -0.5 * ix2_5(h9,p3,h1,p5)                        ) //O2V3
          ( ix2_5(h7,p3,h1,p8)      +=   1.0 * v2iajb(h7,p3,h1,p8)                       )
          ( ix2_1_temp(h9,p3,h1,h2)  =   1.0 * d_t1(p5,h1) * ix2_1_1(h9,p3,h2,p5)        ) //O3V2
          ( ix2_1(h9,p3,h1,h2)      +=  -1.0 * ix2_1_temp(h9,p3,h1,h2)                   )
          ( ix2_1(h9,p3,h2,h1)      +=   1.0 * ix2_1_temp(h9,p3,h1,h2)                   ) 
          ( ix2_1(h9,p3,h1,h2)      +=  -1.0 * d_t2(p3,p8,h1,h2) * ix1_1_1(h9,p8)        ) //O3V2
          (   ix2_1_3(h6,h9,h1,p5)   =   1.0 * v2ijka(h6,h9,h1,p5)                       )
          (   ix2_1_3(h6,h9,h1,p5)  +=  -1.0 * d_t1(p7,h1) * v2ijab(h6,h9,p5,p7)         ) //O3V2
          ( ix2_1_temp(h9,p3,h1,h2)  =   1.0 * d_t2(p3,p5,h1,h6) * ix2_1_3(h6,h9,h2,p5)  ) //O4V2
          ( ix2_1(h9,p3,h1,h2)      +=   1.0 * ix2_1_temp(h9,p3,h1,h2)                   )
          ( ix2_1(h9,p3,h2,h1)      +=  -1.0 * ix2_1_temp(h9,p3,h1,h2)                   ) 
          ( ix2_1(h9,p3,h1,h2)      +=   0.5 * d_t2(p5,p6,h1,h2) * v2iabc(h9,p3,p5,p6)   ) //O3V3
          ( ix2_2(h8,h1)             =   1.0 * d_f1(h8,h1)                               )
          ( ix2_2(h8,h1)            +=   1.0 * d_t1(p9,h1) * ix1_1_1(h8,p9)              )
          ( ix2_2(h8,h1)            +=  -1.0 * d_t1(p5,h6) * v2ijka(h6,h8,h1,p5)         )
          ( ix2_2(h8,h1)            +=  -0.5 * d_t2(p5,p6,h1,h7) * v2ijab(h7,h8,p5,p6)   ) //O3V2
          ( ix2_3(p3,p8)             =   1.0 * d_f1(p3,p8)                               )
          ( ix2_3(p3,p8)            +=   1.0 * d_t1(p5,h6) * v2iabc(h6,p3,p5,p8)         ) 
          ( ix2_3(p3,p8)            +=   0.5 * d_t2(p3,p5,h6,h7) * v2ijab(h6,h7,p5,p8)   ) //O2V3
          ( ix2_4(h9,h10,h1,h2)      =   1.0 * v2ijkl(h9,h10,h1,h2)                      )
          (   ix2_4_1(h9,h10,h1,p5)  =   1.0 * v2ijka(h9,h10,h1,p5)                      )
          (   ix2_4_1(h9,h10,h1,p5) +=  -0.5 * d_t1(p6,h1) * v2ijab(h9,h10,p5,p6)        ) //O3V2
          ( ix2_4_temp(h9,h10,h1,h2) =   1.0 * d_t1(p5,h1) * ix2_4_1(h9,h10,h2,p5)       ) //O4V
          ( ix2_4(h9,h10,h1,h2)     +=  -1.0 * ix2_4_temp(h9,h10,h1,h2)                  )
          ( ix2_4(h9,h10,h2,h1)     +=   1.0 * ix2_4_temp(h9,h10,h1,h2)                  ) 
          ( ix2_4(h9,h10,h1,h2)     +=   0.5 * d_t2(p5,p6,h1,h2) * v2ijab(h9,h10,p5,p6)  ) //O4V2
          ( ix2_6_2(h10,p5)          =   1.0 * d_f1(h10,p5)                              )
          ( ix2_6_2(h10,p5)         +=   1.0 * d_t1(p6,h7) * v2ijab(h7,h10,p5,p6)        )
          ( ix2_6_3(h8,h10,h1,p9)    =   1.0 * v2ijka(h8,h10,h1,p9)                      )
          ( ix2_6_3(h8,h10,h1,p9)   +=   1.0 * d_t1(p5,h1) * v2ijab(h8,h10,p5,p9)        ) //O3V2
          // IP spin explicit  
          ( lt12_o_a(h1_oa,h2_oa)                  =  1.0 * lt12_o(h1_oa,h2_oa)              )
          ( lt12_o_b(h1_ob,h2_ob)                  =  1.0 * lt12_o(h1_ob,h2_ob)              )
          ( ix1_1_1_a(h1_oa,p1_va)                 =  1.0 * ix1_1_1(h1_oa,p1_va)             )
          ( ix1_1_1_b(h1_ob,p1_vb)                 =  1.0 * ix1_1_1(h1_ob,p1_vb)             )
          ( ix2_1_aaaa(h1_oa,p1_va,h2_oa,h3_oa)    =  1.0 * ix2_1(h1_oa,p1_va,h2_oa,h3_oa)   )
          ( ix2_1_abab(h1_oa,p1_vb,h2_oa,h3_ob)    =  1.0 * ix2_1(h1_oa,p1_vb,h2_oa,h3_ob)   )
          ( ix2_1_baba(h1_ob,p1_va,h2_ob,h3_oa)    =  1.0 * ix2_1(h1_ob,p1_va,h2_ob,h3_oa)   )
          ( ix2_1_bbbb(h1_ob,p1_vb,h2_ob,h3_ob)    =  1.0 * ix2_1(h1_ob,p1_vb,h2_ob,h3_ob)   )
          ( ix2_2_a(h1_oa,h2_oa)                   =  1.0 * ix2_2(h1_oa,h2_oa)               )
          ( ix2_2_b(h1_ob,h2_ob)                   =  1.0 * ix2_2(h1_ob,h2_ob)               )
          ( ix2_3_a(p1_va,p2_va)                   =  1.0 * ix2_3(p1_va,p2_va)               )
          ( ix2_3_b(p1_vb,p2_vb)                   =  1.0 * ix2_3(p1_vb,p2_vb)               )
          ( ix2_4_aaaa(h1_oa,h2_oa,h3_oa,h4_oa)    =  1.0 * ix2_4(h1_oa,h2_oa,h3_oa,h4_oa)   )
          ( ix2_4_abab(h1_oa,h2_ob,h3_oa,h4_ob)    =  1.0 * ix2_4(h1_oa,h2_ob,h3_oa,h4_ob)   )
          ( ix2_4_bbbb(h1_ob,h2_ob,h3_ob,h4_ob)    =  1.0 * ix2_4(h1_ob,h2_ob,h3_ob,h4_ob)   )
          ( ix2_5_aaaa(h1_oa,p1_va,h2_oa,p2_va)    =  1.0 * ix2_5(h1_oa,p1_va,h2_oa,p2_va)   )
          ( ix2_5_abba(h1_oa,p1_vb,h2_ob,p2_va)    =  1.0 * ix2_5(h1_oa,p1_vb,h2_ob,p2_va)   )
          ( ix2_5_abab(h1_oa,p1_vb,h2_oa,p2_vb)    =  1.0 * ix2_5(h1_oa,p1_vb,h2_oa,p2_vb)   )
          ( ix2_5_bbbb(h1_ob,p1_vb,h2_ob,p2_vb)    =  1.0 * ix2_5(h1_ob,p1_vb,h2_ob,p2_vb)   )
          ( ix2_5_baab(h1_ob,p1_va,h2_oa,p2_vb)    =  1.0 * ix2_5(h1_ob,p1_va,h2_oa,p2_vb)   )
          ( ix2_5_baba(h1_ob,p1_va,h2_ob,p2_va)    =  1.0 * ix2_5(h1_ob,p1_va,h2_ob,p2_va)   )
          ( ix2_6_2_a(h1_oa,p1_va)                 =  1.0 * ix2_6_2(h1_oa,p1_va)             )
          ( ix2_6_2_b(h1_ob,p1_vb)                 =  1.0 * ix2_6_2(h1_ob,p1_vb)             )
          ( ix2_6_3_aaaa(h1_oa,h2_oa,h3_oa,p1_va)  =  1.0 * ix2_6_3(h1_oa,h2_oa,h3_oa,p1_va) )
          ( ix2_6_3_abba(h1_oa,h2_ob,h3_ob,p1_va)  =  1.0 * ix2_6_3(h1_oa,h2_ob,h3_ob,p1_va) )
          ( ix2_6_3_abab(h1_oa,h2_ob,h3_oa,p1_vb)  =  1.0 * ix2_6_3(h1_oa,h2_ob,h3_oa,p1_vb) )
          ( ix2_6_3_bbbb(h1_ob,h2_ob,h3_ob,p1_vb)  =  1.0 * ix2_6_3(h1_ob,h2_ob,h3_ob,p1_vb) )
          ( ix2_6_3_baab(h1_ob,h2_oa,h3_oa,p1_vb)  =  1.0 * ix2_6_3(h1_ob,h2_oa,h3_oa,p1_vb) )
          ( ix2_6_3_baba(h1_ob,h2_oa,h3_ob,p1_va)  =  1.0 * ix2_6_3(h1_ob,h2_oa,h3_ob,p1_va) )
          .deallocate(lt12_o,
                      ix1_1_1,
                      ix2_1_1,ix2_1_3,ix2_1_temp,ix2_1,
                      ix2_2,
                      ix2_3,
                      ix2_4_1,ix2_4_temp,ix2_4,
                      ix2_5,
                      ix2_6_2,ix2_6_3,
                      v2ijkl,v2iabc);
          
      sch.execute(sch.ec().exhw());

      #if GF_IN_SG
      }
      ec.pg().barrier();
      #endif
      // clang-format on

      // write_to_disk(t2v2_o,t2v2_o_file);
      // write_to_disk(lt12_o_a,lt12_o_a_file);
      // write_to_disk(lt12_o_b,lt12_o_b_file);
      // write_to_disk(ix1_1_1_a,ix1_1_1_a_file);
      // write_to_disk(ix1_1_1_b,ix1_1_1_b_file);
      // write_to_disk(ix2_1_aaaa,ix2_1_aaaa_file);
      // write_to_disk(ix2_1_bbbb,ix2_1_bbbb_file);
      // write_to_disk(ix2_1_abab,ix2_1_abab_file);
      // write_to_disk(ix2_1_baba,ix2_1_baba_file);
      // write_to_disk(ix2_2_a,ix2_2_a_file);
      // write_to_disk(ix2_2_b,ix2_2_b_file);
      // write_to_disk(ix2_3_a,ix2_3_a_file);
      // write_to_disk(ix2_3_b,ix2_3_b_file);
      // write_to_disk(ix2_4_aaaa,ix2_4_aaaa_file);
      // write_to_disk(ix2_4_abab,ix2_4_abab_file);
      // write_to_disk(ix2_4_bbbb,ix2_4_bbbb_file);
      // write_to_disk(ix2_5_aaaa,ix2_5_aaaa_file);
      // write_to_disk(ix2_5_abba,ix2_5_abba_file);
      // write_to_disk(ix2_5_abab,ix2_5_abab_file);
      // write_to_disk(ix2_5_bbbb,ix2_5_bbbb_file);
      // write_to_disk(ix2_5_baab,ix2_5_baab_file);
      // write_to_disk(ix2_5_baba,ix2_5_baba_file);
      // write_to_disk(ix2_6_2_a,ix2_6_2_a_file);
      // write_to_disk(ix2_6_2_b,ix2_6_2_b_file);
      // write_to_disk(ix2_6_3_aaaa,ix2_6_3_aaaa_file);
      // write_to_disk(ix2_6_3_abba,ix2_6_3_abba_file);
      // write_to_disk(ix2_6_3_abab,ix2_6_3_abab_file);
      // write_to_disk(ix2_6_3_bbbb,ix2_6_3_bbbb_file);
      // write_to_disk(ix2_6_3_baab,ix2_6_3_baab_file);
      // write_to_disk(ix2_6_3_baba,ix2_6_3_baba_file);
      std::vector<Tensor<T>> rtensors = {
        t2v2_o,       lt12_o_a,     lt12_o_b,     ix1_1_1_a,    ix1_1_1_b,    ix2_1_aaaa,
        ix2_1_bbbb,   ix2_1_abab,   ix2_1_baba,   ix2_2_a,      ix2_2_b,      ix2_3_a,
        ix2_3_b,      ix2_4_aaaa,   ix2_4_abab,   ix2_4_bbbb,   ix2_5_aaaa,   ix2_5_abba,
        ix2_5_abab,   ix2_5_bbbb,   ix2_5_baab,   ix2_5_baba,   ix2_6_2_a,    ix2_6_2_b,
        ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba};
      std::vector<std::string> rtfnames = {
        t2v2_o_file,       lt12_o_a_file,     lt12_o_b_file,     ix1_1_1_a_file,
        ix1_1_1_b_file,    ix2_1_aaaa_file,   ix2_1_bbbb_file,   ix2_1_abab_file,
        ix2_1_baba_file,   ix2_2_a_file,      ix2_2_b_file,      ix2_3_a_file,
        ix2_3_b_file,      ix2_4_aaaa_file,   ix2_4_abab_file,   ix2_4_bbbb_file,
        ix2_5_aaaa_file,   ix2_5_abba_file,   ix2_5_abab_file,   ix2_5_bbbb_file,
        ix2_5_baab_file,   ix2_5_baba_file,   ix2_6_2_a_file,    ix2_6_2_b_file,
        ix2_6_3_aaaa_file, ix2_6_3_abba_file, ix2_6_3_abab_file, ix2_6_3_bbbb_file,
        ix2_6_3_baab_file, ix2_6_3_baba_file};
      write_to_disk_group(ec, rtensors, rtfnames);
    }

    gfst_end = std::chrono::high_resolution_clock::now();
    gfcc_restart_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((gfst_end - gfst_start)).count();
    if(rank == 0)
      std::cout << " -- GFCC: Time for " << rw_inter
                << " spin-explicit intermediate tensors: " << gfcc_restart_time << " secs"
                << std::endl;

    auto   inter_read_end = std::chrono::high_resolution_clock::now();
    double total_inter_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((inter_read_end - inter_read_start))
        .count();
    if(rank == 0)
      std::cout << "GFCC: Total Time for computing input/intermediate tensors: " << total_inter_time
                << " secs" << std::endl;

    ///////////////////////////////////////
    //                                   //
    //  performing retarded_alpha first  //
    //                                   //
    ///////////////////////////////////////
    if(rank == 0) {
      cout << endl << "_____retarded_GFCCSD_on_alpha_spin______" << endl;
      // ofs_profile << endl << "_____retarded_GFCCSD_on_alpha_spin______" << endl;
    }

    size_t prev_qr_rank_orig    = 0;
    size_t prev_qr_rank_updated = 0;
    // const auto nranks = ec.pg().size().value();

    while(true) {
      const std::string levelstr     = std::to_string(level);
      std::string       q1_a_file    = files_prefix + ".r_q1_a.l" + levelstr;
      std::string       q2_aaa_file  = files_prefix + ".r_q2_aaa.l" + levelstr;
      std::string       q2_bab_file  = files_prefix + ".r_q2_bab.l" + levelstr;
      std::string       hx1_a_file   = files_prefix + ".r_hx1_a.l" + levelstr;
      std::string       hx2_aaa_file = files_prefix + ".r_hx2_aaa.l" + levelstr;
      std::string       hx2_bab_file = files_prefix + ".r_hx2_bab.l" + levelstr;
      std::string       hsub_a_file  = files_prefix + ".r_hsub_a.l" + levelstr;
      std::string       bsub_a_file  = files_prefix + ".r_bsub_a.l" + levelstr;
      std::string       cp_a_file    = files_prefix + ".r_cp_a.l" + levelstr;
      std::string       qrr_up_file  = files_prefix + ".qr_rank_updated.l" + levelstr;

      bool q_exist = fs::exists(q1_a_file) && fs::exists(q2_aaa_file) && fs::exists(q2_bab_file);

      bool gf_restart = q_exist && fs::exists(hx1_a_file) && fs::exists(hx2_aaa_file) &&
                        fs::exists(hx2_bab_file) && fs::exists(hsub_a_file) &&
                        fs::exists(bsub_a_file) && fs::exists(cp_a_file) && ccsd_options.gf_restart;

      // if(rank==0 && debug) cout << "gf_restart: " << gf_restart << endl;

      if(level == 1) {
        omega_extra.push_back(omega_min_ip);
        omega_extra.push_back(omega_max_ip);
      }

      for(auto x: omega_extra) omega_extra_finished.push_back(x);

      auto qr_rank_orig    = omega_extra_finished.size() * noa;
      auto qr_rank_updated = qr_rank_orig;

      if(q_exist) {
        decltype(qr_rank_orig) qrr_up        = 0;
        bool                   qrr_up_exists = fs::exists(qrr_up_file);
        if(rank == 0 && qrr_up_exists) {
          std::ifstream in(qrr_up_file, std::ios::in);
          int           rstatus = 0;
          if(in.is_open()) rstatus = 1;
          if(rstatus == 1) in >> qrr_up;
          qr_rank_updated = qrr_up;
        }
        if(qrr_up_exists) ec.pg().broadcast(&qr_rank_updated, 0);
        else tamm_terminate("q1,q2 files exist, but qr_rank file " + qrr_up_file + " is missing ");
      }

      if(rank == 0) {
        cout << endl << std::string(55, '-') << endl;
        cout << "qr_rank_orig, qr_rank_updated: " << qr_rank_orig << ", " << qr_rank_updated
             << endl;
        cout << "prev_qr_rank_orig, prev_qr_rank_updated: " << prev_qr_rank_orig << ", "
             << prev_qr_rank_updated << endl;
      }

      TiledIndexSpace otis;
      // if(ndiis > qr_rank){
      //   diis_tis = {IndexSpace{range(0,ndiis)}};
      //   otis = {diis_tis, range(0,qr_rank)};
      // }
      // else{
      otis = {IndexSpace{range(qr_rank_orig)}};
      // diis_tis = {otis,range(0,ndiis)};
      // }

      // When restarting, need to read the q1,q2 tensors with last dim qr_rank_updated
      // otis_opt is redefined for the current level if needed post-GS
      TiledIndexSpace otis_opt = {IndexSpace{range(qr_rank_updated)},
                                  static_cast<tamm::Tile>(ccsd_options.tilesize)};
      TiledIndexSpace unit_tis{otis, range(0, 1)};
      // auto [u1] = unit_tis.labels<1>("all");

      for(auto x: omega_extra) {
        // omega_extra_finished.push_back(x);
        ndiis    = ccsd_options.gf_ndiis;
        gf_omega = x;
        if(!gf_restart) {
          gfccsd_driver_ip_a<T>(ec, *sub_ec, subcomm, MO, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                                d_t2_abab, d_f1, t2v2_o, lt12_o_a, lt12_o_b, ix1_1_1_a, ix1_1_1_b,
                                ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b,
                                ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, ix2_5_aaaa,
                                ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                                ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                                ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba, v2ijab_aaaa, v2ijab_abab,
                                v2ijab_bbbb, p_evl_sorted_occ, p_evl_sorted_virt, total_orbitals,
                                nocc, nvir, nptsi, unit_tis, files_prefix, levelstr, noa);
        }
        else if(rank == 0) cout << endl << "Restarting freq: " << gf_omega << endl;
        auto ni             = std::round((x - omega_min_ip) / omega_delta);
        omega_ip_conv_a[ni] = true;
      }

      ComplexTensor q1_tamm_a{o_alpha, otis};
      ComplexTensor q2_tamm_aaa{v_alpha, o_alpha, o_alpha, otis};
      ComplexTensor q2_tamm_bab{v_beta, o_alpha, o_beta, otis};
      if(q_exist) {
        q1_tamm_a   = {o_alpha, otis_opt};
        q2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
        q2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
      }
      ComplexTensor Hx1_tamm_a;
      ComplexTensor Hx2_tamm_aaa;
      ComplexTensor Hx2_tamm_bab;

      if(!gf_restart) {
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        sch.allocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab).execute();

        const std::string plevelstr = std::to_string(level - 1);

        std::string pq1_a_file   = files_prefix + ".r_q1_a.l" + plevelstr;
        std::string pq2_aaa_file = files_prefix + ".r_q2_aaa.l" + plevelstr;
        std::string pq2_bab_file = files_prefix + ".r_q2_bab.l" + plevelstr;

        bool prev_q12 = fs::exists(pq1_a_file) && fs::exists(pq2_aaa_file) &&
                        fs::exists(pq2_bab_file);

        if(rank == 0 && debug) cout << "prev_q12:" << prev_q12 << endl;

        if(prev_q12 && !q_exist) {
          TiledIndexSpace otis_prev_opt = {IndexSpace{range(0, prev_qr_rank_updated)},
                                           static_cast<tamm::Tile>(ccsd_options.tilesize)};
          TiledIndexSpace otis_prev{otis, range(0, prev_qr_rank_updated)};
          auto [op1] = otis_prev.labels<1>("all");
          ComplexTensor q1_prev_a{o_alpha, otis_prev_opt};
          ComplexTensor q2_prev_aaa{v_alpha, o_alpha, o_alpha, otis_prev_opt};
          ComplexTensor q2_prev_bab{v_beta, o_alpha, o_beta, otis_prev_opt};
          sch.allocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();

          read_from_disk(q1_prev_a, pq1_a_file);
          read_from_disk(q2_prev_aaa, pq2_aaa_file);
          read_from_disk(q2_prev_bab, pq2_bab_file);

          { // retile q1 prev a,aaa,bab tensors
            int q1_prev_a_ga   = tamm_to_ga(ec, q1_prev_a);
            int q2_prev_aaa_ga = tamm_to_ga(ec, q2_prev_aaa);
            int q2_prev_bab_ga = tamm_to_ga(ec, q2_prev_bab);
            sch.deallocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();

            q1_prev_a   = {o_alpha, otis_prev};
            q2_prev_aaa = {v_alpha, o_alpha, o_alpha, otis_prev};
            q2_prev_bab = {v_beta, o_alpha, o_beta, otis_prev};
            sch.allocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();

            ga_to_tamm(ec, q1_prev_a, q1_prev_a_ga);
            ga_to_tamm(ec, q2_prev_aaa, q2_prev_aaa_ga);
            ga_to_tamm(ec, q2_prev_bab, q2_prev_bab_ga);
            NGA_Destroy(q1_prev_a_ga);
            NGA_Destroy(q2_prev_aaa_ga);
            NGA_Destroy(q2_prev_bab_ga);
          }

          if(subcomm != MPI_COMM_NULL) {
            // clang-format off
            sub_sch
              (q1_tamm_a(h1_oa,op1) = q1_prev_a(h1_oa,op1))
              (q2_tamm_aaa(p1_va,h1_oa,h2_oa,op1) = q2_prev_aaa(p1_va,h1_oa,h2_oa,op1))
              (q2_tamm_bab(p1_vb,h1_oa,h2_ob,op1) = q2_prev_bab(p1_vb,h1_oa,h2_ob,op1)).execute();
            // clang-format on
          }
          sch.deallocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();

          // check q1/2_prev
          if(debug) {
            auto nrm_q1_a_prev   = norm(q1_tamm_a);
            auto nrm_q2_aaa_prev = norm(q2_tamm_aaa);
            auto nrm_q2_bab_prev = norm(q2_tamm_bab);
            if(rank == 0 && debug) {
              cout << "norm of q1/2 at previous level" << endl;
              cout << nrm_q1_a_prev << "," << nrm_q2_aaa_prev << "," << nrm_q2_bab_prev << endl;
            }
          }
        }

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          cout << endl
               << "Time to read in pre-computed Q1/Q2: " << std::fixed << std::setprecision(6)
               << time << " secs" << endl;

        std::vector<ComplexTensor> gs_q1_tmp_a;   //{o_alpha};
        std::vector<ComplexTensor> gs_q2_tmp_aaa; //{v_alpha,o_alpha,o_alpha};
        std::vector<ComplexTensor> gs_q2_tmp_bab; //{v_beta, o_alpha,o_beta};

        // TODO: optimize Q1/Q2 computation
        // Gram-Schmidt orthogonalization
        double time_gs_orth = 0.0;
        double time_gs_norm = 0.0;

        double q_norm_threshold = sys_data.options_map.scf_options.tol_lindep;
        if(rank == 0 && debug) { cout << "q_norm threshold: " << q_norm_threshold << endl; }

        auto gs_start_timer = std::chrono::high_resolution_clock::now();

        if(!q_exist) {
          size_t gs_cur_lindep  = 0;
          auto   gs_prev_lindep = prev_qr_rank_orig - prev_qr_rank_updated;

          const auto                 ngsvecs = (qr_rank_orig - prev_qr_rank_orig);
          std::vector<ComplexTensor> gsvectors;           //(ngsvecs);
          std::vector<std::string>   gsvectors_filenames; //(ngsvecs);

          auto gs_rv_start = std::chrono::high_resolution_clock::now();

          for(auto ivec = prev_qr_rank_orig; ivec < qr_rank_orig; ivec++) {
            ComplexTensor q1_tmp_a{o_alpha};
            ComplexTensor q2_tmp_aaa{v_alpha, o_alpha, o_alpha};
            ComplexTensor q2_tmp_bab{v_beta, o_alpha, o_beta};
            sch.allocate(q1_tmp_a, q2_tmp_aaa, q2_tmp_bab).execute();

            auto              W_read  = omega_extra_finished[ivec / (noa)];
            auto              pi_read = ivec % (noa);
            std::stringstream gfo;
            gfo << std::fixed << std::setprecision(2) << W_read;

            std::string x1_a_wpi_file =
              files_prefix + ".x1_a.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string x2_aaa_wpi_file =
              files_prefix + ".x2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string x2_bab_wpi_file =
              files_prefix + ".x2_bab.w" + gfo.str() + ".oi" + std::to_string(pi_read);

            if(fs::exists(x1_a_wpi_file) && fs::exists(x2_aaa_wpi_file) &&
               fs::exists(x2_bab_wpi_file)) {
              // read_from_disk(q1_tmp_a,x1_a_wpi_file);
              // read_from_disk(q2_tmp_aaa,x2_aaa_wpi_file);
              // read_from_disk(q2_tmp_bab,x2_bab_wpi_file);
              gs_q1_tmp_a.push_back(q1_tmp_a);
              gs_q2_tmp_aaa.push_back(q2_tmp_aaa);
              gs_q2_tmp_bab.push_back(q2_tmp_bab);
              gsvectors.insert(gsvectors.end(), {q1_tmp_a, q2_tmp_aaa, q2_tmp_bab});
              gsvectors_filenames.insert(gsvectors_filenames.end(),
                                         {x1_a_wpi_file, x2_aaa_wpi_file, x2_bab_wpi_file});
            }
            else {
              tamm_terminate("ERROR: At least one of " + x1_a_wpi_file + " and " + x2_aaa_wpi_file +
                             " and " + x2_bab_wpi_file + " do not exist!");
            }
          }

          EXPECTS(gsvectors.size() == 3 * ngsvecs);
          read_from_disk_group(ec, gsvectors, gsvectors_filenames);
          auto gs_rv_end = std::chrono::high_resolution_clock::now();
          auto gs_read_time =
            std::chrono::duration_cast<std::chrono::duration<double>>((gs_rv_end - gs_rv_start))
              .count();
          if(rank == 0) {
            cout << endl
                 << " -- Gram-Schmidt: Time for reading GS vectors from disk: " << std::fixed
                 << std::setprecision(6) << gs_read_time << " secs" << endl
                 << endl;
          }

          auto ivec_start = prev_qr_rank_orig;

          // setup for restarting ivec loop as needed
          std::string gs_ivec_file = files_prefix + ".gs_ivec.l" + levelstr;
#if 1
          if(ccsd_options.gf_restart) {
            bool gsivec_exists = fs::exists(gs_ivec_file);
            if(rank == 0 && gsivec_exists) {
              decltype(qr_rank_orig) gs_istart = 0;
              std::ifstream          in(gs_ivec_file, std::ios::in);
              int                    rstatus = 0;
              if(in.is_open()) rstatus = 1;
              if(rstatus == 1) in >> gs_istart;
              ivec_start = gs_istart;
            }
            if(gsivec_exists) {
              ec.pg().broadcast(&ivec_start, 0);
              auto q1_a_file   = files_prefix + ".r_q1_a.gs_ivec.l" + levelstr;
              auto q2_aaa_file = files_prefix + ".r_q2_aaa.gs_ivec.l" + levelstr;
              auto q2_bab_file = files_prefix + ".r_q2_bab.gs_ivec.l" + levelstr;
              auto q_exist     = fs::exists(q1_a_file) && fs::exists(q2_aaa_file) &&
                             fs::exists(q2_bab_file);
              if(q_exist) {
                if(rank == 0)
                  std::cout << "Restarting GS loop from ivec: " << ivec_start << std::endl;
                ComplexTensor i_q1_tamm_a   = {o_alpha, otis_opt};
                ComplexTensor i_q2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
                ComplexTensor i_q2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
                sch.allocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();
                read_from_disk_group<std::complex<T>>(
                  ec, std::vector{i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab},
                  {q1_a_file, q2_aaa_file, q2_bab_file}, {}, gf_profile);

                int q1_tamm_a_ga   = tamm_to_ga(ec, i_q1_tamm_a);
                int q2_tamm_aaa_ga = tamm_to_ga(ec, i_q2_tamm_aaa);
                int q2_tamm_bab_ga = tamm_to_ga(ec, i_q2_tamm_bab);
                sch.deallocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();

                ga_to_tamm(ec, q1_tamm_a, q1_tamm_a_ga);
                ga_to_tamm(ec, q2_tamm_aaa, q2_tamm_aaa_ga);
                ga_to_tamm(ec, q2_tamm_bab, q2_tamm_bab_ga);
                NGA_Destroy(q1_tamm_a_ga);
                NGA_Destroy(q2_tamm_aaa_ga);
                NGA_Destroy(q2_tamm_bab_ga);
              }
            }
          }
#endif

          for(auto ivec = ivec_start; ivec < qr_rank_orig; ivec++) {
            if(gf_profile && rank == 0) std::cout << " -- GS: ivec " << ivec;

            auto cc_t0 = std::chrono::high_resolution_clock::now();

            const auto    gs_sind    = ivec % ngsvecs;
            ComplexTensor q1_tmp_a   = gs_q1_tmp_a[gs_sind];
            ComplexTensor q2_tmp_aaa = gs_q2_tmp_aaa[gs_sind];
            ComplexTensor q2_tmp_bab = gs_q2_tmp_bab[gs_sind];

            // TODO: schedule all iterations before executing
            if(ivec > 0) {
              TiledIndexSpace tsc{otis, range(0, ivec - gs_prev_lindep)};
              auto [sc]               = tsc.labels<1>("all");
              TiledIndexSpace tsc_opt = {IndexSpace{range(0, ivec - gs_prev_lindep)},
                                         static_cast<tamm::Tile>(ccsd_options.tilesize)};
              auto [sc_opt]           = tsc_opt.labels<1>("all");

              ComplexTensor oscalar{tsc_opt};
              ComplexTensor x1c_a{o_alpha, tsc};
              ComplexTensor x2c_aaa{v_alpha, o_alpha, o_alpha, tsc};
              ComplexTensor x2c_bab{v_beta, o_alpha, o_beta, tsc};

              // clang-format off

              #if GF_GS_SG
              if(subcomm != MPI_COMM_NULL) {
                sub_sch.allocate
              #else
                sch.allocate
              #endif
                  (x1c_a,x2c_aaa,x2c_bab)
                  (x1c_a(h1_oa,sc) = q1_tamm_a(h1_oa,sc))
                  (x2c_aaa(p1_va,h1_oa,h2_oa,sc) = q2_tamm_aaa(p1_va,h1_oa,h2_oa,sc))
                  (x2c_bab(p1_vb,h1_oa,h2_ob,sc) = q2_tamm_bab(p1_vb,h1_oa,h2_ob,sc))
                  .execute();
                // clang-format on

                { // retile x1c a,aaa,bab tensors
                  int x1c_a_ga   = tamm_to_ga(ec, x1c_a);
                  int x2c_aaa_ga = tamm_to_ga(ec, x2c_aaa);
                  int x2c_bab_ga = tamm_to_ga(ec, x2c_bab);
                  sch.deallocate(x1c_a, x2c_aaa, x2c_bab).execute();

                  x1c_a   = {o_alpha, tsc_opt};
                  x2c_aaa = {v_alpha, o_alpha, o_alpha, tsc_opt};
                  x2c_bab = {v_beta, o_alpha, o_beta, tsc_opt};
                  sch.allocate(x1c_a, x2c_aaa, x2c_bab).execute();

                  ga_to_tamm(ec, x1c_a, x1c_a_ga);
                  ga_to_tamm(ec, x2c_aaa, x2c_aaa_ga);
                  ga_to_tamm(ec, x2c_bab, x2c_bab_ga);
                  NGA_Destroy(x1c_a_ga);
                  NGA_Destroy(x2c_aaa_ga);
                  NGA_Destroy(x2c_bab_ga);
                }

                ComplexTensor x1c_a_conj   = tamm::conj(x1c_a);
                ComplexTensor x2c_aaa_conj = tamm::conj(x2c_aaa);
                ComplexTensor x2c_bab_conj = tamm::conj(x2c_bab);

                // clang-format off

                #if GF_GS_SG
                  sub_sch.allocate
                #else
                  sch.allocate
                #endif

                    // 1st GS
                    (oscalar)
                    (oscalar(sc_opt)  = -1.0 * q1_tmp_a(h1_oa) * x1c_a_conj(h1_oa,sc_opt))
                    (oscalar(sc_opt) += -0.5 * q2_tmp_aaa(p1_va,h1_oa,h2_oa) * x2c_aaa_conj(p1_va,h1_oa,h2_oa,sc_opt))
                    (oscalar(sc_opt) += -1.0 * q2_tmp_bab(p1_vb,h1_oa,h2_ob) * x2c_bab_conj(p1_vb,h1_oa,h2_ob,sc_opt))

                    (q1_tmp_a(h1_oa) += oscalar(sc_opt) * x1c_a(h1_oa,sc_opt))
                    (q2_tmp_aaa(p1_va,h1_oa,h2_oa) += oscalar(sc_opt) * x2c_aaa(p1_va,h1_oa,h2_oa,sc_opt))
                    (q2_tmp_bab(p1_vb,h1_oa,h2_ob) += oscalar(sc_opt) * x2c_bab(p1_vb,h1_oa,h2_ob,sc_opt))

                    // 2nd GS
                    // (oscalar(sc_opt)  = -1.0 * q1_tmp_a(h1_oa) * x1c_a_conj(h1_oa,sc_opt))
                    // (oscalar(sc_opt) += -0.5 * q2_tmp_aaa(p1_va,h1_oa,h2_oa) * x2c_aaa_conj(p1_va,h1_oa,h2_oa,sc_opt))
                    // (oscalar(sc_opt) += -1.0 * q2_tmp_bab(p1_vb,h1_oa,h2_ob) * x2c_bab_conj(p1_vb,h1_oa,h2_ob,sc_opt))
  
                    // (q1_tmp_a(h1_oa) += oscalar(sc_opt) * x1c_a(h1_oa,sc_opt))
                    // (q2_tmp_aaa(p1_va,h1_oa,h2_oa) += oscalar(sc_opt) * x2c_aaa(p1_va,h1_oa,h2_oa,sc_opt))
                    // (q2_tmp_bab(p1_vb,h1_oa,h2_ob) += oscalar(sc_opt) * x2c_bab(p1_vb,h1_oa,h2_ob,sc_opt))
                    .deallocate(oscalar,x1c_a,x2c_aaa,x2c_bab,x1c_a_conj,x2c_aaa_conj,x2c_bab_conj);
                    // end of 2nd GS


                  #if GF_IN_SG
                    sub_sch.execute(sub_sch.ec().exhw());
                  }
                  ec.pg().barrier();
                  #else
                    sch.execute(sch.ec().exhw());
                  #endif
              // clang-format on
            }

            auto       cc_t1 = std::chrono::high_resolution_clock::now();
            const auto time_gs_orth_i =
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_t1 - cc_t0)).count();
            time_gs_orth += time_gs_orth_i;

            if(gf_profile && rank == 0) std::cout << ": Orthogonalization: " << time_gs_orth_i;

            auto q1norm_a   = norm(q1_tmp_a);
            auto q2norm_aaa = norm(q2_tmp_aaa);
            auto q2norm_bab = norm(q2_tmp_bab);

            // Normalization factor
            T q_norm = std::real(
              sqrt(q1norm_a * q1norm_a + 0.5 * q2norm_aaa * q2norm_aaa + q2norm_bab * q2norm_bab));

            if(q_norm < q_norm_threshold) {
              gs_cur_lindep++;
              if(gf_profile && rank == 0) cout << " --- continue" << endl;
#if 1
              if(ccsd_options.gf_restart && ((ivec - ivec_start) % ndiis == 0)) {
                if(rank == 0) {
                  std::ofstream out(gs_ivec_file, std::ios::out);
                  if(!out) cerr << "Error opening file " << gs_ivec_file << endl;
                  out << ivec << std::endl;
                  out.close();
                }
                auto q1_a_file   = files_prefix + ".r_q1_a.gs_ivec.l" + levelstr;
                auto q2_aaa_file = files_prefix + ".r_q2_aaa.gs_ivec.l" + levelstr;
                auto q2_bab_file = files_prefix + ".r_q2_bab.gs_ivec.l" + levelstr;
                { // retile q1 a,aaa,bab tensors
                  int q1_tamm_a_ga   = tamm_to_ga(ec, q1_tamm_a);
                  int q2_tamm_aaa_ga = tamm_to_ga(ec, q2_tamm_aaa);
                  int q2_tamm_bab_ga = tamm_to_ga(ec, q2_tamm_bab);

                  ComplexTensor i_q1_tamm_a   = {o_alpha, otis_opt};
                  ComplexTensor i_q2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
                  ComplexTensor i_q2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
                  sch.allocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();

                  ga_to_tamm(ec, i_q1_tamm_a, q1_tamm_a_ga);
                  ga_to_tamm(ec, i_q2_tamm_aaa, q2_tamm_aaa_ga);
                  ga_to_tamm(ec, i_q2_tamm_bab, q2_tamm_bab_ga);
                  NGA_Destroy(q1_tamm_a_ga);
                  NGA_Destroy(q2_tamm_aaa_ga);
                  NGA_Destroy(q2_tamm_bab_ga);
                  write_to_disk_group<std::complex<T>>(
                    ec, {i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab},
                    {q1_a_file, q2_aaa_file, q2_bab_file}, gf_profile && ivec == ivec_start);
                  sch.deallocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();
                }
              }
#endif
              continue;
            }

            auto cc_norm = std::chrono::high_resolution_clock::now();
            if(gf_profile && rank == 0)
              std::cout << ", Norm: "
                        << std::chrono::duration_cast<std::chrono::duration<double>>(
                             (cc_norm - cc_t1))
                             .count();

            T newsc = 1.0 / q_norm;

            std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);

            TiledIndexSpace tsc{otis, range(ivec - gs_prev_lindep, ivec - gs_prev_lindep + 1)};
            auto [sc] = tsc.labels<1>("all");

            if(subcomm != MPI_COMM_NULL) {
              // clang-format off
              sub_sch
              (q1_tamm_a(h1_oa,sc) = cnewsc * q1_tmp_a(h1_oa))
              (q2_tamm_aaa(p2_va,h1_oa,h2_oa,sc) = cnewsc * q2_tmp_aaa(p2_va,h1_oa,h2_oa))
              (q2_tamm_bab(p2_vb,h1_oa,h2_ob,sc) = cnewsc * q2_tmp_bab(p2_vb,h1_oa,h2_ob))
              .execute();
              // clang-format on
            }
            ec.pg().barrier();

            auto cc_copy = std::chrono::high_resolution_clock::now();
            if(gf_profile && rank == 0)
              std::cout << ", Copy: "
                        << std::chrono::duration_cast<std::chrono::duration<double>>(
                             (cc_copy - cc_norm))
                             .count();

            if(gf_profile && rank == 0) std::cout << std::endl;

            // check q1/2 inside G-S
            if(debug) {
              auto nrm_q1_a_gs   = norm(q1_tamm_a);
              auto nrm_q2_aaa_gs = norm(q2_tamm_aaa);
              auto nrm_q2_bab_gs = norm(q2_tamm_bab);
              if(rank == 0) {
                cout << " -- " << ivec << "," << q1norm_a << "," << q2norm_aaa << "," << q2norm_bab
                     << endl;
                cout << "   "
                     << "," << cnewsc << "," << nrm_q1_a_gs << "," << nrm_q2_aaa_gs << ","
                     << nrm_q2_bab_gs << endl;
              }
            }

            auto cc_gs = std::chrono::high_resolution_clock::now();
            auto time_gs_norm_i =
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
            time_gs_norm += time_gs_norm_i;
            if(gf_profile && rank == 0)
              std::cout << "GS: Time (s) for processing ivec " << ivec
                        << ": Orthogonalization: " << time_gs_orth_i
                        << ", normalization/copy: " << time_gs_norm_i << endl;

#if 1
            if(ccsd_options.gf_restart && ((ivec - ivec_start) % ndiis == 0)) {
              if(rank == 0) {
                std::ofstream out(gs_ivec_file, std::ios::out);
                if(!out) cerr << "Error opening file " << gs_ivec_file << endl;
                out << ivec << std::endl;
                out.close();
              }
              auto q1_a_file   = files_prefix + ".r_q1_a.gs_ivec.l" + levelstr;
              auto q2_aaa_file = files_prefix + ".r_q2_aaa.gs_ivec.l" + levelstr;
              auto q2_bab_file = files_prefix + ".r_q2_bab.gs_ivec.l" + levelstr;
              { // retile q1 a,aaa,bab tensors
                int q1_tamm_a_ga   = tamm_to_ga(ec, q1_tamm_a);
                int q2_tamm_aaa_ga = tamm_to_ga(ec, q2_tamm_aaa);
                int q2_tamm_bab_ga = tamm_to_ga(ec, q2_tamm_bab);

                ComplexTensor i_q1_tamm_a   = {o_alpha, otis_opt};
                ComplexTensor i_q2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
                ComplexTensor i_q2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
                sch.allocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();

                ga_to_tamm(ec, i_q1_tamm_a, q1_tamm_a_ga);
                ga_to_tamm(ec, i_q2_tamm_aaa, q2_tamm_aaa_ga);
                ga_to_tamm(ec, i_q2_tamm_bab, q2_tamm_bab_ga);
                NGA_Destroy(q1_tamm_a_ga);
                NGA_Destroy(q2_tamm_aaa_ga);
                NGA_Destroy(q2_tamm_bab_ga);
                write_to_disk_group<std::complex<T>>(
                  ec, {i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab},
                  {q1_a_file, q2_aaa_file, q2_bab_file}, gf_profile && ivec == ivec_start);
                sch.deallocate(i_q1_tamm_a, i_q2_tamm_aaa, i_q2_tamm_bab).execute();
              }
            }
#endif
          } // end of Gram-Schmidt for loop over ivec

          if(ccsd_options.gf_restart && rank == 0) {
            std::ofstream out(gs_ivec_file, std::ios::out);
            if(!out) cerr << "Error opening file " << gs_ivec_file << endl;
            out << qr_rank_orig << std::endl;
            out.close();
          }

          free_vec_tensors(gs_q1_tmp_a, gs_q2_tmp_aaa, gs_q2_tmp_bab);

          qr_rank_updated = qr_rank_orig - gs_cur_lindep - gs_prev_lindep;
          otis_opt        = {IndexSpace{range(qr_rank_updated)},
                             static_cast<tamm::Tile>(ccsd_options.tilesize)};

          { // retile q1 a,aaa,bab tensors
            int q1_tamm_a_ga   = tamm_to_ga(ec, q1_tamm_a);
            int q2_tamm_aaa_ga = tamm_to_ga(ec, q2_tamm_aaa);
            int q2_tamm_bab_ga = tamm_to_ga(ec, q2_tamm_bab);
            sch.deallocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab).execute();

            q1_tamm_a   = {o_alpha, otis_opt};
            q2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
            q2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
            sch.allocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab).execute();

            ga_to_tamm(ec, q1_tamm_a, q1_tamm_a_ga);
            ga_to_tamm(ec, q2_tamm_aaa, q2_tamm_aaa_ga);
            ga_to_tamm(ec, q2_tamm_bab, q2_tamm_bab_ga);
            NGA_Destroy(q1_tamm_a_ga);
            NGA_Destroy(q2_tamm_aaa_ga);
            NGA_Destroy(q2_tamm_bab_ga);
          }

          write_to_disk(q1_tamm_a, q1_a_file);
          write_to_disk(q2_tamm_aaa, q2_aaa_file);
          write_to_disk(q2_tamm_bab, q2_bab_file);
        }      // end of !gs-restart
        else { // restart GS
          read_from_disk(q1_tamm_a, q1_a_file);
          read_from_disk(q2_tamm_aaa, q2_aaa_file);
          read_from_disk(q2_tamm_bab, q2_bab_file);
        }

        auto total_time_gs = std::chrono::duration_cast<std::chrono::duration<double>>(
                               (std::chrono::high_resolution_clock::now() - gs_start_timer))
                               .count();

        if(rank == 0) {
          cout << endl
               << " -- Gram-Schmidt: Time for orthogonalization: " << std::fixed
               << std::setprecision(6) << time_gs_orth << " secs" << endl;
          cout << " -- Gram-Schmidt: Time for normalizing and copying back: " << std::fixed
               << std::setprecision(6) << time_gs_norm << " secs" << endl;
          cout << "Total time for Gram-Schmidt: " << std::fixed << std::setprecision(6)
               << total_time_gs << " secs" << endl;
        }
        auto cc_gs_x = std::chrono::high_resolution_clock::now();

        Hx1_tamm_a   = {o_alpha, otis_opt};
        Hx2_tamm_aaa = {v_alpha, o_alpha, o_alpha, otis_opt};
        Hx2_tamm_bab = {v_beta, o_alpha, o_beta, otis_opt};
        sch.allocate(Hx1_tamm_a, Hx2_tamm_aaa, Hx2_tamm_bab).execute();

        bool gs_x12_restart = fs::exists(hx1_a_file) && fs::exists(hx2_aaa_file) &&
                              fs::exists(hx2_bab_file);

        if(!gs_x12_restart) {
#if GF_IN_SG
          if(subcomm != MPI_COMM_NULL) {
            gfccsd_x1_a(sub_sch,
#else
          gfccsd_x1_a(sch,
#endif
                        MO, Hx1_tamm_a, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, q1_tamm_a,
                        q2_tamm_aaa, q2_tamm_bab, d_f1, ix2_2_a, ix1_1_1_a, ix1_1_1_b, ix2_6_3_aaaa,
                        ix2_6_3_abab, otis_opt, true);

#if GF_IN_SG
            gfccsd_x2_a(sub_sch,
#else
          gfccsd_x2_a(sch,
#endif
                        MO, Hx2_tamm_aaa, Hx2_tamm_bab, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                        d_t2_abab, q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, d_f1, ix2_1_aaaa,
                        ix2_1_abab, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab,
                        ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_6_2_a,
                        ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb,
                        ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, otis_opt, true);

#if GF_IN_SG
            sub_sch.execute(sub_sch.ec().exhw());
          }
          ec.pg().barrier();
#else
          sch.execute(sch.ec().exhw());
#endif
          write_to_disk(Hx1_tamm_a, hx1_a_file);
          write_to_disk(Hx2_tamm_aaa, hx2_aaa_file);
          write_to_disk(Hx2_tamm_bab, hx2_bab_file);
        }
        else {
          read_from_disk(Hx1_tamm_a, hx1_a_file);
          read_from_disk(Hx2_tamm_aaa, hx2_aaa_file);
          read_from_disk(Hx2_tamm_bab, hx2_bab_file);
        }

        // check q and hx files
        if(debug) {
          auto nrm_q1_tamm_a    = norm(q1_tamm_a);
          auto nrm_q2_tamm_aaa  = norm(q2_tamm_aaa);
          auto nrm_q2_tamm_bab  = norm(q2_tamm_bab);
          auto nrm_hx1_tamm_a   = norm(Hx1_tamm_a);
          auto nrm_hx2_tamm_aaa = norm(Hx2_tamm_aaa);
          auto nrm_hx2_tamm_bab = norm(Hx2_tamm_bab);
          if(rank == 0) {
            cout << endl << "norms of q1/2 and hq1/2" << endl;
            cout << nrm_q1_tamm_a << "," << nrm_q2_tamm_aaa << "," << nrm_q2_tamm_bab << endl;
            cout << nrm_hx1_tamm_a << "," << nrm_hx2_tamm_aaa << "," << nrm_hx2_tamm_bab << endl;
          }
        }

        auto   cc_q12 = std::chrono::high_resolution_clock::now();
        double time_q12 =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
        if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;
      } // if !gf_restart

      prev_qr_rank_orig    = qr_rank_orig;
      prev_qr_rank_updated = qr_rank_updated;

      if(rank == 0) {
        std::ofstream out(qrr_up_file, std::ios::out);
        if(!out) cerr << "Error opening file " << qrr_up_file << endl;
        out << qr_rank_updated << std::endl;
        out.close();
      }

      auto cc_t1 = std::chrono::high_resolution_clock::now();

      auto [otil, otil1, otil2] = otis_opt.labels<3>("all");
      ComplexTensor hsub_tamm_a{otis_opt, otis_opt};
      ComplexTensor bsub_tamm_a{otis_opt, o_alpha};
      ComplexTensor Cp_a{o_alpha, otis_opt};
      ComplexTensor::allocate(&ec, hsub_tamm_a, bsub_tamm_a, Cp_a);

      if(!gf_restart) {
        ComplexTensor p1_k_a{v_alpha, otis_opt};

        ComplexTensor q1_conj_a   = tamm::conj(q1_tamm_a);
        ComplexTensor q2_conj_aaa = tamm::conj(q2_tamm_aaa);
        ComplexTensor q2_conj_bab = tamm::conj(q2_tamm_bab);

        // clang-format off
        sch.allocate(p1_k_a)
            (bsub_tamm_a(otil1,h1_oa)  =       q1_conj_a(h1_oa,otil1))
            (hsub_tamm_a(otil1,otil2)  =       q1_conj_a(h1_oa,otil1) * Hx1_tamm_a(h1_oa,otil2))
            (hsub_tamm_a(otil1,otil2) += 0.5 * q2_conj_aaa(p1_va,h1_oa,h2_oa,otil1) * Hx2_tamm_aaa(p1_va,h1_oa,h2_oa,otil2))
            (hsub_tamm_a(otil1,otil2) +=       q2_conj_bab(p1_vb,h1_oa,h2_ob,otil1) * Hx2_tamm_bab(p1_vb,h1_oa,h2_ob,otil2))
            
            ( Cp_a(h1_oa,otil)    =        q1_tamm_a(h1_oa,otil)                                     )
            ( Cp_a(h2_oa,otil)   += -1.0 * lt12_o_a(h1_oa,h2_oa) * q1_tamm_a(h1_oa,otil)               )
            ( Cp_a(h2_oa,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aaa(p1_va,h2_oa,h1_oa,otil) )
            ( Cp_a(h2_oa,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bab(p1_vb,h2_oa,h1_ob,otil) )
            ( p1_k_a(p1_va,otil)  =        d_t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) * q2_tamm_aaa(p2_va,h1_oa,h2_oa,otil) )
            ( p1_k_a(p1_va,otil) +=  2.0 * d_t2_abab(p1_va,p2_vb,h1_oa,h2_ob) * q2_tamm_bab(p2_vb,h1_oa,h2_ob,otil) )
            ( Cp_a(h1_oa,otil)   += -0.5 * p1_k_a(p1_va,otil) * d_t1_a(p1_va,h1_oa) )
            .deallocate(q1_conj_a,q2_conj_aaa,q2_conj_bab)
            .deallocate(p1_k_a,q1_tamm_a, q2_tamm_aaa, q2_tamm_bab,Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab);
        // clang-format on
        sch.execute(sch.ec().exhw());
      } // if !gf_restart

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

      // Write all tensors
      if(!gf_restart) {
        write_to_disk(hsub_tamm_a, hsub_a_file);
        write_to_disk(bsub_tamm_a, bsub_a_file);
        write_to_disk(Cp_a, cp_a_file);
      }
      else {
        read_from_disk(hsub_tamm_a, hsub_a_file);
        read_from_disk(bsub_tamm_a, bsub_a_file);
        read_from_disk(Cp_a, cp_a_file);
      }

      // check hsub,bsub,Cp
      if(debug) {
        auto nrm_hsub = norm(hsub_tamm_a);
        auto nrm_bsub = norm(bsub_tamm_a);
        auto nrm_cp   = norm(Cp_a);
        if(rank == 0) {
          cout << "norms of hsub,bsub,cp" << endl;
          cout << nrm_hsub << "," << nrm_bsub << "," << nrm_cp << endl;
        }
      }

      Complex2DMatrix hsub_a(qr_rank_updated, qr_rank_updated);
      Complex2DMatrix bsub_a(qr_rank_updated, noa);

      tamm_to_eigen_tensor(hsub_tamm_a, hsub_a);
      tamm_to_eigen_tensor(bsub_tamm_a, bsub_a);
      Complex2DMatrix hident = Complex2DMatrix::Identity(hsub_a.rows(), hsub_a.cols());

      ComplexTensor xsub_local_a{otis_opt, o_alpha};
      ComplexTensor o_local_a{o_alpha};
      ComplexTensor Cp_local_a{o_alpha, otis_opt};

      ComplexTensor::allocate(&ec_l, xsub_local_a, o_local_a, Cp_local_a);

      Scheduler sch_l{ec_l};
      sch_l(Cp_local_a(h1_oa, otil) = Cp_a(h1_oa, otil)).execute();

      if(rank == 0) {
        cout << endl << "spectral function (omega_npts_ip = " << omega_npts_ip << "):" << endl;
      }

      cc_t1 = std::chrono::high_resolution_clock::now();

      std::vector<double> ni_w(omega_npts_ip, 0);
      std::vector<double> ni_A(omega_npts_ip, 0);

      // Compute spectral function for designated omega regime
      for(int64_t ni = 0; ni < omega_npts_ip; ni++) {
        std::complex<T> omega_tmp = std::complex<T>(omega_min_ip + ni * omega_delta, -1.0 * gf_eta);

        Complex2DMatrix xsub_a = (hsub_a + omega_tmp * hident).lu().solve(bsub_a);
        eigen_to_tamm_tensor(xsub_local_a, xsub_a);

        sch_l(o_local_a(h1_oa) = Cp_local_a(h1_oa, otil) * xsub_local_a(otil, h1_oa)).execute();

        auto oscalar = std::imag(tamm::sum(o_local_a));

        if(level == 1) { omega_ip_A0[ni] = oscalar; }
        else {
          if(level > 1) {
            T oerr          = oscalar - omega_ip_A0[ni];
            omega_ip_A0[ni] = oscalar;
            if(std::abs(oerr) < gf_threshold) omega_ip_conv_a[ni] = true;
          }
        }
        if(rank == 0) {
          std::ostringstream spf;
          spf << "W = " << std::fixed << std::setprecision(2) << std::real(omega_tmp)
              << ", omega_ip_A0 = " << std::fixed << std::setprecision(4) << omega_ip_A0[ni]
              << endl;
          cout << spf.str();
          ni_A[ni] = omega_ip_A0[ni];
          ni_w[ni] = std::real(omega_tmp);
        }
      }

      cc_t2 = std::chrono::high_resolution_clock::now();
      time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) {
        cout << endl
             << "omegas processed in level " << level << " = " << std::fixed << std::setprecision(2)
             << omega_extra << endl;
        cout << "Time to compute spectral function in level " << level
             << " (omega_npts_ip = " << omega_npts_ip << "): " << time << " secs" << endl;
        write_results_to_json(ec, sys_data, level, ni_w, ni_A, "retarded_alpha");
      }

      auto               extrap_file = files_prefix + ".extrapolate.retarded.alpha.txt";
      std::ostringstream spfe;
      spfe << "";

      if(level == 1) {
        auto o1 = (omega_extra[0] + omega_extra[1]) / 2;
        omega_extra.clear();
        o1 = find_closest(o1, omega_space_ip);
        omega_extra.push_back(o1);
      }
      else {
        std::sort(omega_extra_finished.begin(), omega_extra_finished.end());
        omega_extra.clear();
        std::vector<T> wtemp;
        for(size_t i = 1; i < omega_extra_finished.size(); i++) {
          bool   oe_add = false;
          auto   w1     = omega_extra_finished[i - 1];
          auto   w2     = omega_extra_finished[i];
          size_t num_w  = (w2 - w1) / omega_delta + 1;
          for(size_t j = 0; j < num_w; j++) {
            T      otmp = w1 + j * omega_delta;
            size_t ind  = (otmp - omega_min_ip) / omega_delta;
            if(!omega_ip_conv_a[ind]) {
              oe_add = true;
              break;
            }
          }
          if(oe_add) {
            T Win = (w1 + w2) / 2;
            Win   = find_closest(Win, omega_space_ip);
            if(std::find(omega_extra_finished.begin(), omega_extra_finished.end(), Win) !=
               omega_extra_finished.end()) {
              continue;
            }
            else { omega_extra.push_back(Win); }
          } // end oe add
        }   // end oe finished
      }
      if(rank == 0) {
        cout << "new freq's:" << std::fixed << std::setprecision(2) << omega_extra << endl;
        cout << "qr_rank_orig, qr_rank_updated: " << qr_rank_orig << ", " << qr_rank_updated
             << endl;
      }

      // extrapolate or proceed to next level
      bool conv_all =
        std::all_of(omega_ip_conv_a.begin(), omega_ip_conv_a.end(), [](bool x) { return x; });

      if(conv_all || gf_extrapolate_level == level || omega_extra.size() == 0) {
        if(rank == 0)
          cout << endl
               << "--------------------extrapolate & converge-----------------------" << endl;
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t taskcount = 0;
        int64_t next      = ac->fetch_add(0, 1);

        for(int64_t ni = 0; ni < lomega_npts_ip; ni++) {
          if(next == taskcount) {
            std::complex<T> omega_tmp =
              std::complex<T>(lomega_min_ip + ni * omega_delta_e, -1.0 * gf_eta);

            Complex2DMatrix xsub_a = (hsub_a + omega_tmp * hident).lu().solve(bsub_a);
            eigen_to_tamm_tensor(xsub_local_a, xsub_a);

            sch_l(o_local_a(h1_oa) = Cp_local_a(h1_oa, otil) * xsub_local_a(otil, h1_oa)).execute();

            auto oscalar = std::imag(tamm::sum(o_local_a));

            Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(noa);
            tamm_to_eigen_tensor(o_local_a, olocala_eig);
            for(TAMM_SIZE nj = 0; nj < noa; nj++) {
              auto gpp = olocala_eig(nj).imag();
              spfe << "orb_index = " << nj << ", gpp_a = " << gpp << endl;
            }

            spfe << "w = " << std::fixed << std::setprecision(3) << std::real(omega_tmp)
                 << ", A_a =  " << std::fixed << std::setprecision(6) << oscalar << endl;
            next = ac->fetch_add(0, 1);
          }
          taskcount++;
        }

        ec.pg().barrier();
        ac->deallocate();
        delete ac;

        write_string_to_disk(ec, spfe.str(), extrap_file);
        if(rank == 0) {
          sys_data.results["output"]["GFCCSD"]["retarded_alpha"]["nlevels"] = level;
          write_json_data(sys_data, "GFCCSD");
        }
        sch.deallocate(xsub_local_a, o_local_a, Cp_local_a, hsub_tamm_a, bsub_tamm_a, Cp_a)
          .execute();
        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          std::cout << endl
                    << "Time taken for extrapolation (lomega_npts_ip = " << lomega_npts_ip
                    << "): " << time << " secs" << endl;
        break;
      }

      level++;

      sch.deallocate(xsub_local_a, o_local_a, Cp_local_a, hsub_tamm_a, bsub_tamm_a, Cp_a).execute();

    } // end while
    // end of alpha

    level               = 1;
    size_t prev_qr_rank = 0;

////////////////////////////////////////////////
//                                            //
// if open_shell, then execute retarded_beta  //
//                                            //
////////////////////////////////////////////////
#if 0
    if(ccsd_options.gf_os && rank == 0) {
      cout << endl << "_____retarded_GFCCSD_on_beta_spin______" << endl;
      // ofs_profile << endl <<"_____retarded_GFCCSD_on_beta_spin______" << endl;
    }

    while(ccsd_options.gf_os) {
      const std::string levelstr = std::to_string(level);

      std::string q1_b_file    = files_prefix + ".r_q1_b.l" + levelstr;
      std::string q2_bbb_file  = files_prefix + ".r_q2_bbb.l" + levelstr;
      std::string q2_aba_file  = files_prefix + ".r_q2_aba.l" + levelstr;
      std::string hx1_b_file   = files_prefix + ".r_hx1_b.l" + levelstr;
      std::string hx2_bbb_file = files_prefix + ".r_hx2_bbb.l" + levelstr;
      std::string hx2_aba_file = files_prefix + ".r_hx2_aba.l" + levelstr;
      std::string hsub_b_file  = files_prefix + ".r_hsub_b.l" + levelstr;
      std::string bsub_b_file  = files_prefix + ".r_bsub_b.l" + levelstr;
      std::string cp_b_file    = files_prefix + ".r_cp_b.l" + levelstr;

      bool gf_restart = fs::exists(q1_b_file) && fs::exists(q2_bbb_file) &&
                        fs::exists(q2_aba_file) && fs::exists(hx1_b_file) &&
                        fs::exists(hx2_bbb_file) && fs::exists(hx2_aba_file) &&
                        fs::exists(hsub_b_file) && fs::exists(bsub_b_file) &&
                        fs::exists(cp_b_file) && ccsd_options.gf_restart;

      if(level == 1) {
        omega_extra.clear();
        omega_extra_finished.clear();
        omega_extra.push_back(omega_min_ip);
        omega_extra.push_back(omega_max_ip);
      }

      for(auto x: omega_extra) omega_extra_finished.push_back(x);

      auto qr_rank = omega_extra_finished.size() * nob;

      TiledIndexSpace otis;
      if(ndiis > qr_rank) {
        diis_tis = {IndexSpace{range(0, ndiis)}};
        otis     = {diis_tis, range(0, qr_rank)};
      }
      else {
        otis     = {IndexSpace{range(qr_rank)}};
        diis_tis = {otis, range(0, ndiis)};
      }

      TiledIndexSpace unit_tis{diis_tis, range(0, 1)};
      // auto [u1] = unit_tis.labels<1>("all");

      for(auto x: omega_extra) {
        ndiis    = ccsd_options.gf_ndiis;
        gf_omega = x;

        if(!gf_restart) {
          gfccsd_driver_ip_b<T>(ec, *sub_ec, subcomm, MO, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                                d_t2_abab, d_f1, t2v2_o, lt12_o_a, lt12_o_b, ix1_1_1_a, ix1_1_1_b,
                                ix2_1_aaaa, ix2_1_abab, ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b,
                                ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab, ix2_4_bbbb, ix2_5_aaaa,
                                ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                                ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab,
                                ix2_6_3_bbbb, ix2_6_3_baab, ix2_6_3_baba, v2ijab_aaaa, v2ijab_abab,
                                v2ijab_bbbb, p_evl_sorted_occ, p_evl_sorted_virt, total_orbitals,
                                nocc, nvir, nptsi, unit_tis, files_prefix, levelstr, noa, nob);
        }
      }

      ComplexTensor q1_tamm_b{o_beta, otis};
      ComplexTensor q2_tamm_bbb{v_beta, o_beta, o_beta, otis};
      ComplexTensor q2_tamm_aba{v_alpha, o_beta, o_alpha, otis};
      ComplexTensor Hx1_tamm_b{o_beta, otis};
      ComplexTensor Hx2_tamm_bbb{v_beta, o_beta, o_beta, otis};
      ComplexTensor Hx2_tamm_aba{v_alpha, o_beta, o_alpha, otis};

      if(!gf_restart) {
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        sch.allocate(q1_tamm_b, q2_tamm_bbb, q2_tamm_aba, Hx1_tamm_b, Hx2_tamm_bbb, Hx2_tamm_aba)
          .execute();

        const std::string plevelstr = std::to_string(level - 1);

        std::string pq1_b_file   = files_prefix + ".r_q1_b.l" + plevelstr;
        std::string pq2_bbb_file = files_prefix + ".r_q2_bbb.l" + plevelstr;
        std::string pq2_aba_file = files_prefix + ".r_q2_aba.l" + plevelstr;

        decltype(qr_rank) ivec_start = 0;
        bool              prev_q12   = fs::exists(pq1_b_file) && fs::exists(pq2_aba_file) &&
                        fs::exists(pq2_bbb_file);

        if(prev_q12) {
          TiledIndexSpace otis_prev{otis, range(0, prev_qr_rank)};
          auto [op1] = otis_prev.labels<1>("all");
          ComplexTensor q1_prev_b{o_beta, otis_prev};
          ComplexTensor q2_prev_bbb{v_beta, o_beta, o_beta, otis_prev};
          ComplexTensor q2_prev_aba{v_alpha, o_beta, o_alpha, otis_prev};
          sch.allocate(q1_prev_b, q2_prev_bbb, q2_prev_aba).execute();

          read_from_disk(q1_prev_b, pq1_b_file);
          read_from_disk(q2_prev_bbb, pq2_bbb_file);
          read_from_disk(q2_prev_aba, pq2_aba_file);

          ivec_start = prev_qr_rank;

          if(subcomm != MPI_COMM_NULL) {
            // clang-format off
            sub_sch
              (q1_tamm_b(h1_ob,op1) = q1_prev_b(h1_ob,op1))
              (q2_tamm_bbb(p1_vb,h1_ob,h2_ob,op1) = q2_prev_bbb(p1_vb,h1_ob,h2_ob,op1))
              (q2_tamm_aba(p1_va,h1_ob,h2_oa,op1) = q2_prev_aba(p1_va,h1_ob,h2_oa,op1)).execute();
            // clang-format on
          }
          sch.deallocate(q1_prev_b, q2_prev_bbb, q2_prev_aba).execute();
        }

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          cout << endl
               << "Time to read in pre-computed Q1/Q2: " << std::fixed << std::setprecision(6)
               << time << " secs" << endl;

        ComplexTensor q1_tmp_b{o_beta};
        ComplexTensor q2_tmp_bbb{v_beta, o_beta, o_beta};
        ComplexTensor q2_tmp_aba{v_alpha, o_beta, o_alpha};

        // TODO: optimize Q1/Q2 computation
        // Gram-Schmidt orthogonalization
        double time_gs_orth  = 0.0;
        double time_gs_norm  = 0.0;
        double total_time_gs = 0.0;

        bool q_exist = fs::exists(q1_b_file) && fs::exists(q2_bbb_file) && fs::exists(q2_aba_file);

        if(!q_exist) {
          sch.allocate(q1_tmp_b, q2_tmp_bbb, q2_tmp_aba).execute();

          for(decltype(qr_rank) ivec = ivec_start; ivec < qr_rank; ivec++) {
            auto cc_t0 = std::chrono::high_resolution_clock::now();

            auto              W_read  = omega_extra_finished[ivec / (nob)];
            auto              pi_read = ivec % (nob) + noa;
            std::stringstream gfo;
            gfo << std::fixed << std::setprecision(2) << W_read;

            std::string x1_b_wpi_file =
              files_prefix + ".x1_b.W" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string x2_bbb_wpi_file =
              files_prefix + ".x2_bbb.W" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string x2_aba_wpi_file =
              files_prefix + ".x2_aba.W" + gfo.str() + ".oi" + std::to_string(pi_read);

            if(fs::exists(x1_b_wpi_file) && fs::exists(x2_bbb_wpi_file) &&
               fs::exists(x2_aba_wpi_file)) {
              read_from_disk(q1_tmp_b, x1_b_wpi_file);
              read_from_disk(q2_tmp_bbb, x2_bbb_wpi_file);
              read_from_disk(q2_tmp_aba, x2_aba_wpi_file);
            }
            else {
              tamm_terminate("ERROR: At least one of " + x1_b_wpi_file + " and " + x2_bbb_wpi_file +
                             " and " + x2_aba_wpi_file + " do not exist!");
            }

            // TODO: schedule all iterations before executing
            if(ivec > 0) {
              TiledIndexSpace tsc{otis, range(0, ivec)};
              auto [sc] = tsc.labels<1>("all");

              ComplexTensor oscalar{tsc};
              ComplexTensor x1c_b{o_beta, tsc};
              ComplexTensor x2c_bbb{v_beta, o_beta, o_beta, tsc};
              ComplexTensor x2c_aba{v_alpha, o_beta, o_alpha, tsc};

              // clang-format off

              #if GF_GS_SG
              if(subcomm != MPI_COMM_NULL) {
                sub_sch.allocate
              #else
                sch.allocate
              #endif
                  (x1c_b,x2c_bbb,x2c_aba)
                  (x1c_b(h1_ob,sc) = q1_tamm_b(h1_ob,sc))
                  (x2c_bbb(p1_vb,h1_ob,h2_ob,sc) = q2_tamm_bbb(p1_vb,h1_ob,h2_ob,sc))
                  (x2c_aba(p1_va,h1_ob,h2_oa,sc) = q2_tamm_aba(p1_va,h1_ob,h2_oa,sc))
                  .execute();
              // clang-format on  
  
              tamm::conj_ip(x1c_b);
              tamm::conj_ip(x2c_bbb);
              tamm::conj_ip(x2c_aba);
  
              // clang-format off
              #if GF_GS_SG
                sub_sch.allocate
              #else
                sch.allocate
              #endif
                  (oscalar)
                  (oscalar(sc)  = -1.0 * q1_tmp_b(h1_ob) * x1c_b(h1_ob,sc))
                  (oscalar(sc) += -0.5 * q2_tmp_bbb(p1_vb,h1_ob,h2_ob) * x2c_bbb(p1_vb,h1_ob,h2_ob,sc))
                  (oscalar(sc) += -1.0 * q2_tmp_aba(p1_va,h1_ob,h2_oa) * x2c_aba(p1_va,h1_ob,h2_oa,sc))

                  (q1_tmp_b(h1_ob) += oscalar(sc) * q1_tamm_b(h1_ob,sc))
                  (q2_tmp_bbb(p1_vb,h1_ob,h2_ob) += oscalar(sc) * q2_tamm_bbb(p1_vb,h1_ob,h2_ob,sc))
                  (q2_tmp_aba(p1_va,h1_ob,h2_oa) += oscalar(sc) * q2_tamm_aba(p1_va,h1_ob,h2_oa,sc))
                  .deallocate(x1c_b,x2c_bbb,x2c_aba,oscalar).execute();
              #if GF_GS_SG
              }
              ec.pg().barrier();
              #endif
              // clang-format on
            }

            auto cc_t1 = std::chrono::high_resolution_clock::now();
            time_gs_orth +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_t1 - cc_t0)).count();

            auto q1norm_b   = norm(q1_tmp_b);
            auto q2norm_bbb = norm(q2_tmp_bbb);
            auto q2norm_aba = norm(q2_tmp_aba);

            // Normalization factor
            T newsc = 1.0 / std::real(sqrt(q1norm_b * q1norm_b + 0.5 * q2norm_bbb * q2norm_bbb +
                                           q2norm_aba * q2norm_aba));

            std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);

            TiledIndexSpace tsc{otis, range(ivec, ivec + 1)};
            auto [sc] = tsc.labels<1>("all");

            if(subcomm != MPI_COMM_NULL) {
              // clang-format off
              sub_sch
              (q1_tamm_b(h1_ob,sc) = cnewsc * q1_tmp_b(h1_ob))
              (q2_tamm_bbb(p2_vb,h1_ob,h2_ob,sc) = cnewsc * q2_tmp_bbb(p2_vb,h1_ob,h2_ob))
              (q2_tamm_aba(p2_va,h1_ob,h2_oa,sc) = cnewsc * q2_tmp_aba(p2_va,h1_ob,h2_oa))
              .execute();
              // clang-format on
            }
            ec.pg().barrier();

            auto cc_gs = std::chrono::high_resolution_clock::now();
            time_gs_norm +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
            total_time_gs +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t0)).count();

          } // end of Gram-Schmidt loop

          sch.deallocate(q1_tmp_b, q2_tmp_bbb, q2_tmp_aba).execute();

          write_to_disk(q1_tamm_b, q1_b_file);
          write_to_disk(q2_tamm_bbb, q2_bbb_file);
          write_to_disk(q2_tamm_aba, q2_aba_file);
        }      // end of !gs-restart
        else { // restart GS
          read_from_disk(q1_tamm_b, q1_b_file);
          read_from_disk(q2_tamm_bbb, q2_bbb_file);
          read_from_disk(q2_tamm_aba, q2_aba_file);
        }

        if(rank == 0) {
          cout << endl << "Time for orthogonalization: " << time_gs_orth << " secs" << endl;
          cout << endl
               << "Time for normalizing and copying back: " << time_gs_norm << " secs" << endl;
          cout << endl << "Total time for Gram-Schmidt: " << total_time_gs << " secs" << endl;
        }
        auto cc_gs_x = std::chrono::high_resolution_clock::now();

        bool gs_x12_restart = fs::exists(hx1_b_file) && fs::exists(hx2_bbb_file) &&
                              fs::exists(hx2_aba_file);

        if(!gs_x12_restart) {
#if GF_IN_SG
          if(subcomm != MPI_COMM_NULL) {
            gfccsd_x1_b(sub_sch,
#else
          gfccsd_x1_b(sch,
#endif
                        MO, Hx1_tamm_b, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, q1_tamm_b,
                        q2_tamm_bbb, q2_tamm_aba, d_f1, ix2_2_b, ix1_1_1_a, ix1_1_1_b, ix2_6_3_bbbb,
                        ix2_6_3_baba, otis, true);

#if GF_IN_SG
            gfccsd_x2_b(sub_sch,
#else
          gfccsd_x2_b(sch,
#endif
                        MO, Hx2_tamm_bbb, Hx2_tamm_aba, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                        d_t2_abab, q1_tamm_b, q2_tamm_bbb, q2_tamm_aba, d_f1, ix2_1_bbbb,
                        ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_bbbb, ix2_4_abab,
                        ix2_5_aaaa, ix2_5_abba, ix2_5_baba, ix2_5_bbbb, ix2_5_baab, ix2_6_2_a,
                        ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_baba, ix2_6_3_bbbb,
                        ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, otis, true);

#if GF_IN_SG
            sub_sch.execute();
          }
          ec.pg().barrier();
#else
          sch.execute();
#endif
          write_to_disk(Hx1_tamm_b, hx1_b_file);
          write_to_disk(Hx2_tamm_bbb, hx2_bbb_file);
          write_to_disk(Hx2_tamm_aba, hx2_aba_file);
        }
        else {
          read_from_disk(Hx1_tamm_b, hx1_b_file);
          read_from_disk(Hx2_tamm_bbb, hx2_bbb_file);
          read_from_disk(Hx2_tamm_aba, hx2_aba_file);
        }
        auto   cc_q12 = std::chrono::high_resolution_clock::now();
        double time_q12 =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
        if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;

      } // if !gf_restart

      prev_qr_rank = qr_rank;

      auto cc_t1 = std::chrono::high_resolution_clock::now();

      auto [otil, otil1, otil2] = otis.labels<3>("all");
      ComplexTensor hsub_tamm_b{otis, otis};
      ComplexTensor bsub_tamm_b{otis, o_beta};
      ComplexTensor Cp_b{o_beta, otis};
      ComplexTensor::allocate(&ec, hsub_tamm_b, bsub_tamm_b, Cp_b);

      if(!gf_restart) {
        ComplexTensor p1_k_b{v_beta, otis};
        ComplexTensor q1_conj_b   = tamm::conj(q1_tamm_b);
        ComplexTensor q2_conj_bbb = tamm::conj(q2_tamm_bbb);
        ComplexTensor q2_conj_aba = tamm::conj(q2_tamm_aba);

        // clang-format off
        sch
          (bsub_tamm_b(otil1,h1_ob)  =       q1_conj_b(h1_ob,otil1))
          (hsub_tamm_b(otil1,otil2)  =       q1_conj_b(h1_ob,otil1) * Hx1_tamm_b(h1_ob,otil2))
          (hsub_tamm_b(otil1,otil2) += 0.5 * q2_conj_bbb(p1_vb,h1_ob,h2_ob,otil1) * Hx2_tamm_bbb(p1_vb,h1_ob,h2_ob,otil2))
          (hsub_tamm_b(otil1,otil2) +=       q2_conj_aba(p1_va,h1_ob,h2_oa,otil1) * Hx2_tamm_aba(p1_va,h1_ob,h2_oa,otil2))
          .deallocate(q1_conj_b,q2_conj_bbb,q2_conj_aba)

          .allocate(p1_k_b)
          ( Cp_b(h1_ob,otil)    =        q1_tamm_b(h1_ob,otil)                                     )
          ( Cp_b(h2_ob,otil)   += -1.0 * lt12_o_b(h1_ob,h2_ob) * q1_tamm_b(h1_ob,otil)               )
          ( Cp_b(h2_ob,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aba(p1_va,h2_ob,h1_oa,otil) )
          ( Cp_b(h2_ob,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bbb(p1_vb,h2_ob,h1_ob,otil) )
          ( p1_k_b(p1_vb,otil)  =        d_t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob) * q2_tamm_bbb(p2_vb,h1_ob,h2_ob,otil) )
          ( p1_k_b(p1_vb,otil) +=  2.0 * d_t2_abab(p2_va,p1_vb,h2_oa,h1_ob) * q2_tamm_aba(p2_va,h1_ob,h2_oa,otil) )
          ( Cp_b(h1_ob,otil)   += -0.5 * p1_k_b(p1_vb,otil) * d_t1_b(p1_vb,h1_ob) )
          .deallocate(p1_k_b,q1_tamm_b, q2_tamm_bbb, q2_tamm_aba)
          .execute();
        // clang-format on

      } // if !gf_restart

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

      // Write all tensors
      if(!gf_restart) {
        write_to_disk(hsub_tamm_b, hsub_b_file);
        write_to_disk(bsub_tamm_b, bsub_b_file);
        write_to_disk(Cp_b, cp_b_file);
        sch.deallocate(Hx1_tamm_b, Hx2_tamm_bbb, Hx2_tamm_aba).execute();
      }
      else {
        read_from_disk(hsub_tamm_b, hsub_b_file);
        read_from_disk(bsub_tamm_b, bsub_b_file);
        read_from_disk(Cp_b, cp_b_file);
      }

      Complex2DMatrix hsub_b(qr_rank, qr_rank);
      Complex2DMatrix bsub_b(qr_rank, nob);

      tamm_to_eigen_tensor(hsub_tamm_b, hsub_b);
      tamm_to_eigen_tensor(bsub_tamm_b, bsub_b);
      Complex2DMatrix hident = Complex2DMatrix::Identity(hsub_b.rows(), hsub_b.cols());

      ComplexTensor xsub_local_b{otis, o_beta};
      ComplexTensor o_local_b{o_beta};
      ComplexTensor Cp_local_b{o_beta, otis};

      ComplexTensor::allocate(&ec_l, xsub_local_b, o_local_b, Cp_local_b);

      Scheduler sch_l{ec_l};
      sch_l(Cp_local_b(h1_ob, otil) = Cp_b(h1_ob, otil)).execute();

      if(rank == 0) {
        std::cout << endl << "spectral function (omega_npts_ip = " << omega_npts_ip << "):" << endl;
      }

      cc_t1 = std::chrono::high_resolution_clock::now();
      std::vector<double> ni_w(omega_npts_ip, 0);
      std::vector<double> ni_A(omega_npts_ip, 0);

      // Compute spectral function for designated omega regime
      for(int64_t ni = 0; ni < omega_npts_ip; ni++) {
        std::complex<T> omega_tmp = std::complex<T>(omega_min_ip + ni * omega_delta, -1.0 * gf_eta);

        Complex2DMatrix xsub_b = (hsub_b + omega_tmp * hident).lu().solve(bsub_b);
        eigen_to_tamm_tensor(xsub_local_b, xsub_b);

        sch_l(o_local_b(h1_ob) = Cp_local_b(h1_ob, otil) * xsub_local_b(otil, h1_ob)).execute();

        auto oscalar = std::imag(tamm::sum(o_local_b));

        if(level == 1) { omega_ip_A0[ni] = oscalar; }
        else {
          if(level > 1) {
            T oerr          = oscalar - omega_ip_A0[ni];
            omega_ip_A0[ni] = oscalar;
            if(std::abs(oerr) < gf_threshold) omega_ip_conv_b[ni] = true;
          }
        }
        if(rank == 0) {
          std::ostringstream spf;
          spf << "W = " << std::fixed << std::setprecision(2) << std::real(omega_tmp)
              << ", omega_ip_A0 =  " << std::fixed << std::setprecision(4) << omega_ip_A0[ni]
              << endl;
          cout << spf.str();
          ni_A[ni] = omega_ip_A0[ni];
          ni_w[ni] = std::real(omega_tmp);
        }
      }

      cc_t2 = std::chrono::high_resolution_clock::now();
      time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) {
        cout << endl
             << "omegas processed in level " << level << " = " << std::fixed << std::setprecision(2)
             << omega_extra << endl;
        cout << "Time to compute spectral function in level " << level
             << " (omega_npts_ip = " << omega_npts_ip << "): " << time << " secs" << endl;
        write_results_to_json(ec, sys_data, level, ni_w, ni_A, "retarded_beta");
      }

      auto               extrap_file = files_prefix + ".extrapolate.retarded.beta.txt";
      std::ostringstream spfe;
      spfe << "";

      // extrapolate or proceed to next level
      if(std::all_of(omega_ip_conv_b.begin(), omega_ip_conv_b.end(), [](bool x) { return x; }) ||
         gf_extrapolate_level == level) {
        if(rank == 0)
          cout << endl
               << "--------------------extrapolate & converge-----------------------" << endl;
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t taskcount = 0;
        int64_t next      = ac->fetch_add(0, 1);

        for(int64_t ni = 0; ni < lomega_npts_ip; ni++) {
          if(next == taskcount) {
            std::complex<T> omega_tmp =
              std::complex<T>(lomega_min_ip + ni * omega_delta_e, -1.0 * gf_eta);

            Complex2DMatrix xsub_b = (hsub_b + omega_tmp * hident).lu().solve(bsub_b);
            eigen_to_tamm_tensor(xsub_local_b, xsub_b);

            sch_l(o_local_b(h1_ob) = Cp_local_b(h1_ob, otil) * xsub_local_b(otil, h1_ob)).execute();

            auto oscalar = std::imag(tamm::sum(o_local_b));

            Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(noa);
            tamm_to_eigen_tensor(o_local_b, olocala_eig);
            for(TAMM_SIZE nj = 0; nj < noa; nj++) {
              auto gpp = olocala_eig(nj).imag();
              spfe << "orb_index = " << nj << ", gpp_b = " << gpp << endl;
            }

            spfe << "w = " << std::fixed << std::setprecision(3) << std::real(omega_tmp)
                 << ", A_b =  " << std::fixed << std::setprecision(6) << oscalar << endl;
            next = ac->fetch_add(0, 1);
          }
          taskcount++;
        }

        ec.pg().barrier();
        ac->deallocate();
        delete ac;

        write_string_to_disk(ec, spfe.str(), extrap_file);
        if(rank == 0) {
          sys_data.results["output"]["GFCCSD"]["retarded_beta"]["nlevels"] = level;
          write_json_data(sys_data, "GFCCSD");
        }

        sch.deallocate(xsub_local_b, o_local_b, Cp_local_b, hsub_tamm_b, bsub_tamm_b, Cp_b)
          .execute();

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          std::cout << endl
                    << "Time taken for extrapolation (lomega_npts_ip = " << lomega_npts_ip
                    << "): " << time << " secs" << endl;

        break;
      }
      else {
        if(level == 1) {
          auto o1 = (omega_extra[0] + omega_extra[1]) / 2;
          omega_extra.clear();
          o1 = find_closest(o1, omega_space_ip);
          omega_extra.push_back(o1);
        }
        else {
          std::sort(omega_extra_finished.begin(), omega_extra_finished.end());
          omega_extra.clear();
          std::vector<T> wtemp;
          for(size_t i = 1; i < omega_extra_finished.size(); i++) {
            bool   oe_add = false;
            auto   w1     = omega_extra_finished[i - 1];
            auto   w2     = omega_extra_finished[i];
            size_t num_w  = (w2 - w1) / omega_delta + 1;
            for(size_t j = 0; j < num_w; j++) {
              T      otmp = w1 + j * omega_delta;
              size_t ind  = (otmp - omega_min_ip) / omega_delta;
              if(!omega_ip_conv_b[ind]) {
                oe_add = true;
                break;
              }
            }

            if(oe_add) {
              T Win = (w1 + w2) / 2;
              Win   = find_closest(Win, omega_space_ip);
              if(std::find(omega_extra_finished.begin(), omega_extra_finished.end(), Win) !=
                 omega_extra_finished.end()) {
                continue;
              }
              else { omega_extra.push_back(Win); }
            } // end oe add
          }   // end oe finished
        }
        if(rank == 0) {
          cout << "new freq's:" << std::fixed << std::setprecision(2) << omega_extra << endl;
        }
        level++;
      }

      sch.deallocate(xsub_local_b, o_local_b, Cp_local_b, hsub_tamm_b, bsub_tamm_b, Cp_b).execute();

    } // end while
    // end of ip beta
#endif
  }

///////// retarded part is done, perform advanced part if needed /////////////
#if 0
  if(ccsd_options.gf_ea) {
    std::string t2v2_v_file     = files_prefix + ".t2v2_v";
    std::string lt12_v_a_file   = files_prefix + ".lt12_v_a";
    std::string lt12_v_b_file   = files_prefix + ".lt12_v_b";
    std::string iy1_1_a_file    = files_prefix + ".iy1_a";      //
    std::string iy1_1_b_file    = files_prefix + ".iy1_b";      //
    std::string iy1_2_1_a_file  = files_prefix + ".iy1_2_1_a";  //
    std::string iy1_2_1_b_file  = files_prefix + ".iy1_2_1_b";  //
    std::string iy1_a_file      = files_prefix + ".iy1_a";      //
    std::string iy1_b_file      = files_prefix + ".iy1_b";      //
    std::string iy2_a_file      = files_prefix + ".iy2_a";      //
    std::string iy2_b_file      = files_prefix + ".iy2_b";      //
    std::string iy3_1_aaaa_file = files_prefix + ".iy3_1_aaaa"; //
    std::string iy3_1_abba_file = files_prefix + ".iy3_1_abba"; //
    std::string iy3_1_baba_file = files_prefix + ".iy3_1_baba"; //
    std::string iy3_1_bbbb_file = files_prefix + ".iy3_1_bbbb"; //
    std::string iy3_1_baab_file = files_prefix + ".iy3_1_baab"; //
    std::string iy3_1_abab_file = files_prefix + ".iy3_1_abab"; //
    std::string iy3_1_2_a_file  = files_prefix + ".iy3_1_2_a";  //
    std::string iy3_1_2_b_file  = files_prefix + ".iy3_1_2_b";  //
    std::string iy3_aaaa_file   = files_prefix + ".iy3_aaaa";   //
    std::string iy3_baab_file   = files_prefix + ".iy3_baab";   //
    std::string iy3_abba_file   = files_prefix + ".iy3_abba";   //
    std::string iy3_bbbb_file   = files_prefix + ".iy3_bbbb";   //
    std::string iy3_abab_file   = files_prefix + ".iy3_abab";   //
    std::string iy3_baba_file   = files_prefix + ".iy3_baba";   //
    std::string iy4_1_aaaa_file = files_prefix + ".iy4_1_aaaa"; //
    std::string iy4_1_baab_file = files_prefix + ".iy4_1_baab"; //
    std::string iy4_1_baba_file = files_prefix + ".iy4_1_baba"; //
    std::string iy4_1_bbbb_file = files_prefix + ".iy4_1_bbbb"; //
    std::string iy4_1_abba_file = files_prefix + ".iy4_1_abba"; //
    std::string iy4_1_abab_file = files_prefix + ".iy4_1_abab"; //
    std::string iy4_2_aaaa_file = files_prefix + ".iy4_2_aaaa"; //
    std::string iy4_2_abba_file = files_prefix + ".iy4_2_abba"; //
    std::string iy4_2_bbbb_file = files_prefix + ".iy4_2_bbbb"; //
    std::string iy4_2_baab_file = files_prefix + ".iy4_2_baab"; //
    std::string iy5_aaaa_file   = files_prefix + ".iy5_aaaa";   //
    std::string iy5_baab_file   = files_prefix + ".iy5_baab";   //
    std::string iy5_baba_file   = files_prefix + ".iy5_baba";   //
    std::string iy5_bbbb_file   = files_prefix + ".iy5_bbbb";   //
    std::string iy5_abba_file   = files_prefix + ".iy5_abba";   //
    std::string iy5_abab_file   = files_prefix + ".iy5_abab";   //
    std::string iy6_a_file      = files_prefix + ".iy6_a";      //
    std::string iy6_b_file      = files_prefix + ".iy6_b";      //

    t2v2_v     = Tensor<T>{{V, V}, {1, 1}};
    lt12_v_a   = Tensor<T>{v_alpha, v_alpha};
    lt12_v_b   = Tensor<T>{v_beta, v_beta};
    iy1_1_a    = Tensor<T>{v_alpha, v_alpha};
    iy1_1_b    = Tensor<T>{v_beta, v_beta};
    iy1_2_1_a  = Tensor<T>{o_alpha, v_alpha};
    iy1_2_1_b  = Tensor<T>{o_beta, v_beta};
    iy1_a      = Tensor<T>{v_alpha, v_alpha};
    iy1_b      = Tensor<T>{v_beta, v_beta};
    iy2_a      = Tensor<T>{o_alpha, o_alpha};
    iy2_b      = Tensor<T>{o_beta, o_beta};
    iy3_1_aaaa = Tensor<T>{o_alpha, v_alpha, o_alpha, v_alpha};
    iy3_1_abba = Tensor<T>{o_alpha, v_beta, o_beta, v_alpha};
    iy3_1_baba = Tensor<T>{o_beta, v_alpha, o_beta, v_alpha};
    iy3_1_bbbb = Tensor<T>{o_beta, v_beta, o_beta, v_beta};
    iy3_1_baab = Tensor<T>{o_beta, v_alpha, o_alpha, v_beta};
    iy3_1_abab = Tensor<T>{o_alpha, v_beta, o_alpha, v_beta};
    iy3_1_2_a  = Tensor<T>{o_alpha, v_alpha, CI};
    iy3_1_2_b  = Tensor<T>{o_beta, v_beta, CI};
    iy3_aaaa   = Tensor<T>{o_alpha, v_alpha, o_alpha, v_alpha};
    iy3_baab   = Tensor<T>{o_beta, v_alpha, o_alpha, v_beta};
    iy3_abba   = Tensor<T>{o_alpha, v_beta, o_beta, v_alpha};
    iy3_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, v_beta};
    iy3_abab   = Tensor<T>{o_alpha, v_beta, o_alpha, v_beta};
    iy3_baba   = Tensor<T>{o_beta, v_alpha, o_beta, v_alpha};
    iy4_1_aaaa = Tensor<T>{o_alpha, o_alpha, o_alpha, v_alpha};
    iy4_1_baab = Tensor<T>{o_beta, o_alpha, o_alpha, v_beta};
    iy4_1_baba = Tensor<T>{o_beta, o_alpha, o_beta, v_alpha};
    iy4_1_bbbb = Tensor<T>{o_beta, o_beta, o_beta, v_beta};
    iy4_1_abba = Tensor<T>{o_alpha, o_beta, o_beta, v_alpha};
    iy4_1_abab = Tensor<T>{o_alpha, o_beta, o_alpha, v_beta};
    iy4_2_aaaa = Tensor<T>{o_alpha, o_alpha, o_alpha, v_alpha};
    iy4_2_abba = Tensor<T>{o_alpha, o_beta, o_beta, v_alpha};
    iy4_2_bbbb = Tensor<T>{o_beta, o_beta, o_beta, v_beta};
    iy4_2_baab = Tensor<T>{o_beta, o_alpha, o_alpha, v_beta};
    iy5_aaaa   = Tensor<T>{o_alpha, v_alpha, o_alpha, v_alpha};
    iy5_baab   = Tensor<T>{o_beta, v_alpha, o_alpha, v_beta};
    iy5_baba   = Tensor<T>{o_beta, v_alpha, o_beta, v_alpha};
    iy5_bbbb   = Tensor<T>{o_beta, v_beta, o_beta, v_beta};
    iy5_abba   = Tensor<T>{o_alpha, v_beta, o_beta, v_alpha};
    iy5_abab   = Tensor<T>{o_alpha, v_beta, o_alpha, v_beta};
    iy6_a      = Tensor<T>{o_alpha, v_alpha, CI};
    iy6_b      = Tensor<T>{o_beta, v_beta, CI};

    Tensor<T> lt12_v{{V, V}, {1, 1}};
    Tensor<T> iy1_1_1{CI};
    Tensor<T> iy1_1_2{{V, V, CI}, {1, 1}};
    Tensor<T> iy1_1{{V, V}, {1, 1}};
    Tensor<T> iy1_2_1{{O, V}, {1, 1}};
    Tensor<T> iy1{{V, V}, {1, 1}};
    Tensor<T> iy2{{O, O}, {1, 1}};
    Tensor<T> iy3_1_1{{O, O, CI}, {1, 1}};
    Tensor<T> iy3_1_2{{O, V, CI}, {1, 1}};
    Tensor<T> iy3_1{{O, V, O, V}, {2, 2}};
    Tensor<T> iy3{{O, V, O, V}, {2, 2}};
    Tensor<T> iy4_1{{O, O, O, V}, {2, 2}};
    Tensor<T> iy4_2{{O, O, O, V}, {2, 2}};
    Tensor<T> iy6{{O, V, CI}, {1, 1}};

    sch
      .allocate(t2v2_v, lt12_v_a, lt12_v_b, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy1_a, iy1_b,
                iy2_a, iy2_b, iy3_1_aaaa, iy3_1_bbbb, iy3_1_abab, iy3_1_baba, iy3_1_baab,
                iy3_1_abba, iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_bbbb, iy3_abab, iy3_baba, iy3_baab,
                iy3_abba, iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb, iy4_1_abba, iy4_1_abab,
                iy4_2_aaaa, iy4_2_baab, iy4_2_bbbb, iy4_2_abba, iy5_aaaa, iy5_abab, iy5_baab,
                iy5_bbbb, iy5_baba, iy5_abba, iy6_a, iy6_b)
      .execute();

    if(fs::exists(t2v2_v_file) && fs::exists(lt12_v_a_file) && fs::exists(lt12_v_b_file) &&
       fs::exists(iy1_1_a_file) && fs::exists(iy1_1_b_file) && fs::exists(iy1_2_1_a_file) &&
       fs::exists(iy1_2_1_b_file) && fs::exists(iy1_a_file) && fs::exists(iy1_b_file) &&
       fs::exists(iy2_a_file) && fs::exists(iy2_b_file) && fs::exists(iy3_1_aaaa_file) &&
       fs::exists(iy3_1_bbbb_file) && fs::exists(iy3_1_abab_file) && fs::exists(iy3_1_baba_file) &&
       fs::exists(iy3_1_baab_file) && fs::exists(iy3_1_abba_file) && fs::exists(iy3_1_2_a_file) &&
       fs::exists(iy3_1_2_b_file) && fs::exists(iy3_aaaa_file) && fs::exists(iy3_bbbb_file) &&
       fs::exists(iy3_abab_file) && fs::exists(iy3_baba_file) && fs::exists(iy3_baab_file) &&
       fs::exists(iy3_abba_file) && fs::exists(iy4_1_aaaa_file) && fs::exists(iy4_1_baab_file) &&
       fs::exists(iy4_1_baba_file) && fs::exists(iy4_1_bbbb_file) && fs::exists(iy4_1_abba_file) &&
       fs::exists(iy4_1_abab_file) && fs::exists(iy4_2_aaaa_file) && fs::exists(iy4_2_baab_file) &&
       fs::exists(iy4_2_bbbb_file) && fs::exists(iy4_2_abba_file) && fs::exists(iy5_aaaa_file) &&
       fs::exists(iy5_abab_file) && fs::exists(iy5_baab_file) && fs::exists(iy5_bbbb_file) &&
       fs::exists(iy5_baba_file) && fs::exists(iy5_abba_file) && fs::exists(iy6_a_file) &&
       fs::exists(iy6_b_file) && ccsd_options.gf_restart) {
      read_from_disk(t2v2_v, t2v2_v_file);
      read_from_disk(lt12_v_a, lt12_v_a_file);
      read_from_disk(lt12_v_b, lt12_v_b_file);
      read_from_disk(iy1_1_a, iy1_1_a_file);
      read_from_disk(iy1_1_b, iy1_1_b_file);
      read_from_disk(iy1_2_1_a, iy1_2_1_a_file);
      read_from_disk(iy1_2_1_b, iy1_2_1_b_file);
      read_from_disk(iy1_a, iy1_a_file);
      read_from_disk(iy1_b, iy1_b_file);
      read_from_disk(iy2_a, iy1_a_file);
      read_from_disk(iy2_b, iy1_b_file);
      read_from_disk(iy3_1_aaaa, iy3_1_aaaa_file);
      read_from_disk(iy3_1_bbbb, iy3_1_bbbb_file);
      read_from_disk(iy3_1_abab, iy3_1_abab_file);
      read_from_disk(iy3_1_baba, iy3_1_baba_file);
      read_from_disk(iy3_1_baab, iy3_1_baab_file);
      read_from_disk(iy3_1_abba, iy3_1_abba_file);
      read_from_disk(iy3_1_2_a, iy3_1_2_a_file);
      read_from_disk(iy3_1_2_b, iy3_1_2_b_file);
      read_from_disk(iy3_aaaa, iy3_aaaa_file);
      read_from_disk(iy3_bbbb, iy3_bbbb_file);
      read_from_disk(iy3_abab, iy3_abab_file);
      read_from_disk(iy3_baba, iy3_baba_file);
      read_from_disk(iy3_baab, iy3_baab_file);
      read_from_disk(iy3_abba, iy3_abba_file);
      read_from_disk(iy4_1_aaaa, iy4_1_aaaa_file);
      read_from_disk(iy4_1_bbbb, iy4_1_bbbb_file);
      read_from_disk(iy4_1_baba, iy4_1_baba_file);
      read_from_disk(iy4_1_baab, iy4_1_baab_file);
      read_from_disk(iy4_1_abba, iy4_1_abba_file);
      read_from_disk(iy4_1_abab, iy4_1_abab_file);
      read_from_disk(iy4_2_aaaa, iy4_2_aaaa_file);
      read_from_disk(iy4_2_bbbb, iy4_2_bbbb_file);
      read_from_disk(iy4_2_baab, iy4_2_baab_file);
      read_from_disk(iy4_2_abba, iy4_2_abba_file);
      read_from_disk(iy5_aaaa, iy5_aaaa_file);
      read_from_disk(iy5_bbbb, iy5_bbbb_file);
      read_from_disk(iy5_abab, iy5_abab_file);
      read_from_disk(iy5_baba, iy5_baba_file);
      read_from_disk(iy5_baab, iy5_baab_file);
      read_from_disk(iy5_abba, iy5_abba_file);
      read_from_disk(iy6_a, iy6_a_file);
      read_from_disk(iy6_b, iy6_b_file);
    }
    else {
      // clang-format off

      #if GF_IN_SG
      if(subcomm != MPI_COMM_NULL) {
        sub_sch
      #else
        sch
      #endif
          .allocate(lt12_v,
                    iy1_1_1,iy1_1_2,iy1_1,iy1_2_1,iy1,
                    iy2,
                    iy3_1_1,iy3_1_2,iy3_1,iy3,
                    iy4_1,iy4_2,
                    iy6)
          ( t2v2_v(p1,p2)            =  0.5 * d_t2(p1,p3,h1,h2) * v2ijab(h1,h2,p3,p2)                       )
          ( lt12_v(p1_va,p3_va)      = -0.5 * d_t2(p1_va,p2_va,h1_oa,h2_oa) * d_t2(p3_va,p2_va,h1_oa,h2_oa) )
          ( lt12_v(p1_va,p3_va)     += -1.0 * d_t2(p1_va,p2_vb,h1_oa,h2_ob) * d_t2(p3_va,p2_vb,h1_oa,h2_ob) )
          ( lt12_v(p1_va,p2_va)     += -1.0 * d_t1(p1_va,h1_oa) * d_t1(p2_va,h1_oa)                         )
          ( lt12_v(p1_vb,p3_vb)      = -0.5 * d_t2(p1_vb,p2_vb,h1_ob,h2_ob) * d_t2(p3_vb,p2_vb,h1_ob,h2_ob) )
          ( lt12_v(p1_vb,p3_vb)     += -1.0 * d_t2(p2_va,p1_vb,h2_oa,h1_ob) * d_t2(p2_va,p3_vb,h2_oa,h1_ob) )
          ( lt12_v(p1_vb,p2_vb)     += -1.0 * d_t1(p1_vb,h1_ob) * d_t1(p2_vb,h1_ob)                         )
          (   iy1_1(p2,p6)           =  1.0 * d_f1(p2,p6)                                                   )
          (     iy1_1_1(cind)        =  1.0 * d_t1(p3,h4) * cholVpr(h4,p3,cind)                             )
          (   iy1_1(p2,p6)          +=  1.0 * iy1_1_1(cind) * cholVpr(p2,p6,cind)                           )
          (     iy1_1_2(p3,p6,cind)  =  1.0 * d_t1(p3,h4) * cholVpr(h4,p6,cind)                             )
          (   iy1_1(p2,p6)          += -1.0 * iy1_1_2(p3,p6,cind) * cholVpr(p2,p3,cind)                     )
          (   iy1_1(p2,p7)          +=  0.5 * d_t2(p2,p3,h4,h5) * v2ijab(h4,h5,p3,p7)                       )
          ( iy1(p2,p7)               =  1.0 * iy1_1(p2,p7)                                                  )
          (   iy1_2_1(h3,p7)         =  1.0 * d_f1(h3,p7)                                                   )
          (   iy1_2_1(h3,p7)        +=  1.0 * d_t1(p4,h5) * v2ijab(h5,h3,p4,p7)                             )
          ( iy1(p2,p7)              += -1.0 * d_t1(p2,h3) * iy1_2_1(h3,p7)                                  )
          ( iy2(h8,h1)               =  1.0 * d_f1(h8,h1)                                                   )
          ( iy2(h8,h1)              +=  1.0 * d_t1(p9,h1) * iy1_2_1(h8,p9)                                  )
          ( iy2(h8,h1)              += -1.0 * d_t1(p5,h6) * v2ijka(h6,h8,h1,p5)                             )
          ( iy2(h8,h1)              += -0.5 * d_t2(p5,p6,h1,h7) * v2ijab(h7,h8,p5,p6)                       )
          (     iy3_1_1(h1,h7,cind)  =  1.0 * d_t1(p5,h1) * cholVpr(h7,p5,cind)                             )
          (   iy3_1(h7,p3,h1,p8)     =  1.0 * iy3_1_1(h1,h7,cind) * cholVpr(p3,p8,cind)                     )
          (     iy3_1_2(h1,p3,cind)  =  1.0 * d_t1(p5,h1) * cholVpr(p3,p5,cind)                             )
          (   iy3_1(h7,p3,h1,p8)    += -1.0 * iy3_1_2(h1,p3,cind) * cholVpr(h7,p8,cind)                     )
          ( iy3(h7,p3,h1,p8)         =  1.0 * iy3_1(h7,p3,h1,p8)                                            )
          ( iy3(h7,p3,h1,p8)        +=  1.0 * v2iajb(h7,p3,h1,p8)                                           )
          ( iy4_1(h8,h9,h1,p10)      =  1.0 * d_t1(p5,h1) * v2ijab(h8,h9,p5,p10)                            )
          ( iy4_2(h8,h9,h1,p10)      = -1.0 * iy4_1(h8,h9,h1,p10)                                           )
          ( iy4_1(h8,h9,h1,p10)     +=  1.0 * v2ijka(h8,h9,h1,p10)                                          )
          ( iy4_2(h8,h9,h1,p10)     +=  1.0 * v2ijka(h8,h9,h1,p10)                                          )
          ( iy6(h1,p3,cind)          =  1.0 * cholVpr(h6,p5,cind) * d_t2(p3,p5,h1,h6)                       )
          // EA spin explicit
          ( lt12_v_a(p1_va,p2_va)                 =  1.0 * lt12_v(p1_va,p2_va)               )
          ( lt12_v_b(p1_vb,p2_vb)                 =  1.0 * lt12_v(p1_vb,p2_vb)               )
          ( iy1_1_a(p2_va,p6_va)                  =  1.0 * iy1_1(p2_va,p6_va)                )
          ( iy1_1_b(p2_vb,p6_vb)                  =  1.0 * iy1_1(p2_vb,p6_vb)                )
          ( iy1_2_1_a(h3_oa,p7_va)                =  1.0 * iy1_2_1(h3_oa,p7_va)              )
          ( iy1_2_1_b(h3_ob,p7_vb)                =  1.0 * iy1_2_1(h3_ob,p7_vb)              )
          ( iy1_a(p2_va,p7_va)                    =  1.0 * iy1(p2_va,p7_va)                  )
          ( iy1_b(p2_vb,p7_vb)                    =  1.0 * iy1(p2_vb,p7_vb)                  )
          ( iy2_a(h8_oa,h1_oa)                    =  1.0 * iy2(h8_oa,h1_oa)                  )
          ( iy2_b(h8_ob,h1_ob)                    =  1.0 * iy2(h8_ob,h1_ob)                  )
          ( iy3_1_2_a(h1_oa,p3_va,cind)           =  1.0 * iy3_1_2(h1_oa,p3_va,cind)         )
          ( iy3_1_2_b(h1_ob,p3_vb,cind)           =  1.0 * iy3_1_2(h1_ob,p3_vb,cind)         )
          ( iy4_1_aaaa(h8_oa,h9_oa,h2_oa,p10_va)  =  1.0 * iy4_1(h8_oa,h9_oa,h2_oa,p10_va)   )
          ( iy4_1_baab(h8_ob,h9_oa,h2_oa,p10_vb)  =  1.0 * iy4_1(h8_ob,h9_oa,h2_oa,p10_vb)   )
          ( iy4_1_baba(h8_ob,h9_oa,h2_ob,p10_va)  =  1.0 * iy4_1(h8_ob,h9_oa,h2_ob,p10_va)   )
          ( iy4_1_bbbb(h8_ob,h9_ob,h2_ob,p10_vb)  =  1.0 * iy4_1(h8_ob,h9_ob,h2_ob,p10_vb)   )
          ( iy4_1_abba(h8_oa,h9_ob,h2_ob,p10_va)  =  1.0 * iy4_1(h8_oa,h9_ob,h2_ob,p10_va)   )
          ( iy4_1_abab(h8_oa,h9_ob,h2_oa,p10_vb)  =  1.0 * iy4_1(h8_oa,h9_ob,h2_oa,p10_vb)   )
          ( iy4_2_aaaa(h8_oa,h9_oa,h2_oa,p10_va)  =  1.0 * iy4_2(h8_oa,h9_oa,h2_oa,p10_va)   )
          ( iy4_2_baab(h8_ob,h9_oa,h2_oa,p10_vb)  =  1.0 * iy4_2(h8_ob,h9_oa,h2_oa,p10_vb)   )
          ( iy4_2_bbbb(h8_ob,h9_ob,h2_ob,p10_vb)  =  1.0 * iy4_2(h8_ob,h9_ob,h2_ob,p10_vb)   )
          ( iy4_2_abba(h8_oa,h9_ob,h2_ob,p10_va)  =  1.0 * iy4_2(h8_oa,h9_ob,h2_ob,p10_va)   )
          ( iy5_aaaa(h9_oa,p3_va,h1_oa,p8_va)     = -1.0 * d_t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * v2ijab_aaaa(h6_oa,h9_oa,p5_va,p8_va) )
          ( iy5_aaaa(h9_oa,p3_va,h1_oa,p8_va)    += -1.0 * d_t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * v2ijab_abab(h9_oa,h6_ob,p8_va,p5_vb) )
          ( iy5_bbbb(h9_ob,p3_vb,h1_ob,p8_vb)     = -1.0 * d_t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * v2ijab_bbbb(h6_ob,h9_ob,p5_vb,p8_vb) )
          ( iy5_bbbb(h9_ob,p3_vb,h1_ob,p8_vb)    += -1.0 * d_t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * v2ijab_abab(h6_oa,h9_ob,p5_va,p8_vb) )
          ( iy5_abab(h9_oa,p3_vb,h1_oa,p8_vb)     = -1.0 * d_t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * v2ijab_abab(h9_oa,h6_ob,p5_va,p8_vb) )
          ( iy5_baba(h9_ob,p3_va,h1_ob,p8_va)     = -1.0 * d_t2_abab(p3_va,p5_vb,h6_oa,h1_ob) * v2ijab_abab(h6_oa,h9_ob,p8_va,p5_vb) )
          ( iy5_baab(h9_ob,p3_va,h1_oa,p8_vb)     = -1.0 * d_t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * v2ijab_abab(h6_oa,h9_ob,p5_va,p8_vb) )
          ( iy5_baab(h9_ob,p3_va,h1_oa,p8_vb)    += -1.0 * d_t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * v2ijab_bbbb(h6_ob,h9_ob,p5_vb,p8_vb) )
          ( iy5_abba(h9_oa,p3_vb,h1_ob,p8_va)     = -1.0 * d_t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * v2ijab_abab(h9_oa,h6_ob,p8_va,p5_vb) )
          ( iy5_abba(h9_oa,p3_vb,h1_ob,p8_va)    += -1.0 * d_t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * v2ijab_aaaa(h6_oa,h9_oa,p5_va,p8_va) )
          ( iy3_1_aaaa(h9_oa,p3_va,h1_oa,p8_va)   =  1.0 * iy3_1(h9_oa,p3_va,h1_oa,p8_va)    )
          ( iy3_1_bbbb(h9_ob,p3_vb,h1_ob,p8_vb)   =  1.0 * iy3_1(h9_ob,p3_vb,h1_ob,p8_vb)    )
          ( iy3_1_abab(h9_oa,p3_vb,h1_oa,p8_vb)   =  1.0 * iy3_1(h9_oa,p3_vb,h1_oa,p8_vb)    )
          ( iy3_1_baba(h9_ob,p3_va,h1_ob,p8_va)   =  1.0 * iy3_1(h9_ob,p3_va,h1_ob,p8_va)    )
          ( iy3_1_baab(h9_ob,p3_va,h1_oa,p8_vb)   =  1.0 * iy3_1(h9_ob,p3_va,h1_oa,p8_vb)    )
          ( iy3_1_abba(h9_oa,p3_vb,h1_ob,p8_va)   =  1.0 * iy3_1(h9_oa,p3_vb,h1_ob,p8_va)    )
          ( iy3_aaaa(h9_oa,p3_va,h1_oa,p8_va)     =  1.0 * iy3(h9_oa,p3_va,h1_oa,p8_va)      )
          ( iy3_bbbb(h9_ob,p3_vb,h1_ob,p8_vb)     =  1.0 * iy3(h9_ob,p3_vb,h1_ob,p8_vb)      )
          ( iy3_abab(h9_oa,p3_vb,h1_oa,p8_vb)     =  1.0 * iy3(h9_oa,p3_vb,h1_oa,p8_vb)      )
          ( iy3_baba(h9_ob,p3_va,h1_ob,p8_va)     =  1.0 * iy3(h9_ob,p3_va,h1_ob,p8_va)      )
          ( iy3_baab(h9_ob,p3_va,h1_oa,p8_vb)     =  1.0 * iy3(h9_ob,p3_va,h1_oa,p8_vb)      )
          ( iy3_abba(h9_oa,p3_vb,h1_ob,p8_va)     =  1.0 * iy3(h9_oa,p3_vb,h1_ob,p8_va)      )
          ( iy3_1_aaaa(h9_oa,p3_va,h1_oa,p8_va)  +=  1.0 * iy5_aaaa(h9_oa,p3_va,h1_oa,p8_va) )
          ( iy3_1_bbbb(h9_ob,p3_vb,h1_ob,p8_vb)  +=  1.0 * iy5_bbbb(h9_ob,p3_vb,h1_ob,p8_vb) )
          ( iy3_1_abab(h9_oa,p3_vb,h1_oa,p8_vb)  +=  1.0 * iy5_abab(h9_oa,p3_vb,h1_oa,p8_vb) )
          ( iy3_1_baba(h9_ob,p3_va,h1_ob,p8_va)  +=  1.0 * iy5_baba(h9_ob,p3_va,h1_ob,p8_va) )
          ( iy3_1_baab(h9_ob,p3_va,h1_oa,p8_vb)  +=  1.0 * iy5_baab(h9_ob,p3_va,h1_oa,p8_vb) )
          ( iy3_1_abba(h9_oa,p3_vb,h1_ob,p8_va)  +=  1.0 * iy5_abba(h9_oa,p3_vb,h1_ob,p8_va) )
          ( iy6_a(h1_oa,p3_va,cind)               =  1.0 * iy6(h1_oa,p3_va,cind)             )
          ( iy6_b(h1_ob,p3_vb,cind)               =  1.0 * iy6(h1_ob,p3_vb,cind)             )
          .execute();
          #if GF_IN_SG
            }
          ec.pg().barrier();
          #endif

          #if GF_IN_SG
          if(subcomm != MPI_COMM_NULL) {
            sub_sch
          #else
            sch
          #endif            
          .deallocate(lt12_v,
                      iy1_1_1,iy1_1_2,iy1_1,iy1_2_1,iy1,
                      iy2,
                      iy3_1_1,iy3_1_2,iy3_1,iy3,
                      iy4_1,iy4_2,
                      iy6).execute();
      #if GF_IN_SG
        }
      ec.pg().barrier();
      #endif
      // clang-format on

      write_to_disk(t2v2_v, t2v2_v_file);
      write_to_disk(lt12_v_a, lt12_v_a_file);
      write_to_disk(lt12_v_b, lt12_v_b_file);
      write_to_disk(iy1_1_a, iy1_1_a_file);
      write_to_disk(iy1_1_b, iy1_1_b_file);
      write_to_disk(iy1_2_1_a, iy1_2_1_a_file);
      write_to_disk(iy1_2_1_b, iy1_2_1_b_file);
      write_to_disk(iy1_a, iy1_a_file);
      write_to_disk(iy1_b, iy1_b_file);
      write_to_disk(iy2_a, iy2_a_file);
      write_to_disk(iy2_b, iy2_b_file);
      write_to_disk(iy3_1_aaaa, iy3_1_aaaa_file);
      write_to_disk(iy3_1_bbbb, iy3_1_bbbb_file);
      write_to_disk(iy3_1_abab, iy3_1_abab_file);
      write_to_disk(iy3_1_baba, iy3_1_baba_file);
      write_to_disk(iy3_1_baab, iy3_1_baab_file);
      write_to_disk(iy3_1_abba, iy3_1_abba_file);
      write_to_disk(iy3_1_2_a, iy3_1_2_a_file);
      write_to_disk(iy3_1_2_b, iy3_1_2_b_file);
      write_to_disk(iy3_aaaa, iy3_aaaa_file);
      write_to_disk(iy3_bbbb, iy3_bbbb_file);
      write_to_disk(iy3_abab, iy3_abab_file);
      write_to_disk(iy3_baba, iy3_baba_file);
      write_to_disk(iy3_baab, iy3_baab_file);
      write_to_disk(iy3_abba, iy3_abba_file);
      write_to_disk(iy4_1_aaaa, iy4_1_aaaa_file);
      write_to_disk(iy4_1_bbbb, iy4_1_bbbb_file);
      write_to_disk(iy4_1_baba, iy4_1_baba_file);
      write_to_disk(iy4_1_baab, iy4_1_baab_file);
      write_to_disk(iy4_1_abba, iy4_1_abba_file);
      write_to_disk(iy4_1_abab, iy4_1_abab_file);
      write_to_disk(iy4_2_aaaa, iy4_2_aaaa_file);
      write_to_disk(iy4_2_bbbb, iy4_2_bbbb_file);
      write_to_disk(iy4_2_baab, iy4_2_baab_file);
      write_to_disk(iy4_2_abba, iy4_2_abba_file);
      write_to_disk(iy5_aaaa, iy5_aaaa_file);
      write_to_disk(iy5_bbbb, iy5_bbbb_file);
      write_to_disk(iy5_abab, iy5_abab_file);
      write_to_disk(iy5_baba, iy5_baba_file);
      write_to_disk(iy5_baab, iy5_baab_file);
      write_to_disk(iy5_abba, iy5_abba_file);
      write_to_disk(iy6_a, iy6_a_file);
      write_to_disk(iy6_b, iy6_b_file);
    }

    level               = 1;
    size_t prev_qr_rank = 0;

    ///////////////////////////////////////
    //                                   //
    //  performing advanced_alpha first  //
    //                                   //
    ///////////////////////////////////////
    if(rank == 0) {
      cout << endl << "_____advanced_GFCCSD_for_alpha_spin______" << endl;
      // ofs_profile << endl << "_____advanced_GFCCSD_for_alpha_spin______" << endl;
    }

    while(true) {
      const std::string levelstr     = std::to_string(level);
      std::string       q1_a_file    = files_prefix + ".a_q1_a.l" + levelstr;
      std::string       q2_aaa_file  = files_prefix + ".a_q2_aaa.l" + levelstr;
      std::string       q2_bab_file  = files_prefix + ".a_q2_bab.l" + levelstr;
      std::string       hx1_a_file   = files_prefix + ".a_hx1_a.l" + levelstr;
      std::string       hx2_aaa_file = files_prefix + ".a_hx2_aaa.l" + levelstr;
      std::string       hx2_bab_file = files_prefix + ".a_hx2_bab.l" + levelstr;
      std::string       hsub_a_file  = files_prefix + ".a_hsub_a.l" + levelstr;
      std::string       bsub_a_file  = files_prefix + ".a_bsub_a.l" + levelstr;
      std::string       cp_a_file    = files_prefix + ".a_cp_a.l" + levelstr;

      bool gf_restart = fs::exists(q1_a_file) && fs::exists(q2_aaa_file) &&
                        fs::exists(q2_bab_file) && fs::exists(hx1_a_file) &&
                        fs::exists(hx2_aaa_file) && fs::exists(hx2_bab_file) &&
                        fs::exists(hsub_a_file) && fs::exists(bsub_a_file) &&
                        fs::exists(cp_a_file) && ccsd_options.gf_restart;

      if(level == 1) {
        omega_extra.clear();
        omega_extra_finished.clear();
        omega_extra.push_back(omega_min_ea);
        omega_extra.push_back(omega_max_ea);
      }

      for(auto x: omega_extra) omega_extra_finished.push_back(x);

      auto qr_rank = omega_extra_finished.size() * nva;

      TiledIndexSpace otis;
      if(ndiis > qr_rank) {
        diis_tis = {IndexSpace{range(0, ndiis)}};
        otis     = {diis_tis, range(0, qr_rank)};
      }
      else {
        otis     = {IndexSpace{range(qr_rank)}};
        diis_tis = {otis, range(0, ndiis)};
      }

      TiledIndexSpace unit_tis{diis_tis, range(0, 1)};
      // auto [u1] = unit_tis.labels<1>("all");

      for(auto x: omega_extra) {
        ndiis    = ccsd_options.gf_ndiis;
        gf_omega = x;

        if(!gf_restart) {
          gfccsd_driver_ea_a<T>(
            ec, *sub_ec, subcomm, MO, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, d_f1, t2v2_v,
            lt12_v_a, lt12_v_b, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy1_a, iy1_b, iy2_a, iy2_b,
            iy3_1_aaaa, iy3_1_bbbb, iy3_1_abab, iy3_1_baba, iy3_1_baab, iy3_1_abba, iy3_1_2_a,
            iy3_1_2_b, iy3_aaaa, iy3_bbbb, iy3_abab, iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa,
            iy4_1_baab, iy4_1_baba, iy4_1_bbbb, iy4_1_abba, iy4_1_abab, iy4_2_aaaa, iy4_2_baab,
            iy4_2_bbbb, iy4_2_abba, iy5_aaaa, iy5_abab, iy5_baab, iy5_bbbb, iy5_baba, iy5_abba,
            iy6_a, iy6_b, v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, cholOO_a, cholOO_b, cholOV_a,
            cholOV_b, cholVV_a, cholVV_b, p_evl_sorted_occ, p_evl_sorted_virt, total_orbitals, nocc,
            nvir, nptsi, CI, unit_tis, files_prefix, levelstr, nva);
        }
      }

      ComplexTensor q1_tamm_a{v_alpha, otis};
      ComplexTensor q2_tamm_aaa{o_alpha, v_alpha, v_alpha, otis};
      ComplexTensor q2_tamm_bab{o_beta, v_alpha, v_beta, otis};
      ComplexTensor Hx1_tamm_a{v_alpha, otis};
      ComplexTensor Hx2_tamm_aaa{o_alpha, v_alpha, v_alpha, otis};
      ComplexTensor Hx2_tamm_bab{o_beta, v_alpha, v_beta, otis};

      if(!gf_restart) {
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        sch.allocate(q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, Hx1_tamm_a, Hx2_tamm_aaa, Hx2_tamm_bab)
          .execute();

        const std::string plevelstr = std::to_string(level - 1);

        std::string pq1_a_file   = files_prefix + ".a_q1_a.l" + plevelstr;
        std::string pq2_aaa_file = files_prefix + ".a_q2_aaa.l" + plevelstr;
        std::string pq2_bab_file = files_prefix + ".a_q2_bab.l" + plevelstr;

        if(rank == 0) {
          cout << "level = " << level << endl;
          // ofs_profile << "level = " << level << endl;
        }

        decltype(qr_rank) ivec_start = 0;
        bool              prev_q12   = fs::exists(pq1_a_file) && fs::exists(pq2_aaa_file) &&
                        fs::exists(pq2_bab_file);

        if(prev_q12) {
          TiledIndexSpace otis_prev{otis, range(0, prev_qr_rank)};
          auto [op1] = otis_prev.labels<1>("all");
          ComplexTensor q1_prev_a{v_alpha, otis_prev};
          ComplexTensor q2_prev_aaa{o_alpha, v_alpha, v_alpha, otis_prev};
          ComplexTensor q2_prev_bab{o_beta, v_alpha, v_beta, otis_prev};
          sch.allocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();

          read_from_disk(q1_prev_a, pq1_a_file);
          read_from_disk(q2_prev_aaa, pq2_aaa_file);
          read_from_disk(q2_prev_bab, pq2_bab_file);

          ivec_start = prev_qr_rank;

          if(subcomm != MPI_COMM_NULL) {
            // clang-format off
            sub_sch
              (q1_tamm_a(p1_va,op1) = q1_prev_a(p1_va,op1))
              (q2_tamm_aaa(h1_oa,p1_va,p2_va,op1) = q2_prev_aaa(h1_oa,p1_va,p2_va,op1))
              (q2_tamm_bab(h1_ob,p1_va,p2_vb,op1) = q2_prev_bab(h1_ob,p1_va,p2_vb,op1)).execute();
            // clang-format on
          }
          sch.deallocate(q1_prev_a, q2_prev_aaa, q2_prev_bab).execute();
        }

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          cout << endl
               << "Time to read in pre-computed Q1/Q2: " << std::fixed << std::setprecision(6)
               << time << " secs" << endl;

        ComplexTensor q1_tmp_a{v_alpha};
        ComplexTensor q2_tmp_aaa{o_alpha, v_alpha, v_alpha};
        ComplexTensor q2_tmp_bab{o_beta, v_alpha, v_beta};

        // TODO: optimize Q1/Q2 computation
        // Gram-Schmidt orthogonalization
        double time_gs_orth  = 0.0;
        double time_gs_norm  = 0.0;
        double total_time_gs = 0.0;

        bool q_exist = fs::exists(q1_a_file) && fs::exists(q2_aaa_file) && fs::exists(q2_bab_file);

        if(!q_exist) {
          sch.allocate(q1_tmp_a, q2_tmp_aaa, q2_tmp_bab).execute();

          if(rank == 0) {
            cout << "ivec_start,qr_rank = " << ivec_start << "," << qr_rank << endl;
            // ofs_profile << "ivec_start,qr_rank = " << ivec_start << "," << qr_rank << endl;
          }

          for(decltype(qr_rank) ivec = ivec_start; ivec < qr_rank; ivec++) {
            auto cc_t0 = std::chrono::high_resolution_clock::now();

            auto              W_read  = omega_extra_finished[ivec / (nva)];
            auto              pi_read = ivec % (nva);
            std::stringstream gfo;
            gfo << std::fixed << std::setprecision(2) << W_read;

            std::string y1_a_wpi_file =
              files_prefix + ".y1_a.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string y2_aaa_wpi_file =
              files_prefix + ".y2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string y2_bab_wpi_file =
              files_prefix + ".y2_bab.w" + gfo.str() + ".oi" + std::to_string(pi_read);

            if(fs::exists(y1_a_wpi_file) && fs::exists(y2_aaa_wpi_file) &&
               fs::exists(y2_bab_wpi_file)) {
              read_from_disk(q1_tmp_a, y1_a_wpi_file);
              read_from_disk(q2_tmp_aaa, y2_aaa_wpi_file);
              read_from_disk(q2_tmp_bab, y2_bab_wpi_file);
            }
            else {
              tamm_terminate("ERROR: At least one of " + y1_a_wpi_file + " and " + y2_aaa_wpi_file +
                             " and " + y2_bab_wpi_file + " do not exist!");
            }

            // TODO: schedule all iterations before executing
            if(ivec > 0) {
              TiledIndexSpace tsc{otis, range(0, ivec)};
              auto [sc] = tsc.labels<1>("all");

              ComplexTensor oscalar{tsc};
              ComplexTensor y1c_a{v_alpha, tsc};
              ComplexTensor y2c_aaa{o_alpha, v_alpha, v_alpha, tsc};
              ComplexTensor y2c_bab{o_beta, v_alpha, v_beta, tsc};

              // clang-format off
                #if GF_GS_SG
                if(subcomm != MPI_COMM_NULL){
                  sub_sch.allocate
                #else
                  sch.allocate
                #endif
                    (y1c_a,y2c_aaa,y2c_bab)                        
                    (y1c_a(p1_va,sc) = q1_tamm_a(p1_va,sc))
                    (y2c_aaa(h1_oa,p1_va,p2_va,sc) = q2_tamm_aaa(h1_oa,p1_va,p2_va,sc))
                    (y2c_bab(h1_ob,p1_va,p2_vb,sc) = q2_tamm_bab(h1_ob,p1_va,p2_vb,sc))
                    .execute();      
  
                    tamm::conj_ip(y1c_a);
                    tamm::conj_ip(y2c_aaa);
                    tamm::conj_ip(y2c_bab); 
  
                #if GF_GS_SG
                  sub_sch.allocate
                #else
                  sch.allocate
                #endif
                    (oscalar)
                    (oscalar(sc)  = -1.0 * q1_tmp_a(p1_va) * y1c_a(p1_va,sc))
                    (oscalar(sc) += -0.5 * q2_tmp_aaa(h1_oa,p1_va,p2_va) * y2c_aaa(h1_oa,p1_va,p2_va,sc))
                    (oscalar(sc) += -1.0 * q2_tmp_bab(h1_ob,p1_va,p2_vb) * y2c_bab(h1_ob,p1_va,p2_vb,sc))
  
                    (q1_tmp_a(p1_va) += oscalar(sc) * q1_tamm_a(p1_va,sc))
                    (q2_tmp_aaa(h1_oa,p1_va,p2_va) += oscalar(sc) * q2_tamm_aaa(h1_oa,p1_va,p2_va,sc))
                    (q2_tmp_bab(h1_ob,p1_va,p2_vb) += oscalar(sc) * q2_tamm_bab(h1_ob,p1_va,p2_vb,sc))
                    .deallocate(y1c_a,y2c_aaa,y2c_bab,oscalar).execute();
                #if GF_GS_SG
                }
                ec.pg().barrier();
                #endif
              // clang-format on
            }

            auto cc_t1 = std::chrono::high_resolution_clock::now();
            time_gs_orth +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_t1 - cc_t0)).count();

            auto q1norm_a   = norm(q1_tmp_a);
            auto q2norm_aaa = norm(q2_tmp_aaa);
            auto q2norm_bab = norm(q2_tmp_bab);

            // Normalization factor
            T newsc = 1.0 / std::real(sqrt(q1norm_a * q1norm_a + 0.5 * q2norm_aaa * q2norm_aaa +
                                           q2norm_bab * q2norm_bab));

            std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);

            TiledIndexSpace tsc{otis, range(ivec, ivec + 1)};
            auto [sc] = tsc.labels<1>("all");

            if(subcomm != MPI_COMM_NULL) {
              // clang-format off
              sub_sch
                (q1_tamm_a(p1_va,sc) = cnewsc * q1_tmp_a(p1_va))
                (q2_tamm_aaa(h2_oa,p1_va,p2_va,sc) = cnewsc * q2_tmp_aaa(h2_oa,p1_va,p2_va))
                (q2_tamm_bab(h2_ob,p1_va,p2_vb,sc) = cnewsc * q2_tmp_bab(h2_ob,p1_va,p2_vb))
                .execute();
              // clang-format on
            }
            ec.pg().barrier();

            auto cc_gs = std::chrono::high_resolution_clock::now();
            time_gs_norm +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
            total_time_gs +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t0)).count();
          } // end of Gram-Schmidt for loop over ivec

          sch.deallocate(q1_tmp_a, q2_tmp_aaa, q2_tmp_bab).execute();

          write_to_disk(q1_tamm_a, q1_a_file);
          write_to_disk(q2_tamm_aaa, q2_aaa_file);
          write_to_disk(q2_tamm_bab, q2_bab_file);
        }      // end of !gs-restart
        else { // restart GS
          read_from_disk(q1_tamm_a, q1_a_file);
          read_from_disk(q2_tamm_aaa, q2_aaa_file);
          read_from_disk(q2_tamm_bab, q2_bab_file);
        }

        if(rank == 0) {
          cout << endl
               << "Time for orthogonalization: " << std::fixed << std::setprecision(6)
               << time_gs_orth << " secs" << endl;
          cout << endl
               << "Time for normalizing and copying back: " << std::fixed << std::setprecision(6)
               << time_gs_norm << " secs" << endl;
          cout << endl
               << "Total time for Gram-Schmidt: " << std::fixed << std::setprecision(6)
               << total_time_gs << " secs" << endl;
        }
        auto cc_gs_x = std::chrono::high_resolution_clock::now();

        bool gs_x12_restart = fs::exists(hx1_a_file) && fs::exists(hx2_aaa_file) &&
                              fs::exists(hx2_bab_file);

        if(!gs_x12_restart) {
#if GF_IN_SG
          if(subcomm != MPI_COMM_NULL) {
            gfccsd_y1_a(sub_sch,
#else
          gfccsd_y1_a(sch,
#endif
                        MO, Hx1_tamm_a, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, q1_tamm_a,
                        q2_tamm_aaa, q2_tamm_bab, d_f1, iy1_a, iy1_2_1_a, iy1_2_1_b, v2ijab_aaaa,
                        v2ijab_abab, cholOV_a, cholOV_b, cholVV_a, CI, otis, true);

#if GF_IN_SG
            gfccsd_y2_a(sub_sch,
#else
          gfccsd_y2_a(sch,
#endif
                        MO, Hx2_tamm_aaa, Hx2_tamm_bab, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                        d_t2_abab, q1_tamm_a, q2_tamm_aaa, q2_tamm_bab, d_f1, iy1_1_a, iy1_1_b,
                        iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b, iy3_1_aaaa, iy3_1_baba, iy3_1_abba,
                        iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_bbbb, iy3_baba, iy3_baab, iy3_abba,
                        iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb, iy4_1_abba, iy4_2_aaaa,
                        iy4_2_abba, iy5_aaaa, iy5_baab, iy5_bbbb, iy5_baba, iy5_abba, iy6_a, iy6_b,
                        cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a, cholVV_b, v2ijab_aaaa,
                        v2ijab_abab, CI, otis, true);
#if GF_IN_SG
            sub_sch.execute();
          }
          ec.pg().barrier();
#else
          sch.execute();
#endif
          write_to_disk(Hx1_tamm_a, hx1_a_file);
          write_to_disk(Hx2_tamm_aaa, hx2_aaa_file);
          write_to_disk(Hx2_tamm_bab, hx2_bab_file);
        }
        else {
          read_from_disk(Hx1_tamm_a, hx1_a_file);
          read_from_disk(Hx2_tamm_aaa, hx2_aaa_file);
          read_from_disk(Hx2_tamm_bab, hx2_bab_file);
        }
        auto   cc_q12 = std::chrono::high_resolution_clock::now();
        double time_q12 =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
        if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;
      } // if !gf_restart

      prev_qr_rank = qr_rank;

      auto cc_t1 = std::chrono::high_resolution_clock::now();

      auto [otil, otil1, otil2] = otis.labels<3>("all");
      ComplexTensor hsub_tamm_a{otis, otis};
      ComplexTensor bsub_tamm_a{otis, v_alpha};
      ComplexTensor Cp_a{v_alpha, otis};

      sch.allocate(hsub_tamm_a, bsub_tamm_a, Cp_a).execute();

      if(!gf_restart) {
        ComplexTensor p1_k_a{o_alpha, otis};
        ComplexTensor q1_conj_a   = tamm::conj(q1_tamm_a);
        ComplexTensor q2_conj_aaa = tamm::conj(q2_tamm_aaa);
        ComplexTensor q2_conj_bab = tamm::conj(q2_tamm_bab);

        // clang-format off
        sch
          (bsub_tamm_a(otil1,p1_va)  =       q1_conj_a(p1_va,otil1))
          (hsub_tamm_a(otil1,otil2)  =       q1_conj_a(p1_va,otil1) * Hx1_tamm_a(p1_va,otil2))
          (hsub_tamm_a(otil1,otil2) += 0.5 * q2_conj_aaa(h1_oa,p1_va,p2_va,otil1) * Hx2_tamm_aaa(h1_oa,p1_va,p2_va,otil2))
          (hsub_tamm_a(otil1,otil2) +=       q2_conj_bab(h1_ob,p1_va,p2_vb,otil1) * Hx2_tamm_bab(h1_ob,p1_va,p2_vb,otil2))
          .deallocate(q1_conj_a,q2_conj_aaa,q2_conj_bab)
          
          .allocate(p1_k_a)
          ( Cp_a(p1_va,otil)    =        q1_tamm_a(p1_va,otil)                                     )
          ( Cp_a(p1_va,otil)   +=  1.0 * lt12_v_a(p1_va,p2_va) * q1_tamm_a(p2_va,otil)             )
          ( Cp_a(p2_va,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aaa(h1_oa,p2_va,p1_va,otil) )
          ( Cp_a(p2_va,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bab(h1_ob,p2_va,p1_vb,otil) )
          ( p1_k_a(h1_oa,otil)  =        d_t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) * q2_tamm_aaa(h2_oa,p1_va,p2_va,otil) )
          ( p1_k_a(h1_oa,otil) +=  2.0 * d_t2_abab(p1_va,p2_vb,h1_oa,h2_ob) * q2_tamm_bab(h2_ob,p1_va,p2_vb,otil) )
          ( Cp_a(p1_va,otil)   += -0.5 * p1_k_a(h1_oa,otil) * d_t1_a(p1_va,h1_oa) )
          .deallocate(p1_k_a,
                    q1_tamm_a, q2_tamm_aaa, q2_tamm_bab,
                  Hx1_tamm_a,Hx2_tamm_aaa,Hx2_tamm_bab)
          .execute();
        // clang-format on
      } // if !gf_restart

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

      if(!gf_restart) {
        write_to_disk(hsub_tamm_a, hsub_a_file);
        write_to_disk(bsub_tamm_a, bsub_a_file);
        write_to_disk(Cp_a, cp_a_file);
      }
      else {
        read_from_disk(hsub_tamm_a, hsub_a_file);
        read_from_disk(bsub_tamm_a, bsub_a_file);
        read_from_disk(Cp_a, cp_a_file);
      }

      Complex2DMatrix hsub_a(qr_rank, qr_rank);
      Complex2DMatrix bsub_a(qr_rank, nva);

      tamm_to_eigen_tensor(hsub_tamm_a, hsub_a);
      tamm_to_eigen_tensor(bsub_tamm_a, bsub_a);
      Complex2DMatrix hident = Complex2DMatrix::Identity(hsub_a.rows(), hsub_a.cols());

      ComplexTensor Cp_local_a{v_alpha, otis};

      Scheduler sch_l{ec_l};
      sch_l.allocate(Cp_local_a)(Cp_local_a() = Cp_a()).execute();

      if(rank == 0) {
        cout << endl << "spectral function (omega_npts_ea = " << omega_npts_ea << "):" << endl;
      }

      cc_t1 = std::chrono::high_resolution_clock::now();
      std::vector<double> ni_w(omega_npts_ea, 0);
      std::vector<double> ni_A(omega_npts_ea, 0);

      // Compute spectral function for designated omega regime
      for(int64_t ni = 0; ni < omega_npts_ea; ni++) {
        std::complex<T> omega_tmp = std::complex<T>(omega_min_ea + ni * omega_delta, gf_eta);

        Complex2DMatrix ysub_a = (-1.0 * hsub_a + omega_tmp * hident).lu().solve(bsub_a);

        ComplexTensor ysub_local_a{otis, v_alpha};
        ComplexTensor o_local_a{v_alpha};
        sch_l.allocate(ysub_local_a, o_local_a).execute();

        eigen_to_tamm_tensor(ysub_local_a, ysub_a);

        sch_l(o_local_a(p1_va) = Cp_local_a(p1_va, otil) * ysub_local_a(otil, p1_va)).execute();

        auto oscalar = std::imag(tamm::sum(o_local_a));

        sch_l.deallocate(ysub_local_a, o_local_a).execute();

        if(level == 1) { omega_ea_A0[ni] = oscalar; }
        else {
          if(level > 1) {
            T oerr          = oscalar - omega_ea_A0[ni];
            omega_ea_A0[ni] = oscalar;
            if(std::abs(oerr) < gf_threshold) omega_ea_conv_a[ni] = true;
          }
        }
        if(rank == 0) {
          std::ostringstream spf;
          spf << "W = " << std::fixed << std::setprecision(2) << std::real(omega_tmp)
              << ", omega_ea_A0 =  " << std::fixed << std::setprecision(4) << omega_ea_A0[ni]
              << endl;
          cout << spf.str();
          ni_A[ni] = omega_ea_A0[ni];
          ni_w[ni] = std::real(omega_tmp);
        }
      }

      cc_t2 = std::chrono::high_resolution_clock::now();
      time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) {
        cout << endl << "omegas processed in level " << level << " = " << omega_extra << endl;
        cout << "Time to compute spectral function in level " << level
             << " (omega_npts_ea = " << omega_npts_ea << "): " << time << " secs" << endl;
        write_results_to_json(ec, sys_data, level, ni_w, ni_A, "advanced_alpha");
      }

      auto               extrap_file = files_prefix + ".extrapolate.advanced.alpha.txt";
      std::ostringstream spfe;
      spfe << "";

      // extrapolate or proceed to next level
      if(std::all_of(omega_ea_conv_a.begin(), omega_ea_conv_a.end(), [](bool x) { return x; }) ||
         gf_extrapolate_level == level) {
        if(rank == 0)
          cout << endl
               << "--------------------extrapolate & converge-----------------------" << endl;
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t taskcount = 0;
        int64_t next      = ac->fetch_add(0, 1);

        ComplexTensor ysub_local_a{otis, v_alpha};
        ComplexTensor o_local_a{v_alpha};
        sch_l.allocate(ysub_local_a, o_local_a).execute();

        for(int64_t ni = 0; ni < lomega_npts_ea; ni++) {
          if(next == taskcount) {
            std::complex<T> omega_tmp = std::complex<T>(lomega_min_ea + ni * omega_delta_e, gf_eta);

            Complex2DMatrix ysub_a = (-1.0 * hsub_a + omega_tmp * hident).lu().solve(bsub_a);
            eigen_to_tamm_tensor(ysub_local_a, ysub_a);

            sch_l(o_local_a(p1_va) = Cp_local_a(p1_va, otil) * ysub_local_a(otil, p1_va)).execute();

            auto oscalar = std::imag(tamm::sum(o_local_a));

            Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(nva);
            tamm_to_eigen_tensor(o_local_a, olocala_eig);
            for(TAMM_SIZE nj = 0; nj < nva; nj++) {
              auto gpp = olocala_eig(nj).imag();
              spfe << "orb_index = " << nj << ", gpp_a = " << gpp << endl;
            }

            spfe << "w = " << std::fixed << std::setprecision(3) << std::real(omega_tmp)
                 << ", A_a =  " << std::fixed << std::setprecision(6) << oscalar << endl;

            next = ac->fetch_add(0, 1);
          }
          taskcount++;
        }

        sch_l.deallocate(ysub_local_a, o_local_a).execute();

        ec.pg().barrier();
        ac->deallocate();
        delete ac;

        write_string_to_disk(ec, spfe.str(), extrap_file);
        if(rank == 0) {
          sys_data.results["output"]["GFCCSD"]["advanced_alpha"]["nlevels"] = level;
          write_json_data(sys_data, "GFCCSD");
        }

        sch_l.deallocate(Cp_local_a).execute();
        sch.deallocate(hsub_tamm_a, bsub_tamm_a, Cp_a).execute();

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          std::cout << endl
                    << "Time taken for extrapolation (lomega_npts_ea = " << lomega_npts_ea
                    << "): " << time << " secs" << endl;

        break;
      }
      else {
        if(level == 1) {
          auto o1 = (omega_extra[0] + omega_extra[1]) / 2;
          omega_extra.clear();
          o1 = find_closest(o1, omega_space_ea);
          omega_extra.push_back(o1);
        }
        else {
          std::sort(omega_extra_finished.begin(), omega_extra_finished.end());
          omega_extra.clear();
          std::vector<T> wtemp;
          for(size_t i = 1; i < omega_extra_finished.size(); i++) {
            bool   oe_add = false;
            auto   w1     = omega_extra_finished[i - 1];
            auto   w2     = omega_extra_finished[i];
            size_t num_w  = (w2 - w1) / omega_delta + 1;
            for(size_t j = 0; j < num_w; j++) {
              T      otmp = w1 + j * omega_delta;
              size_t ind  = (otmp - omega_min_ea) / omega_delta;
              if(!omega_ea_conv_a[ind]) {
                oe_add = true;
                break;
              }
            }

            if(oe_add) {
              T Win = (w1 + w2) / 2;
              Win   = find_closest(Win, omega_space_ea);
              if(std::find(omega_extra_finished.begin(), omega_extra_finished.end(), Win) !=
                 omega_extra_finished.end()) {
                continue;
              }
              else { omega_extra.push_back(Win); }
            } // end oe add
          }   // end oe finished
        }
        if(rank == 0) {
          cout << "new freq's:" << std::fixed << std::setprecision(2) << omega_extra << endl;
        }
        level++;
      }

      sch_l.deallocate(Cp_local_a).execute();
      sch.deallocate(hsub_tamm_a, bsub_tamm_a, Cp_a).execute();

    } // end while
    // end of alpha

    level        = 1;
    prev_qr_rank = 0;

    ////////////////////////////////////////////////
    //                                            //
    // if open_shell, then execute advanced_beta  //
    //                                            //
    ////////////////////////////////////////////////
    if(rank == 0 && ccsd_options.gf_os) {
      cout << endl << "_____advanced_GFCCSD_on_beta_spin______" << endl;
      // ofs_profile << endl << "_____advanced_GFCCSD_on_beta_spin______" << endl;
    }

    while(ccsd_options.gf_os) {
      const std::string levelstr     = std::to_string(level);
      std::string       q1_b_file    = files_prefix + ".a_q1_b.l" + levelstr;
      std::string       q2_bbb_file  = files_prefix + ".a_q2_bbb.l" + levelstr;
      std::string       q2_aba_file  = files_prefix + ".a_q2_aba.l" + levelstr;
      std::string       hx1_b_file   = files_prefix + ".a_hx1_b.l" + levelstr;
      std::string       hx2_bbb_file = files_prefix + ".a_hx2_bbb.l" + levelstr;
      std::string       hx2_aba_file = files_prefix + ".a_hx2_aba.l" + levelstr;
      std::string       hsub_b_file  = files_prefix + ".a_hsub_b.l" + levelstr;
      std::string       bsub_b_file  = files_prefix + ".a_bsub_b.l" + levelstr;
      std::string       cp_b_file    = files_prefix + ".a_cp_b.l" + levelstr;

      bool gf_restart = fs::exists(q1_b_file) && fs::exists(q2_bbb_file) &&
                        fs::exists(q2_aba_file) && fs::exists(hx1_b_file) &&
                        fs::exists(hx2_bbb_file) && fs::exists(hx2_aba_file) &&
                        fs::exists(hsub_b_file) && fs::exists(bsub_b_file) &&
                        fs::exists(cp_b_file) && ccsd_options.gf_restart;

      if(level == 1) {
        omega_extra.clear();
        omega_extra_finished.clear();
        omega_extra.push_back(omega_min_ea);
        omega_extra.push_back(omega_max_ea);
      }

      for(auto x: omega_extra) omega_extra_finished.push_back(x);

      auto qr_rank = omega_extra_finished.size() * nva;

      TiledIndexSpace otis;
      if(ndiis > qr_rank) {
        diis_tis = {IndexSpace{range(0, ndiis)}};
        otis     = {diis_tis, range(0, qr_rank)};
      }
      else {
        otis     = {IndexSpace{range(qr_rank)}};
        diis_tis = {otis, range(0, ndiis)};
      }

      TiledIndexSpace unit_tis{diis_tis, range(0, 1)};
      // auto [u1] = unit_tis.labels<1>("all");

      for(auto x: omega_extra) {
        ndiis    = ccsd_options.gf_ndiis;
        gf_omega = x;

        if(!gf_restart) {
          gfccsd_driver_ea_b<T>(
            ec, *sub_ec, subcomm, MO, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, d_f1, t2v2_v,
            lt12_v_a, lt12_v_b, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy1_a, iy1_b, iy2_a, iy2_b,
            iy3_1_aaaa, iy3_1_bbbb, iy3_1_abab, iy3_1_baba, iy3_1_baab, iy3_1_abba, iy3_1_2_a,
            iy3_1_2_b, iy3_aaaa, iy3_bbbb, iy3_abab, iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa,
            iy4_1_baab, iy4_1_baba, iy4_1_bbbb, iy4_1_abba, iy4_1_abab, iy4_2_aaaa, iy4_2_baab,
            iy4_2_bbbb, iy4_2_abba, iy5_aaaa, iy5_abab, iy5_baab, iy5_bbbb, iy5_baba, iy5_abba,
            iy6_a, iy6_b, v2ijab_aaaa, v2ijab_abab, v2ijab_bbbb, cholOO_a, cholOO_b, cholOV_a,
            cholOV_b, cholVV_a, cholVV_b, p_evl_sorted_occ, p_evl_sorted_virt, total_orbitals, nocc,
            nvir, nptsi, CI, unit_tis, files_prefix, levelstr, nva, nvb);
        }
      }

      ComplexTensor q1_tamm_b{v_beta, otis};
      ComplexTensor q2_tamm_bbb{o_beta, v_beta, v_beta, otis};
      ComplexTensor q2_tamm_aba{o_alpha, v_beta, v_alpha, otis};
      ComplexTensor Hx1_tamm_b{v_beta, otis};
      ComplexTensor Hx2_tamm_bbb{o_beta, v_beta, v_beta, otis};
      ComplexTensor Hx2_tamm_aba{o_alpha, v_beta, v_alpha, otis};

      if(!gf_restart) {
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        sch.allocate(q1_tamm_b, q2_tamm_bbb, q2_tamm_aba, Hx1_tamm_b, Hx2_tamm_bbb, Hx2_tamm_aba)
          .execute();

        const std::string plevelstr = std::to_string(level - 1);

        std::string pq1_b_file   = files_prefix + ".a_q1_b.l" + plevelstr;
        std::string pq2_bbb_file = files_prefix + ".a_q2_bbb.l" + plevelstr;
        std::string pq2_aba_file = files_prefix + ".a_q2_aba.l" + plevelstr;

        decltype(qr_rank) ivec_start = 0;
        bool              prev_q12   = fs::exists(pq1_b_file) && fs::exists(pq2_bbb_file) &&
                        fs::exists(pq2_aba_file);

        if(prev_q12) {
          TiledIndexSpace otis_prev{otis, range(0, prev_qr_rank)};
          auto [op1] = otis_prev.labels<1>("all");
          ComplexTensor q1_prev_b{v_beta, otis_prev};
          ComplexTensor q2_prev_bbb{o_beta, v_beta, v_beta, otis_prev};
          ComplexTensor q2_prev_aba{o_alpha, v_beta, v_alpha, otis_prev};
          sch.allocate(q1_prev_b, q2_prev_bbb, q2_prev_aba).execute();

          read_from_disk(q1_prev_b, pq1_b_file);
          read_from_disk(q2_prev_bbb, pq2_bbb_file);
          read_from_disk(q2_prev_aba, pq2_aba_file);

          ivec_start = prev_qr_rank;

          if(subcomm != MPI_COMM_NULL) {
            // clang-format off
            sub_sch
              (q1_tamm_b(p1_vb,op1) = q1_prev_b(p1_vb,op1))
              (q2_tamm_bbb(h1_ob,p1_vb,p2_vb,op1) = q2_prev_bbb(h1_ob,p1_vb,p2_vb,op1))
              (q2_tamm_aba(h1_oa,p1_vb,p2_va,op1) = q2_prev_aba(h1_oa,p1_vb,p2_va,op1)).execute();
            // clang-format on
          }
          sch.deallocate(q1_prev_b, q2_prev_bbb, q2_prev_aba).execute();
        }

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          cout << endl
               << "Time to read in pre-computed Q1/Q2: " << std::fixed << std::setprecision(6)
               << time << " secs" << endl;

        ComplexTensor q1_tmp_b{v_beta};
        ComplexTensor q2_tmp_bbb{o_beta, v_beta, v_beta};
        ComplexTensor q2_tmp_aba{o_alpha, v_beta, v_alpha};

        // TODO: optimize Q1/Q2 computation
        // Gram-Schmidt orthogonalization
        double time_gs_orth  = 0.0;
        double time_gs_norm  = 0.0;
        double total_time_gs = 0.0;

        bool q_exist = fs::exists(q1_b_file) && fs::exists(q2_bbb_file) && fs::exists(q2_aba_file);

        if(!q_exist) {
          sch.allocate(q1_tmp_b, q2_tmp_bbb, q2_tmp_aba).execute();

          for(decltype(qr_rank) ivec = ivec_start; ivec < qr_rank; ivec++) {
            auto cc_t0 = std::chrono::high_resolution_clock::now();

            auto              W_read  = omega_extra_finished[ivec / (nva)];
            auto              pi_read = ivec % (nva) + nva;
            std::stringstream gfo;
            gfo << std::fixed << std::setprecision(2) << W_read;

            std::string y1_b_wpi_file =
              files_prefix + ".y1_b.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string y2_bbb_wpi_file =
              files_prefix + ".y2_bbb.w" + gfo.str() + ".oi" + std::to_string(pi_read);
            std::string y2_aba_wpi_file =
              files_prefix + ".y2_aba.w" + gfo.str() + ".oi" + std::to_string(pi_read);

            if(fs::exists(y1_b_wpi_file) && fs::exists(y2_bbb_wpi_file) &&
               fs::exists(y2_aba_wpi_file)) {
              read_from_disk(q1_tmp_b, y1_b_wpi_file);
              read_from_disk(q2_tmp_bbb, y2_bbb_wpi_file);
              read_from_disk(q2_tmp_aba, y2_aba_wpi_file);
            }
            else {
              tamm_terminate("ERROR: At least one of " + y1_b_wpi_file + " and " + y2_bbb_wpi_file +
                             " and " + y2_aba_wpi_file + " do not exist!");
            }

            // TODO: schedule all iterations before executing
            if(ivec > 0) {
              TiledIndexSpace tsc{otis, range(0, ivec)};
              auto [sc] = tsc.labels<1>("all");

              ComplexTensor oscalar{tsc};
              ComplexTensor y1c_b{v_beta, tsc};
              ComplexTensor y2c_bbb{o_beta, v_beta, v_beta, tsc};
              ComplexTensor y2c_aba{o_alpha, v_beta, v_alpha, tsc};

              // clang-format off
              
              #if GF_GS_SG
              if(subcomm != MPI_COMM_NULL){
                sub_sch.allocate
              #else
                sch.allocate
              #endif
                  (y1c_b,y2c_bbb,y2c_aba)
                  (y1c_b(p1_vb,sc) = q1_tamm_b(p1_vb,sc))
                  (y2c_bbb(h1_ob,p1_vb,p2_vb,sc) = q2_tamm_bbb(h1_ob,p1_vb,p2_vb,sc))
                  (y2c_aba(h1_oa,p1_vb,p2_va,sc) = q2_tamm_aba(h1_oa,p1_vb,p2_va,sc))
                  .execute();
                // clang-format on

                tamm::conj_ip(y1c_b);
                tamm::conj_ip(y2c_bbb);
                tamm::conj_ip(y2c_aba);

                // clang-format off

                #if GF_GS_SG
                  sub_sch.allocate
                #else
                  sch.allocate
                #endif
                    (oscalar)
                    (oscalar(sc)  = -1.0 * q1_tmp_b(p1_vb) * y1c_b(p1_vb,sc))
                    (oscalar(sc) += -0.5 * q2_tmp_bbb(h1_ob,p1_vb,p2_vb) * y2c_bbb(h1_ob,p1_vb,p2_vb,sc))
                    (oscalar(sc) += -1.0 * q2_tmp_aba(h1_oa,p1_vb,p2_va) * y2c_aba(h1_oa,p1_vb,p2_va,sc))

                    (q1_tmp_b(p1_vb) += oscalar(sc) * q1_tamm_b(p1_vb,sc))
                    (q2_tmp_bbb(h1_ob,p1_vb,p2_vb) += oscalar(sc) * q2_tamm_bbb(h1_ob,p1_vb,p2_vb,sc))
                    (q2_tmp_aba(h1_oa,p1_vb,p2_va) += oscalar(sc) * q2_tamm_aba(h1_oa,p1_vb,p2_va,sc))
                    .deallocate(y1c_b,y2c_bbb,y2c_aba,oscalar).execute();
                #if GF_GS_SG
                }
                ec.pg().barrier();
                #endif
              // clang-format on
            }

            auto cc_t1 = std::chrono::high_resolution_clock::now();
            time_gs_orth +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_t1 - cc_t0)).count();

            auto q1norm_b   = norm(q1_tmp_b);
            auto q2norm_bbb = norm(q2_tmp_bbb);
            auto q2norm_aba = norm(q2_tmp_aba);

            // Normalization factor
            T newsc = 1.0 / std::real(sqrt(q1norm_b * q1norm_b + 0.5 * q2norm_bbb * q2norm_bbb +
                                           q2norm_aba * q2norm_aba));

            std::complex<T> cnewsc = static_cast<std::complex<T>>(newsc);

            TiledIndexSpace tsc{otis, range(ivec, ivec + 1)};
            auto [sc] = tsc.labels<1>("all");

            if(subcomm != MPI_COMM_NULL) {
              // clang-format off
              sub_sch
              (q1_tamm_b(p1_vb,sc) = cnewsc * q1_tmp_b(p1_vb))
              (q2_tamm_bbb(h2_ob,p1_vb,p2_vb,sc) = cnewsc * q2_tmp_bbb(h2_ob,p1_vb,p2_vb))
              (q2_tamm_aba(h2_oa,p1_vb,p2_va,sc) = cnewsc * q2_tmp_aba(h2_oa,p1_vb,p2_va))
              .execute();
              // clang-format on
            }
            ec.pg().barrier();

            auto cc_gs = std::chrono::high_resolution_clock::now();
            time_gs_norm +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t1)).count();
            total_time_gs +=
              std::chrono::duration_cast<std::chrono::duration<double>>((cc_gs - cc_t0)).count();
          } // end of Gram-Schmidt for loop over ivec

          sch.deallocate(q1_tmp_b, q2_tmp_bbb, q2_tmp_aba).execute();

          write_to_disk(q1_tamm_b, q1_b_file);
          write_to_disk(q2_tamm_bbb, q2_bbb_file);
          write_to_disk(q2_tamm_aba, q2_aba_file);
        }      // end of !gs-restart
        else { // restart GS
          read_from_disk(q1_tamm_b, q1_b_file);
          read_from_disk(q2_tamm_bbb, q2_bbb_file);
          read_from_disk(q2_tamm_aba, q2_aba_file);
        }

        if(rank == 0) {
          cout << endl
               << "Time for orthogonalization: " << std::fixed << std::setprecision(6)
               << time_gs_orth << " secs" << endl;
          cout << endl
               << "Time for normalizing and copying back: " << std::fixed << std::setprecision(6)
               << time_gs_norm << " secs" << endl;
          cout << endl
               << "Total time for Gram-Schmidt: " << std::fixed << std::setprecision(6)
               << total_time_gs << " secs" << endl;
        }
        auto cc_gs_x = std::chrono::high_resolution_clock::now();

        bool gs_x12_restart = fs::exists(hx1_b_file) && fs::exists(hx2_bbb_file) &&
                              fs::exists(hx2_aba_file);

        if(!gs_x12_restart) {
#if GF_IN_SG
          if(subcomm != MPI_COMM_NULL) {
            gfccsd_y1_b(sub_sch,
#else
          gfccsd_y1_b(sch,
#endif
                        MO, Hx1_tamm_b, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, q1_tamm_b,
                        q2_tamm_bbb, q2_tamm_aba, d_f1, iy1_b, iy1_2_1_a, iy1_2_1_b, v2ijab_bbbb,
                        v2ijab_abab, cholOV_a, cholOV_b, cholVV_b, CI, otis, true);

#if GF_IN_SG
            gfccsd_y2_b(sub_sch,
#else
          gfccsd_y2_b(sch,
#endif
                        MO, Hx2_tamm_bbb, Hx2_tamm_aba, d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb,
                        d_t2_abab, q1_tamm_b, q2_tamm_bbb, q2_tamm_aba, d_f1, iy1_1_a, iy1_1_b,
                        iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b, iy3_1_bbbb, iy3_1_abab, iy3_1_baab,
                        iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_bbbb, iy3_abab, iy3_baab, iy3_abba,
                        iy4_1_aaaa, iy4_1_baab, iy4_1_bbbb, iy4_1_abba, iy4_1_abab, iy4_2_bbbb,
                        iy4_2_baab, iy5_aaaa, iy5_baab, iy5_bbbb, iy5_abab, iy5_abba, iy6_a, iy6_b,
                        cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a, cholVV_b, v2ijab_bbbb,
                        v2ijab_abab, CI, otis, true);

#if GF_IN_SG
            sub_sch.execute();
          }
          ec.pg().barrier();
#else
          sch.execute();
#endif
          write_to_disk(Hx1_tamm_b, hx1_b_file);
          write_to_disk(Hx2_tamm_bbb, hx2_bbb_file);
          write_to_disk(Hx2_tamm_aba, hx2_aba_file);
        }
        else {
          read_from_disk(Hx1_tamm_b, hx1_b_file);
          read_from_disk(Hx2_tamm_bbb, hx2_bbb_file);
          read_from_disk(Hx2_tamm_aba, hx2_aba_file);
        }
        auto   cc_q12 = std::chrono::high_resolution_clock::now();
        double time_q12 =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_q12 - cc_gs_x)).count();
        if(rank == 0) cout << endl << "Time to contract Q1/Q2: " << time_q12 << " secs" << endl;
      } // if !gf_restart

      prev_qr_rank = qr_rank;

      auto cc_t1 = std::chrono::high_resolution_clock::now();

      auto [otil, otil1, otil2] = otis.labels<3>("all");
      ComplexTensor hsub_tamm_b{otis, otis};
      ComplexTensor bsub_tamm_b{otis, v_alpha};
      ComplexTensor Cp_b{v_beta, otis};
      ComplexTensor::allocate(&ec, hsub_tamm_b, bsub_tamm_b, Cp_b);

      if(!gf_restart) {
        ComplexTensor p1_k_b{o_beta, otis};
        ComplexTensor q1_conj_b   = tamm::conj(q1_tamm_b);
        ComplexTensor q2_conj_bbb = tamm::conj(q2_tamm_bbb);
        ComplexTensor q2_conj_aba = tamm::conj(q2_tamm_aba);

        // clang-format off
        sch
          (bsub_tamm_b(otil1,p1_vb)  =       q1_conj_b(p1_vb,otil1))
          (hsub_tamm_b(otil1,otil2)  =       q1_conj_b(p1_vb,otil1) * Hx1_tamm_b(p1_vb,otil2))
          (hsub_tamm_b(otil1,otil2) += 0.5 * q2_conj_bbb(h1_ob,p1_vb,p2_vb,otil1) * Hx2_tamm_bbb(h1_ob,p1_vb,p2_vb,otil2))
          (hsub_tamm_b(otil1,otil2) +=       q2_conj_aba(h1_oa,p1_vb,p2_va,otil1) * Hx2_tamm_aba(h1_oa,p1_vb,p2_va,otil2))
          .deallocate(q1_conj_b,q2_conj_bbb,q2_conj_aba)
          
          .allocate(p1_k_b)
          ( Cp_b(p1_vb,otil)    =        q1_tamm_b(p1_vb,otil)                                     )
          ( Cp_b(p2_vb,otil)   += -1.0 * lt12_v_b(p1_vb,p2_vb) * q1_tamm_b(p1_vb,otil)             )
          ( Cp_b(p2_vb,otil)   +=        d_t1_b(p1_vb,h1_ob) * q2_tamm_bbb(h1_ob,p2_vb,p1_vb,otil) )
          ( Cp_b(p2_vb,otil)   +=        d_t1_a(p1_va,h1_oa) * q2_tamm_aba(h1_oa,p2_vb,p1_va,otil) )
          ( p1_k_b(h1_ob,otil)  =        d_t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob) * q2_tamm_bbb(h2_ob,p1_vb,p2_vb,otil) )
          ( p1_k_b(h1_ob,otil) +=  2.0 * d_t2_abab(p2_va,p1_vb,h2_oa,h1_ob) * q2_tamm_aba(h2_oa,p1_vb,p2_va,otil) )
          ( Cp_b(p1_vb,otil)   += -0.5 * p1_k_b(h1_ob,otil) * d_t1_b(p1_vb,h1_ob) )
          .deallocate(p1_k_b,q1_tamm_b, q2_tamm_bbb, q2_tamm_aba)
          .execute();
        // clang-format on
      } // if !gf_restart

      auto cc_t2 = std::chrono::high_resolution_clock::now();
      auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

      if(rank == 0) cout << endl << "Time to compute Cp: " << time << " secs" << endl;

      // Write all tensors
      if(!gf_restart) {
        write_to_disk(hsub_tamm_b, hsub_b_file);
        write_to_disk(bsub_tamm_b, bsub_b_file);
        write_to_disk(Cp_b, cp_b_file);
        sch.deallocate(Hx1_tamm_b, Hx2_tamm_bbb, Hx2_tamm_aba).execute();
      }
      else {
        read_from_disk(hsub_tamm_b, hsub_b_file);
        read_from_disk(bsub_tamm_b, bsub_b_file);
        read_from_disk(Cp_b, cp_b_file);
      }

      Complex2DMatrix hsub_b(qr_rank, qr_rank);
      Complex2DMatrix bsub_b(qr_rank, nvb);

      tamm_to_eigen_tensor(hsub_tamm_b, hsub_b);
      tamm_to_eigen_tensor(bsub_tamm_b, bsub_b);
      Complex2DMatrix hident = Complex2DMatrix::Identity(hsub_b.rows(), hsub_b.cols());

      ComplexTensor ysub_local_b{otis, v_beta};
      ComplexTensor o_local_b{v_beta};
      ComplexTensor Cp_local_b{v_beta, otis};

      ComplexTensor::allocate(&ec_l, ysub_local_b, o_local_b, Cp_local_b);

      Scheduler sch_l{ec_l};
      sch_l(Cp_local_b(p1_vb, otil) = Cp_b(p1_vb, otil)).execute();

      if(rank == 0) {
        cout << endl << "spectral function (omega_npts_ea = " << omega_npts_ea << "):" << endl;
      }

      cc_t1 = std::chrono::high_resolution_clock::now();
      std::vector<double> ni_w(omega_npts_ea, 0);
      std::vector<double> ni_A(omega_npts_ea, 0);

      // Compute spectral function for designated omega regime
      for(int64_t ni = 0; ni < omega_npts_ea; ni++) {
        std::complex<T> omega_tmp = std::complex<T>(omega_min_ea + ni * omega_delta, gf_eta);

        Complex2DMatrix ysub_b = (hsub_b + omega_tmp * hident).lu().solve(bsub_b);
        eigen_to_tamm_tensor(ysub_local_b, ysub_b);

        sch_l(o_local_b(p1_vb) = Cp_local_b(p1_vb, otil) * ysub_local_b(otil, p1_vb)).execute();

        auto oscalar = std::imag(tamm::sum(o_local_b));

        if(level == 1) { omega_ea_A0[ni] = oscalar; }
        else {
          if(level > 1) {
            T oerr          = oscalar - omega_ea_A0[ni];
            omega_ea_A0[ni] = oscalar;
            if(std::abs(oerr) < gf_threshold) omega_ea_conv_b[ni] = true;
          }
        }
        if(rank == 0) {
          std::ostringstream spf;
          spf << "W = " << std::fixed << std::setprecision(2) << std::real(omega_tmp)
              << ", omega_ea_A0 =  " << std::fixed << std::setprecision(4) << omega_ea_A0[ni]
              << endl;
          cout << spf.str();
          ni_A[ni] = omega_ea_A0[ni];
          ni_w[ni] = std::real(omega_tmp);
        }
      }

      cc_t2 = std::chrono::high_resolution_clock::now();
      time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
      if(rank == 0) {
        cout << endl << "omegas processed in level " << level << " = " << omega_extra << endl;
        cout << "Time to compute spectral function in level " << level
             << " (omega_npts_ea = " << omega_npts_ea << "): " << time << " secs" << endl;
        write_results_to_json(ec, sys_data, level, ni_w, ni_A, "advanced_beta");
      }

      auto               extrap_file = files_prefix + ".extrapolate.advanced.beta.txt";
      std::ostringstream spfe;
      spfe << "";

      // extrapolate or proceed to next level
      if(std::all_of(omega_ea_conv_b.begin(), omega_ea_conv_b.end(), [](bool x) { return x; }) ||
         gf_extrapolate_level == level) {
        if(rank == 0)
          cout << endl
               << "--------------------extrapolate & converge-----------------------" << endl;
        auto cc_t1 = std::chrono::high_resolution_clock::now();

        AtomicCounter* ac = new AtomicCounterGA(ec.pg(), 1);
        ac->allocate(0);
        int64_t taskcount = 0;
        int64_t next      = ac->fetch_add(0, 1);

        for(int64_t ni = 0; ni < lomega_npts_ea; ni++) {
          if(next == taskcount) {
            std::complex<T> omega_tmp = std::complex<T>(lomega_min_ea + ni * omega_delta_e, gf_eta);

            Complex2DMatrix ysub_b = (omega_tmp * hident - hsub_b).lu().solve(bsub_b);
            eigen_to_tamm_tensor(ysub_local_b, ysub_b);

            sch_l(o_local_b(p1_vb) = Cp_local_b(p1_vb, otil) * ysub_local_b(otil, p1_vb)).execute();

            auto oscalar = std::imag(tamm::sum(o_local_b));

            Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> olocala_eig(nvb);
            tamm_to_eigen_tensor(o_local_b, olocala_eig);
            for(TAMM_SIZE nj = 0; nj < nvb; nj++) {
              auto gpp = olocala_eig(nj).imag();
              spfe << "orb_index = " << nj << ", gpp_a = " << gpp << endl;
            }

            spfe << "w = " << std::fixed << std::setprecision(3) << std::real(omega_tmp)
                 << ", A_a =  " << std::fixed << std::setprecision(6) << oscalar << endl;
            next = ac->fetch_add(0, 1);
          }
          taskcount++;
        }

        ec.pg().barrier();
        ac->deallocate();
        delete ac;

        write_string_to_disk(ec, spfe.str(), extrap_file);
        if(rank == 0) {
          sys_data.results["output"]["GFCCSD"]["advanced_beta"]["nlevels"] = level;
          write_json_data(sys_data, "GFCCSD");
        }

        sch.deallocate(ysub_local_b, o_local_b, Cp_local_b, hsub_tamm_b, bsub_tamm_b, Cp_b)
          .execute();

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double time =
          std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        if(rank == 0)
          std::cout << endl
                    << "Time taken for extrapolation (lomega_npts_ea = " << lomega_npts_ea
                    << "): " << time << " secs" << endl;

        break;
      }
      else {
        if(level == 1) {
          auto o1 = (omega_extra[0] + omega_extra[1]) / 2;
          omega_extra.clear();
          o1 = find_closest(o1, omega_space_ea);
          omega_extra.push_back(o1);
        }
        else {
          std::sort(omega_extra_finished.begin(), omega_extra_finished.end());
          omega_extra.clear();
          std::vector<T> wtemp;
          for(size_t i = 1; i < omega_extra_finished.size(); i++) {
            bool   oe_add = false;
            auto   w1     = omega_extra_finished[i - 1];
            auto   w2     = omega_extra_finished[i];
            size_t num_w  = (w2 - w1) / omega_delta + 1;
            for(size_t j = 0; j < num_w; j++) {
              T      otmp = w1 + j * omega_delta;
              size_t ind  = (otmp - omega_min_ea) / omega_delta;
              if(!omega_ea_conv_b[ind]) {
                oe_add = true;
                break;
              }
            }

            if(oe_add) {
              T Win = (w1 + w2) / 2;
              Win   = find_closest(Win, omega_space_ea);
              if(std::find(omega_extra_finished.begin(), omega_extra_finished.end(), Win) !=
                 omega_extra_finished.end()) {
                continue;
              }
              else { omega_extra.push_back(Win); }
            } // end oe add
          }   // end oe finished
        }
        if(rank == 0) {
          cout << "new freq's:" << std::fixed << std::setprecision(2) << omega_extra << endl;
        }
        level++;
      }

      sch.deallocate(ysub_local_b, o_local_b, Cp_local_b, hsub_tamm_b, bsub_tamm_b, Cp_b).execute();

    } // end while
    // end of beta
  }
#endif // gf_ea

  sch.deallocate(cholVpr, d_f1, d_t1, d_t2).execute();

  /////////////////Free tensors////////////////////////////
  //#endif

  free_tensors(d_t1_a, d_t1_b, d_t2_aaaa, d_t2_bbbb, d_t2_abab, v2ijab_aaaa, v2ijab_abab,
               v2ijab_bbbb, v2ijab, v2ijka, v2iajb, cholOO_a, cholOO_b, cholOV_a, cholOV_b,
               cholVV_a, cholVV_b);

  if(ccsd_options.gf_ip) {
    free_tensors(t2v2_o, lt12_o_a, lt12_o_b, ix1_1_1_a, ix1_1_1_b, ix2_1_aaaa, ix2_1_abab,
                 ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a, ix2_3_b, ix2_4_aaaa, ix2_4_abab,
                 ix2_4_bbbb, ix2_5_aaaa, ix2_5_abba, ix2_5_abab, ix2_5_bbbb, ix2_5_baab, ix2_5_baba,
                 ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba, ix2_6_3_abab, ix2_6_3_bbbb,
                 ix2_6_3_baab, ix2_6_3_baba);
  }

#if 0
  if(ccsd_options.gf_ea) {
    free_tensors(t2v2_v, lt12_v_a, lt12_v_b, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy1_a, iy1_b,
                 iy2_a, iy2_b, iy3_1_aaaa, iy3_1_abba, iy3_1_baba, iy3_1_bbbb, iy3_1_baab,
                 iy3_1_abab, iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_baab, iy3_abba, iy3_bbbb, iy3_baba,
                 iy3_abab, iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb, iy4_1_abba, iy4_1_abab,
                 iy4_2_aaaa, iy4_2_abba, iy4_2_bbbb, iy4_2_baab, iy5_aaaa, iy5_baab, iy5_baba,
                 iy5_abba, iy5_bbbb, iy5_abab, iy6_a, iy6_b);
  }
#endif

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time = std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    cout << std::endl
         << "Time taken for GF-CCSD: " << std::fixed << std::setprecision(2) << ccsd_time << " secs"
         << std::endl;

  // ofs_profile.close();

  // --------- END GF CCSD -----------
  // GA_Summarize(0);
  ec.flush_and_sync();
  // MemoryManagerGA::destroy_coll(mgr);
  pg.destroy_coll();
  ec_l.flush_and_sync();
  // MemoryManagerLocal::destroy_coll(mgr_l);
  pg_l.destroy_coll();
  if(subcomm != MPI_COMM_NULL) {
    (*sub_ec).flush_and_sync();
    // MemoryManagerGA::destroy_coll(sub_mgr);
    MPI_Comm_free(&subcomm);
  }
  // delete ec;
}
