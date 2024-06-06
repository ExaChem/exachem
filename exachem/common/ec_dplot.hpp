/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "common/chemenv.hpp"

// #define X angular.row(3)
// #define Y angular.row(1)
// #define Z angular.row(2)
// #define dm2 4
// #define dm1 5
// #define d0 6
// #define dp1 7
// #define dp2 8
// #define fm3 9
// #define fm2 10
// #define fm1 11
// #define f0 12
// #define fp1 13
// #define fp2 14
// #define fp3 15
// #define gm4 16
// #define gm3 17
// #define gm2 18
// #define gm1 19
// #define g0 20
// #define gp1 21
// #define gp2 22
// #define gp3 23
// #define gp4 24
// #define hm5 25
// #define hm4 26
// #define hm3 27
// #define hm2 28
// #define hm1 29
// #define h0 30
// #define hp1 31
// #define hp2 32
// #define hp3 33
// #define hp4 34
// #define hp5 35

class EC_DPLOT {
public:
  static void write_dencube(ExecutionContext& ec, ChemEnv& chem_env, Matrix& D_alpha,
                            Matrix& D_beta, std::string files_prefix) {
    libint2::BasisSet& shells = chem_env.shells;
    std::vector<Atom>& atoms  = chem_env.atoms;
    // const bool         is_uhf     = chem_env.sys_data.is_unrestricted;
    // const int          nalpha     = chem_env.sys_data.nelectrons_alpha;
    // const int          nbeta      = chem_env.sys_data.nelectrons_beta;
    const int    natoms     = atoms.size();
    const int    nshells    = shells.size();
    const int    nbf        = shells.nbf();
    const int    batch_size = 8;
    const int    rank       = ec.pg().rank().value();
    const int    nranks     = ec.pg().size().value();
    const int    _x         = 3;
    const int    _y         = 1;
    const int    _z         = 2;
    const double sqrt3      = std::sqrt(3.0);
    const double sqrt5      = std::sqrt(5.0);
    const double sqrt10     = std::sqrt(10.0);
    const double sqrt6      = std::sqrt(6.0);
    const double sqrt70     = std::sqrt(70.0);
    const double sqrt35     = std::sqrt(35.0);
    const double sqrt14     = std::sqrt(14.0);
    const double sqrt15     = std::sqrt(15.0);
    const double sqrt105    = std::sqrt(105.0);

    Matrix D_shblk_norm = chem_env.compute_shellblock_norm(shells, D_alpha);

    Eigen::MatrixXd coords = Matrix::Zero(natoms, 3);
    for(int iatom = 0; iatom < natoms; iatom++) {
      coords(iatom, 0) = atoms[iatom].x;
      coords(iatom, 1) = atoms[iatom].y;
      coords(iatom, 2) = atoms[iatom].z;
    }
    const std::vector<double> spacing{0.1, 0.1, 0.1};
    const std::vector<double> minim{coords.col(0).minCoeff() - 5.0, coords.col(1).minCoeff() - 5.0,
                                    coords.col(2).minCoeff() - 5.0};
    const std::vector<double> maxim{coords.col(0).maxCoeff() + 5.0, coords.col(1).maxCoeff() + 5.0,
                                    coords.col(2).maxCoeff() + 5.0};
    const std::vector<int>    npoints{(int) std::ceil((maxim[0] - minim[0]) / spacing[0]),
                                   (int) std::ceil((maxim[1] - minim[1]) / spacing[1]),
                                   (int) std::ceil((maxim[2] - minim[2]) / spacing[2])};
    int                       xbatches = npoints[0] % batch_size == 0 ? npoints[0] / batch_size
                                                                      : npoints[0] / batch_size + 1;
    int                       ybatches = npoints[1] % batch_size == 0 ? npoints[1] / batch_size
                                                                      : npoints[1] / batch_size + 1;
    int                       zbatches = npoints[2] % batch_size == 0 ? npoints[2] / batch_size
                                                                      : npoints[2] / batch_size + 1;

    const double dV = spacing[0] * spacing[1] * spacing[2];
    double       rho_int{0.0};

    Eigen::VectorXi lmax_at(natoms);
    Eigen::VectorXd minalpha_at(natoms);
    auto            atom2shell = shells.atom2shell(atoms);
    for(int iatom = 0; iatom < natoms; iatom++) {
      auto atshells      = atom2shell[iatom];
      int  atnshells     = atshells.size();
      lmax_at(iatom)     = 0;
      minalpha_at(iatom) = 1.0e6;
      for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
        lmax_at(iatom) = std::max(lmax_at(iatom), shells[ish].contr[0].l);
        for(size_t iprim = 0; iprim < shells[ish].alpha.size(); iprim++) {
          minalpha_at(iatom) = std::min(minalpha_at(iatom), shells[ish].alpha[iprim]);
        }
      }
    }

    std::ofstream cubefile;
    if(rank == 0) {
      auto current_time   = std::chrono::system_clock::now();
      auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
      auto cur_local_time = localtime(&current_time_t);

      std::cout << std::endl << "Generation of CUBE file" << std::endl;
      cubefile.open(files_prefix + ".Dtot.cube");
      cubefile << std::setprecision(6);
      cubefile << "Total Density\n";
      cubefile << "Generated by ExaChem on " << std::put_time(cur_local_time, "%c") << "\n";
      cubefile << std::setw(6) << natoms << " " << std::fixed << std::setw(10) << minim[0] << " "
               << std::setw(10) << minim[1] << " " << std::setw(10) << minim[2] << std::endl;
      cubefile << std::setw(6) << npoints[0] << " " << std::setw(10) << spacing[0]
               << "   0.000000   0.000000\n";
      cubefile << std::setw(6) << npoints[1] << "   0.000000 " << std::setw(10) << spacing[1]
               << "   0.000000\n";
      cubefile << std::setw(6) << npoints[2] << "   0.000000   0.000000 " << std::setw(10)
               << spacing[2] << std::endl;
      for(int iatom = 0; iatom < natoms; iatom++) {
        cubefile << std::setw(3) << atoms[iatom].atomic_number << " " << std::setw(10)
                 << (double) atoms[iatom].atomic_number << " " << std::setw(10) << coords(iatom, 0)
                 << " " << std::setw(10) << coords(iatom, 1) << " " << std::setw(10)
                 << coords(iatom, 2) << std::endl;
      }
      cubefile << std::scientific << std::setprecision(6);
    }
    auto                     shell2bf = shells.shell2bf();
    int                      lmax     = shells.max_l();
    Eigen::Tensor<double, 3> rho(npoints[0], npoints[1], npoints[2]);
    rho.setZero();
    Eigen::ArrayXXd     angular((lmax + 1) * (lmax + 2) * (lmax + 3) / 6,
                                batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     gridxyz(3, batch_size * batch_size * batch_size);
    Eigen::MatrixXd     chi = Matrix::Zero(nbf, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     rgrid(3, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     radial(1, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     z2(1, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     rsq(1, batch_size * batch_size * batch_size);
    std::vector<bool>   skipshells(nshells);
    std::vector<double> local_rho(batch_size * batch_size * batch_size);
    angular.row(0).setConstant(1.0);

    int tbatch = -1;
    for(int xbatch = 0; xbatch < xbatches; xbatch++) {
      int ix_first = xbatch * batch_size;
      int xpoints  = std::min((xbatch + 1) * batch_size, npoints[0]) - ix_first;
      for(int ybatch = 0; ybatch < ybatches; ybatch++) {
        int iy_first = ybatch * batch_size;
        int ypoints  = std::min((ybatch + 1) * batch_size, npoints[1]) - iy_first;
        for(int zbatch = 0; zbatch < zbatches; zbatch++) {
          int iz_first     = zbatch * batch_size;
          int zpoints      = std::min((zbatch + 1) * batch_size, npoints[2]) - iz_first;
          int local_points = xpoints * ypoints * zpoints;
          tbatch += 1;
          if(tbatch % nranks != rank) continue;
          std::fill(skipshells.begin(), skipshells.end(), true);

          for(int ipoint = 0; ipoint < local_points; ipoint++) {
            int ix = ipoint / (ypoints * zpoints);
            int iy = (ipoint - ix * ypoints * zpoints) / zpoints;
            int iz = (ipoint - ix * ypoints * zpoints - iy * zpoints);
            ix += ix_first;
            iy += iy_first;
            iz += iz_first;
            gridxyz(0, ipoint) = minim[0] + spacing[0] * ix;
            gridxyz(1, ipoint) = minim[1] + spacing[1] * iy;
            gridxyz(2, ipoint) = minim[2] + spacing[2] * iz;
          }
          for(int iatom = 0; iatom < natoms; iatom++) {
            rgrid.row(0) = gridxyz.row(0) - atoms[iatom].x;
            rgrid.row(1) = gridxyz.row(1) - atoms[iatom].y;
            rgrid.row(2) = gridxyz.row(2) - atoms[iatom].z;
            for(int ipoint = 0; ipoint < local_points; ipoint++) {
              rsq(0, ipoint) = rgrid(0, ipoint) * rgrid(0, ipoint) +
                               rgrid(1, ipoint) * rgrid(1, ipoint) +
                               rgrid(2, ipoint) * rgrid(2, ipoint);
            }
            double minrsq = rsq.leftCols(local_points).minCoeff();
            if(minalpha_at(iatom) * minrsq > 30.0) continue;

            if(lmax_at(iatom) > 0) {
              angular.row(_y) = rgrid.row(1); // Y
              angular.row(_z) = rgrid.row(2); // Z
              angular.row(_x) = rgrid.row(0); // X
            }
            if(lmax_at(iatom) > 1) {
              angular.row(4) = sqrt3 * angular.row(_x) * angular.row(_y);
              angular.row(5) = sqrt3 * angular.row(_y) * angular.row(_z);
              angular.row(6) = 1.5 * angular.row(_z).pow(2) - 0.5 * rsq.row(0);
              angular.row(7) = sqrt3 * angular.row(_x) * angular.row(_z);
              angular.row(8) = 0.5 * sqrt3 * (angular.row(_x).pow(2) - angular.row(_y).pow(2));
            }
            if(lmax_at(iatom) > 2) {
              z2.row(0) = 5.0 * angular.row(_z).pow(2) - rsq.row(0);
              angular.row(9) =
                0.25 * sqrt10 * (sqrt3 * angular.row(4) * angular.row(_x) - angular.row(_y).pow(3));
              angular.row(10) = sqrt5 * angular.row(4) * angular.row(_z);
              angular.row(11) = 0.25 * sqrt6 * angular.row(_y) * z2.row(0);
              angular.row(12) = 0.5 * angular.row(_z) * (z2.row(0) - 2.0 * rsq.row(0));
              angular.row(13) = 0.25 * sqrt6 * angular.row(_x) * z2.row(0);
              angular.row(14) = sqrt5 * angular.row(8) * angular.row(2);
              angular.row(15) =
                0.25 * sqrt10 * (angular.row(_x).pow(3) - sqrt3 * angular.row(4) * angular.row(_y));
            }
            if(lmax_at(iatom) > 3) {
              angular.row(16) = 0.125 * sqrt35 *
                                (4.0 * angular.row(_x).pow(3) * angular.row(_y) -
                                 4.0 * angular.row(_x) * angular.row(_y).pow(3));
              angular.row(17) =
                0.25 * sqrt70 *
                (3.0 * angular.row(_x).pow(2) * angular.row(_y) - angular.row(_y).pow(3)) *
                angular.row(_z);
              angular.row(18) = 0.25 * sqrt5 * (2.0 * angular.row(_x) * angular.row(_y)) *
                                (7.0 * angular.row(_z).pow(2) - rsq.row(0));
              angular.row(19) = 0.25 * sqrt10 * angular.row(_y) * angular.row(_z) *
                                (7.0 * angular.row(_z).pow(2) - 3.0 * rsq.row(0));
              angular.row(20) =
                0.125 * (35.0 * angular.row(_z).pow(4) -
                         30.0 * angular.row(_z).pow(2) * rsq.row(0) + 3.0 * rsq.row(0).pow(2));
              angular.row(21) = 0.25 * sqrt10 * angular.row(_x) * angular.row(_z) *
                                (7.0 * angular.row(_z).pow(2) - 3.0 * rsq.row(0));
              angular.row(22) = 0.25 * sqrt5 * (angular.row(_x).pow(2) - angular.row(_y).pow(2)) *
                                (7.0 * angular.row(_z).pow(2) - rsq.row(0));
              angular.row(23) =
                0.25 * sqrt70 *
                (angular.row(_x).pow(3) - 3.0 * angular.row(_x).pow(2) * angular.row(_y)) *
                angular.row(_z);
              angular.row(24) =
                0.125 * sqrt35 *
                (angular.row(_x).pow(4) - 4.0 * angular.row(_x).pow(2) * angular.row(_y).pow(2) +
                 angular.row(_y).pow(4));
            }
            if(lmax_at(iatom) > 4) {
              angular.row(25) =
                0.1875 * sqrt14 *
                (5.0 * angular.row(_x).pow(4) * angular.row(_y) -
                 10.0 * angular.row(_x).pow(2) * angular.row(_y).pow(3) + angular.row(_y).pow(5));
              angular.row(26) = 0.375 * sqrt35 * angular.row(_z) *
                                (4.0 * angular.row(_x).pow(3) * angular.row(_y) -
                                 4.0 * angular.row(_x) * angular.row(_y).pow(3));
              angular.row(27) =
                0.0625 * sqrt70 * (angular.row(_z).pow(2) - rsq.row(0)) *
                (3.0 * angular.row(_x).pow(2) * angular.row(_y) - angular.row(_y).pow(3));
              angular.row(28) = 0.25 * sqrt105 * angular.row(_z) *
                                (3.0 * angular.row(_z).pow(2) - rsq.row(0)) *
                                (2.0 * angular.row(_x) * angular.row(_y));
              angular.row(29) = 0.125 * sqrt15 *
                                (21.0 * angular.row(_z).pow(4) -
                                 14.0 * angular.row(_z).pow(2) * rsq.row(0) + rsq.row(0).pow(2)) *
                                angular.row(_y);
              angular.row(30) =
                0.125 * angular.row(_z) *
                (63.0 * angular.row(_z).pow(4) - 70.0 * rsq.row(0) * angular.row(_z).pow(2) +
                 15.0 * rsq.row(0).pow(2));
              angular.row(31) = 0.125 * sqrt15 *
                                (21.0 * angular.row(_z).pow(4) -
                                 14.0 * angular.row(_z).pow(2) * rsq.row(0) + rsq.row(0).pow(2)) *
                                angular.row(_x);
              angular.row(32) = 0.25 * sqrt105 * angular.row(_z) *
                                (3.0 * angular.row(_z).pow(2) - rsq.row(0)) *
                                (angular.row(_x).pow(2) - angular.row(_y).pow(2));
              angular.row(33) =
                0.0625 * sqrt70 * (angular.row(_z).pow(2) - rsq.row(0)) *
                (angular.row(_x).pow(3) - 3.0 * angular.row(_x).pow(2) * angular.row(_y));
              angular.row(34) =
                0.375 * sqrt35 * angular.row(_z) *
                (angular.row(_x).pow(4) - 4.0 * angular.row(_x).pow(2) * angular.row(_y).pow(2) +
                 angular.row(_y).pow(4));
              angular.row(35) =
                0.1875 * sqrt14 *
                (angular.row(_x).pow(5) - 10.0 * angular.row(_x).pow(3) * angular.row(_y).pow(2) +
                 5.0 * angular.row(_x) * angular.row(_y).pow(4));
            }
            auto atshells  = atom2shell[iatom];
            int  atnshells = atshells.size();
            for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
              int l     = shells[ish].contr[0].l;
              int nprim = shells[ish].alpha.size();
              radial.setZero();
              int  bf_first  = shell2bf[ish];
              bool skipshell = true;
              for(int iprim = 0; iprim < nprim; iprim++) {
                if(minrsq * shells[ish].alpha[iprim] > 30.0) continue;
                skipshell = false;
                radial.row(0) +=
                  shells[ish].contr[0].coeff[iprim] * exp(-shells[ish].alpha[iprim] * rsq.row(0));
              }
              if(skipshell) continue;
              skipshells[ish] = skipshell;
              for(size_t ibf = bf_first, iang = (l * (l + 1) * (l + 2)) / 6;
                  ibf < bf_first + shells[ish].size(); ibf++, iang++)
                chi.row(ibf) = radial.row(0) * angular.row(iang);
            }
          }
          std::fill(local_rho.begin(), local_rho.end(), 0.0);
          for(int ish = 0; ish < nshells; ish++) {
            if(skipshells[ish]) continue;
            for(int jsh = ish; jsh < nshells; jsh++) {
              if(D_shblk_norm(ish, jsh) < 1.0e-30) continue;
              double factor = ish == jsh ? 1.0 : 2.0;
              if(skipshells[jsh]) continue;
              for(size_t ibf = shell2bf[ish]; ibf < shell2bf[ish] + shells[ish].size(); ibf++) {
                for(size_t jbf = shell2bf[jsh]; jbf < shell2bf[jsh] + shells[jsh].size(); jbf++) {
                  for(int ipoint = 0; ipoint < local_points; ipoint++) {
                    local_rho[ipoint] +=
                      factor * chi(ibf, ipoint) * D_alpha(ibf, jbf) * chi(jbf, ipoint);
                  }
                }
              }
            }
          }
          for(int ipoint = 0; ipoint < local_points; ipoint++) {
            int ix = ipoint / (ypoints * zpoints);
            int iy = (ipoint - ix * ypoints * zpoints) / zpoints;
            int iz = (ipoint - ix * ypoints * zpoints - iy * zpoints);
            ix += ix_first;
            iy += iy_first;
            iz += iz_first;
            rho_int += local_rho[ipoint] * dV;
            rho(ix, iy, iz) = local_rho[ipoint];
          }
        }
      }
    }
    Eigen::Tensor<double, 3> rrho(npoints[0], npoints[1], npoints[2]);
    double                   rho_tot{0.0};
    rrho.setZero();
    ec.pg().barrier();
    ec.pg().reduce(rho.data(), rrho.data(), npoints[0] * npoints[1] * npoints[2], ReduceOp::sum, 0);
    ec.pg().reduce(&rho_int, &rho_tot, 1, ReduceOp::sum, 0);
    if(rank == 0) {
      std::cout << "  - Number of Electrons = " << rho_tot << std::endl;
      for(int ix = 0; ix < npoints[0]; ix++) {
        for(int iy = 0; iy < npoints[1]; iy++) {
          for(int iz = 0; iz < npoints[2]; iz++) {
            cubefile << std::setw(14) << rrho(ix, iy, iz) << " ";
            if(iz % 6 == 5 || iz % npoints[2] == npoints[2] - 1) cubefile << "\n";
          }
        }
      }
    }
    cubefile.close();
  }

  static void write_mocube(ExecutionContext& ec, ChemEnv& chem_env, Matrix& movecs, int iorb,
                           std::string spin, std::string files_prefix) {
    libint2::BasisSet& shells = chem_env.shells;
    std::vector<Atom>& atoms  = chem_env.atoms;
    // const bool         is_uhf     = chem_env.sys_data.is_unrestricted;
    // const int          nalpha     = chem_env.sys_data.nelectrons_alpha;
    // const int          nbeta      = chem_env.sys_data.nelectrons_beta;
    const int    natoms     = atoms.size();
    const int    nshells    = shells.size();
    const int    nbf        = shells.nbf();
    const int    batch_size = 8;
    const int    rank       = ec.pg().rank().value();
    const int    nranks     = ec.pg().size().value();
    const int    _x         = 3;
    const int    _y         = 1;
    const int    _z         = 2;
    const double sqrt3      = std::sqrt(3.0);
    const double sqrt5      = std::sqrt(5.0);
    const double sqrt10     = std::sqrt(10.0);
    const double sqrt6      = std::sqrt(6.0);
    const double sqrt70     = std::sqrt(70.0);
    const double sqrt35     = std::sqrt(35.0);
    const double sqrt14     = std::sqrt(14.0);
    const double sqrt15     = std::sqrt(15.0);
    const double sqrt105    = std::sqrt(105.0);

    Eigen::MatrixXd coords = Matrix::Zero(natoms, 3);
    for(int iatom = 0; iatom < natoms; iatom++) {
      coords(iatom, 0) = atoms[iatom].x;
      coords(iatom, 1) = atoms[iatom].y;
      coords(iatom, 2) = atoms[iatom].z;
    }
    const std::vector<double> spacing{0.1, 0.1, 0.1};
    const std::vector<double> minim{coords.col(0).minCoeff() - 5.0, coords.col(1).minCoeff() - 5.0,
                                    coords.col(2).minCoeff() - 5.0};
    const std::vector<double> maxim{coords.col(0).maxCoeff() + 5.0, coords.col(1).maxCoeff() + 5.0,
                                    coords.col(2).maxCoeff() + 5.0};
    const std::vector<int>    npoints{(int) std::ceil((maxim[0] - minim[0]) / spacing[0]),
                                   (int) std::ceil((maxim[1] - minim[1]) / spacing[1]),
                                   (int) std::ceil((maxim[2] - minim[2]) / spacing[2])};
    int                       xbatches = npoints[0] % batch_size == 0 ? npoints[0] / batch_size
                                                                      : npoints[0] / batch_size + 1;
    int                       ybatches = npoints[1] % batch_size == 0 ? npoints[1] / batch_size
                                                                      : npoints[1] / batch_size + 1;
    int                       zbatches = npoints[2] % batch_size == 0 ? npoints[2] / batch_size
                                                                      : npoints[2] / batch_size + 1;

    // const double dV = spacing[0] * spacing[1] * spacing[2];
    // double       rho_int{0.0};

    Eigen::VectorXi lmax_at(natoms);
    Eigen::VectorXd minalpha_at(natoms);
    auto            atom2shell = shells.atom2shell(atoms);
    for(int iatom = 0; iatom < natoms; iatom++) {
      auto atshells      = atom2shell[iatom];
      int  atnshells     = atshells.size();
      lmax_at(iatom)     = 0;
      minalpha_at(iatom) = 1.0e6;
      for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
        lmax_at(iatom) = std::max(lmax_at(iatom), shells[ish].contr[0].l);
        for(size_t iprim = 0; iprim < shells[ish].alpha.size(); iprim++) {
          minalpha_at(iatom) = std::min(minalpha_at(iatom), shells[ish].alpha[iprim]);
        }
      }
    }

    std::ofstream cubefile;
    if(rank == 0) {
      auto current_time   = std::chrono::system_clock::now();
      auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
      auto cur_local_time = localtime(&current_time_t);

      std::cout << std::endl << "Generation of CUBE file" << std::endl;
      cubefile.open(files_prefix + ".MO" + std::to_string(iorb) + spin + ".cube");
      cubefile << std::setprecision(6);
      cubefile << "Molecular Orbital" + std::to_string(iorb) + "\n";
      cubefile << "Generated by ExaChem on " << std::put_time(cur_local_time, "%c") << "\n";
      cubefile << std::setw(6) << natoms << " " << std::fixed << std::setw(10) << minim[0] << " "
               << std::setw(10) << minim[1] << " " << std::setw(10) << minim[2] << std::endl;
      cubefile << std::setw(6) << npoints[0] << " " << std::setw(10) << spacing[0]
               << "   0.000000   0.000000\n";
      cubefile << std::setw(6) << npoints[1] << "   0.000000 " << std::setw(10) << spacing[1]
               << "   0.000000\n";
      cubefile << std::setw(6) << npoints[2] << "   0.000000   0.000000 " << std::setw(10)
               << spacing[2] << std::endl;
      for(int iatom = 0; iatom < natoms; iatom++) {
        cubefile << std::setw(3) << atoms[iatom].atomic_number << " " << std::setw(10)
                 << (double) atoms[iatom].atomic_number << " " << std::setw(10) << coords(iatom, 0)
                 << " " << std::setw(10) << coords(iatom, 1) << " " << std::setw(10)
                 << coords(iatom, 2) << std::endl;
      }
      cubefile << std::scientific << std::setprecision(6);
    }
    auto                     shell2bf = shells.shell2bf();
    int                      lmax     = shells.max_l();
    Eigen::Tensor<double, 3> rho(npoints[0], npoints[1], npoints[2]);
    rho.setZero();
    Eigen::ArrayXXd     angular((lmax + 1) * (lmax + 2) * (lmax + 3) / 6,
                                batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     gridxyz(3, batch_size * batch_size * batch_size);
    Eigen::MatrixXd     chi = Matrix::Zero(nbf, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     rgrid(3, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     radial(1, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     z2(1, batch_size * batch_size * batch_size);
    Eigen::ArrayXXd     rsq(1, batch_size * batch_size * batch_size);
    std::vector<bool>   skipshells(nshells);
    std::vector<double> local_rho(batch_size * batch_size * batch_size);
    angular.row(0).setConstant(1.0);

    int tbatch = -1;
    for(int xbatch = 0; xbatch < xbatches; xbatch++) {
      int ix_first = xbatch * batch_size;
      int xpoints  = std::min((xbatch + 1) * batch_size, npoints[0]) - ix_first;
      for(int ybatch = 0; ybatch < ybatches; ybatch++) {
        int iy_first = ybatch * batch_size;
        int ypoints  = std::min((ybatch + 1) * batch_size, npoints[1]) - iy_first;
        for(int zbatch = 0; zbatch < zbatches; zbatch++) {
          int iz_first     = zbatch * batch_size;
          int zpoints      = std::min((zbatch + 1) * batch_size, npoints[2]) - iz_first;
          int local_points = xpoints * ypoints * zpoints;
          tbatch += 1;
          if(tbatch % nranks != rank) continue;
          std::fill(skipshells.begin(), skipshells.end(), true);

          for(int ipoint = 0; ipoint < local_points; ipoint++) {
            int ix = ipoint / (ypoints * zpoints);
            int iy = (ipoint - ix * ypoints * zpoints) / zpoints;
            int iz = (ipoint - ix * ypoints * zpoints - iy * zpoints);
            ix += ix_first;
            iy += iy_first;
            iz += iz_first;
            gridxyz(0, ipoint) = minim[0] + spacing[0] * ix;
            gridxyz(1, ipoint) = minim[1] + spacing[1] * iy;
            gridxyz(2, ipoint) = minim[2] + spacing[2] * iz;
          }
          for(int iatom = 0; iatom < natoms; iatom++) {
            rgrid.row(0) = gridxyz.row(0) - atoms[iatom].x;
            rgrid.row(1) = gridxyz.row(1) - atoms[iatom].y;
            rgrid.row(2) = gridxyz.row(2) - atoms[iatom].z;
            for(int ipoint = 0; ipoint < local_points; ipoint++) {
              rsq(0, ipoint) = rgrid(0, ipoint) * rgrid(0, ipoint) +
                               rgrid(1, ipoint) * rgrid(1, ipoint) +
                               rgrid(2, ipoint) * rgrid(2, ipoint);
            }
            double minrsq = rsq.leftCols(local_points).minCoeff();
            if(minalpha_at(iatom) * minrsq > 30.0) continue;

            if(lmax_at(iatom) > 0) {
              angular.row(_y) = rgrid.row(1); // Y
              angular.row(_z) = rgrid.row(2); // Z
              angular.row(_x) = rgrid.row(0); // X
            }
            if(lmax_at(iatom) > 1) {
              angular.row(4) = sqrt3 * angular.row(_x) * angular.row(_y);
              angular.row(5) = sqrt3 * angular.row(_y) * angular.row(_z);
              angular.row(6) = 1.5 * angular.row(_z).pow(2) - 0.5 * rsq.row(0);
              angular.row(7) = sqrt3 * angular.row(_x) * angular.row(_z);
              angular.row(8) = 0.5 * sqrt3 * (angular.row(_x).pow(2) - angular.row(_y).pow(2));
            }
            if(lmax_at(iatom) > 2) {
              z2.row(0) = 5.0 * angular.row(_z).pow(2) - rsq.row(0);
              angular.row(9) =
                0.25 * sqrt10 * (sqrt3 * angular.row(4) * angular.row(_x) - angular.row(_y).pow(3));
              angular.row(10) = sqrt5 * angular.row(4) * angular.row(_z);
              angular.row(11) = 0.25 * sqrt6 * angular.row(_y) * z2.row(0);
              angular.row(12) = 0.5 * angular.row(_z) * (z2.row(0) - 2.0 * rsq.row(0));
              angular.row(13) = 0.25 * sqrt6 * angular.row(_x) * z2.row(0);
              angular.row(14) = sqrt5 * angular.row(8) * angular.row(2);
              angular.row(15) =
                0.25 * sqrt10 * (angular.row(_x).pow(3) - sqrt3 * angular.row(4) * angular.row(_y));
            }
            if(lmax_at(iatom) > 3) {
              angular.row(16) = 0.125 * sqrt35 *
                                (4.0 * angular.row(_x).pow(3) * angular.row(_y) -
                                 4.0 * angular.row(_x) * angular.row(_y).pow(3));
              angular.row(17) =
                0.25 * sqrt70 *
                (3.0 * angular.row(_x).pow(2) * angular.row(_y) - angular.row(_y).pow(3)) *
                angular.row(_z);
              angular.row(18) = 0.25 * sqrt5 * (2.0 * angular.row(_x) * angular.row(_y)) *
                                (7.0 * angular.row(_z).pow(2) - rsq.row(0));
              angular.row(19) = 0.25 * sqrt10 * angular.row(_y) * angular.row(_z) *
                                (7.0 * angular.row(_z).pow(2) - 3.0 * rsq.row(0));
              angular.row(20) =
                0.125 * (35.0 * angular.row(_z).pow(4) -
                         30.0 * angular.row(_z).pow(2) * rsq.row(0) + 3.0 * rsq.row(0).pow(2));
              angular.row(21) = 0.25 * sqrt10 * angular.row(_x) * angular.row(_z) *
                                (7.0 * angular.row(_z).pow(2) - 3.0 * rsq.row(0));
              angular.row(22) = 0.25 * sqrt5 * (angular.row(_x).pow(2) - angular.row(_y).pow(2)) *
                                (7.0 * angular.row(_z).pow(2) - rsq.row(0));
              angular.row(23) =
                0.25 * sqrt70 *
                (angular.row(_x).pow(3) - 3.0 * angular.row(_x).pow(2) * angular.row(_y)) *
                angular.row(_z);
              angular.row(24) =
                0.125 * sqrt35 *
                (angular.row(_x).pow(4) - 4.0 * angular.row(_x).pow(2) * angular.row(_y).pow(2) +
                 angular.row(_y).pow(4));
            }
            if(lmax_at(iatom) > 4) {
              angular.row(25) =
                0.1875 * sqrt14 *
                (5.0 * angular.row(_x).pow(4) * angular.row(_y) -
                 10.0 * angular.row(_x).pow(2) * angular.row(_y).pow(3) + angular.row(_y).pow(5));
              angular.row(26) = 0.375 * sqrt35 * angular.row(_z) *
                                (4.0 * angular.row(_x).pow(3) * angular.row(_y) -
                                 4.0 * angular.row(_x) * angular.row(_y).pow(3));
              angular.row(27) =
                0.0625 * sqrt70 * (angular.row(_z).pow(2) - rsq.row(0)) *
                (3.0 * angular.row(_x).pow(2) * angular.row(_y) - angular.row(_y).pow(3));
              angular.row(28) = 0.25 * sqrt105 * angular.row(_z) *
                                (3.0 * angular.row(_z).pow(2) - rsq.row(0)) *
                                (2.0 * angular.row(_x) * angular.row(_y));
              angular.row(29) = 0.125 * sqrt15 *
                                (21.0 * angular.row(_z).pow(4) -
                                 14.0 * angular.row(_z).pow(2) * rsq.row(0) + rsq.row(0).pow(2)) *
                                angular.row(_y);
              angular.row(30) =
                0.125 * angular.row(_z) *
                (63.0 * angular.row(_z).pow(4) - 70.0 * rsq.row(0) * angular.row(_z).pow(2) +
                 15.0 * rsq.row(0).pow(2));
              angular.row(31) = 0.125 * sqrt15 *
                                (21.0 * angular.row(_z).pow(4) -
                                 14.0 * angular.row(_z).pow(2) * rsq.row(0) + rsq.row(0).pow(2)) *
                                angular.row(_x);
              angular.row(32) = 0.25 * sqrt105 * angular.row(_z) *
                                (3.0 * angular.row(_z).pow(2) - rsq.row(0)) *
                                (angular.row(_x).pow(2) - angular.row(_y).pow(2));
              angular.row(33) =
                0.0625 * sqrt70 * (angular.row(_z).pow(2) - rsq.row(0)) *
                (angular.row(_x).pow(3) - 3.0 * angular.row(_x).pow(2) * angular.row(_y));
              angular.row(34) =
                0.375 * sqrt35 * angular.row(_z) *
                (angular.row(_x).pow(4) - 4.0 * angular.row(_x).pow(2) * angular.row(_y).pow(2) +
                 angular.row(_y).pow(4));
              angular.row(35) =
                0.1875 * sqrt14 *
                (angular.row(_x).pow(5) - 10.0 * angular.row(_x).pow(3) * angular.row(_y).pow(2) +
                 5.0 * angular.row(_x) * angular.row(_y).pow(4));
            }
            auto atshells  = atom2shell[iatom];
            int  atnshells = atshells.size();
            for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
              int l     = shells[ish].contr[0].l;
              int nprim = shells[ish].alpha.size();
              radial.setZero();
              int  bf_first  = shell2bf[ish];
              bool skipshell = true;
              for(int iprim = 0; iprim < nprim; iprim++) {
                if(minrsq * shells[ish].alpha[iprim] > 30.0) continue;
                skipshell = false;
                radial.row(0) +=
                  shells[ish].contr[0].coeff[iprim] * exp(-shells[ish].alpha[iprim] * rsq.row(0));
              }
              if(skipshell) continue;
              skipshells[ish] = skipshell;
              for(size_t ibf = bf_first, iang = (l * (l + 1) * (l + 2)) / 6;
                  ibf < bf_first + shells[ish].size(); ibf++, iang++)
                chi.row(ibf) = radial.row(0) * angular.row(iang);
            }
          }
          std::fill(local_rho.begin(), local_rho.end(), 0.0);
          for(int ish = 0; ish < nshells; ish++) {
            if(skipshells[ish]) continue;
            if(movecs.block(shell2bf[ish], iorb, shells[ish].size(), 1).norm() < 1.0e-30) continue;
            for(size_t ibf = shell2bf[ish]; ibf < shell2bf[ish] + shells[ish].size(); ibf++) {
              for(int ipoint = 0; ipoint < local_points; ipoint++) {
                local_rho[ipoint] += chi(ibf, ipoint) * movecs(ibf, iorb);
              }
            }
          }
          for(int ipoint = 0; ipoint < local_points; ipoint++) {
            int ix = ipoint / (ypoints * zpoints);
            int iy = (ipoint - ix * ypoints * zpoints) / zpoints;
            int iz = (ipoint - ix * ypoints * zpoints - iy * zpoints);
            ix += ix_first;
            iy += iy_first;
            iz += iz_first;
            rho(ix, iy, iz) = local_rho[ipoint];
          }
        }
      }
    }
    Eigen::Tensor<double, 3> rrho(npoints[0], npoints[1], npoints[2]);
    rrho.setZero();
    ec.pg().barrier();
    ec.pg().reduce(rho.data(), rrho.data(), npoints[0] * npoints[1] * npoints[2], ReduceOp::sum, 0);
    if(rank == 0) {
      for(int ix = 0; ix < npoints[0]; ix++) {
        for(int iy = 0; iy < npoints[1]; iy++) {
          for(int iz = 0; iz < npoints[2]; iz++) {
            cubefile << std::setw(14) << rrho(ix, iy, iz) << " ";
            if(iz % 6 == 5 || iz % npoints[2] == npoints[2] - 1) cubefile << "\n";
          }
        }
      }
    }
    cubefile.close();
  }

#if 0
  void compute_chi(int natoms, int local_points, std::vector<bool>& skipshells,
                   std::vector<Atom>& atoms, libint2::BasisSet& shells, Eigen::VectorXi lmax_at,
                   Eigen::ArrayXXd& angular, Eigen::ArrayXXd& radial, Eigen::ArrayXXd& rgrid,
                   Eigen::ArrayXXd& z2, EigenArrayXXd& rsq, Eigen::MatrixXd& chi,
                   Eigen::ArrayXXd& gridxyz, Eigen::ArrayXXd& Am, Eigen::ArrayXXd& Bm,
                   Eigen::ArrayXXd& Pi) {
    constexpr double sqrt3   = std::sqrt(3.0);
    constexpr double sqrt5   = std::sqrt(5.0);
    constexpr double sqrt10  = std::sqrt(10.0);
    constexpr double sqrt6   = std::sqrt(6.0);
    constexpr double sqrt70  = std::sqrt(70.0);
    constexpr double sqrt35  = std::sqrt(35.0);
    constexpr double sqrt14  = std::sqrt(14.0);
    constexpr double sqrt15  = std::sqrt(15.0);
    constexpr double sqrt105 = std::sqrt(105.0);

    for(int iatom = 0; iatom < natoms; iatom++) {
      rgrid.row(0) = gridxyz.row(0) - atoms[iatom].x;
      rgrid.row(1) = gridxyz.row(1) - atoms[iatom].y;
      rgrid.row(2) = gridxyz.row(2) - atoms[iatom].z;
      for(int ipoint = 0; ipoint < local_points; ipoint++) {
        rsq(0, ipoint) = rgrid(0, ipoint) * rgrid(0, ipoint) + rgrid(1, ipoint) * rgrid(1, ipoint) +
                         rgrid(2, ipoint) * rgrid(2, ipoint);
      }
      double minrsq = rsq.leftCols(local_points).minCoeff();
      if(minalpha_at(iatom) * minrsq > 30.0) continue;

      Am.row(1) = X;
      Bm.row(1) = Y;
      for(int m = 2; m <= lmax_at(iatom); m++) {
        Am.row(m) = Am.row(m - 1) * X - Bm.row(m - 1) * Y;
        Bm.row(m) = Am.row(m - 1) * Y + Bm.row(m - 1) * X;
      }

      Pi.row(0) = 1.0;                                           // 0,0
      Pi.row(1) = Z;                                             // 0,1
      Pi.row(2) = 1.0;                                           // 1,1
      Pi.row(3) = 0.5 * (3.0 * Z.pow(2) - rsq.row(0));           // 0,2
      Pi.row(4) = sqrt3 * Z;                                     // 1,2
      Pi.row(5) = 0.5 * sqrt3;                                   // 2,2
      Pi.row(6) = 0.5 * Z * (5.0 * Z.pow(2) - 3.0 * rsq.row(0)); // 0,3
      Pi.row(7) = 0.25 * sqrt6 * (5.0 * Z.pow(2) - rsq.row(0));  // 1,3
      Pi.row(8) = 0.5 * sqrt15 * Z;                              // 2,3
      Pi.row(9) = 0.25 * sqrt10;                                 // 3,3
      Pi.row(10) =
        0.125 * (35.0 * Z.pow(4) - 30.0 * rsq.row(0) * Z.pow(2) + 3 * rsq.row(0).pow(2)); // 0,4
      Pi.row(11) = 0.25 * sqrt10 * Z * (7.0 * rsq.row(0) - 3 * rsq.row(0));               // 1,4
      Pi.row(12) = 0.25 * sqrt5 * (7.0 * rsq.row(0) - rsq.row(0));                        // 2,4
      Pi.row(13) = 0.25 * sqrt70 * Z;                                                     // 3,4
      Pi.row(14) = 0.125 * sqrt35;                                                        // 4,4
      Pi.row(15) =
        0.125 * Z * (63 * Z.pow(4) - 70 * Z.pow(2) * rsq.row(0) + 15.0 * rsq.row(0).pow(2)); // 0,5
      Pi.row(16) =
        0.125 * sqrt15 * (21 * Z.pow(4) - 14 * Z.pow(2) * rsq.row(0) + rsq.row(0).pow(2)); // 1,5
      Pi.row(17) = 0.25 * sqrt105 * Z * (3.0 * Z.pow(2) - rsq.row(0));                     // 2,5
      Pi.row(18) = 0.0625 * sqrt70 * (9.0 * Z.pow(2) - rsq.row(0));                        // 3,5
      Pi.row(19) = 0.375 * sqrt35 * Z;                                                     // 4,5
      Pi.row(20) = 0.1875 * sqrt14;                                                        // 5,5

      for(int l = 1, iang = 1; l <= lmax_at(iatom); l++) {
        ipi = l * (l + 1) / 2;

        // For m < 0
        for(int m = l; m > 0; m--, iang++) { angular.row(iang) = Bm(m) * Pi.row(ipi + m); }
        // For m == 0
        angular.row(iang) = Pi.row(ipi);
        iang++;
        // For m > 0
        for(int m = 1; m <= l; m++, iang++) { angular.row(iang) = Am(m) * Pi.row(ipi + m); }
      }

      auto atshells  = atom2shell[iatom];
      int  atnshells = atshells.size();
      for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
        int l     = shells[ish].contr[0].l;
        int nprim = shells[ish].alpha.size();
        radial.setZero();
        int  bf_first  = shell2bf[ish];
        bool skipshell = true;
        for(int iprim = 0; iprim < nprim; iprim++) {
          if(minrsq * shells[ish].alpha[iprim] > 30.0) continue;
          skipshell = false;
          radial.row(0) +=
            shells[ish].contr[0].coeff[iprim] * exp(-shells[ish].alpha[iprim] * rsq.row(0));
        }
        if(skipshell) continue;
        skipshells[ish] = skipshell;
        for(size_t ibf = bf_first, iang = (l * (l + 1) * (l + 2)) / 6;
            ibf < bf_first + shells[ish].size(); ibf++, iang++)
          chi.row(ibf) = radial.row(0) * angular.row(iang);
      }
    }
  }
#endif
};
