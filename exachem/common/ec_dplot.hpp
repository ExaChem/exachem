/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"

class EigenDPLOT {
public:
  Eigen::MatrixXd chi;
  Eigen::ArrayXXd gridxyz, Am, Bm, Pi, angular;
  Eigen::ArrayXd  x, y, z, z2, z4, r2, r4, radial;
  Eigen::VectorXi lmax_at;
  Eigen::VectorXd minalpha_at;

  void init(std::vector<Atom>& atoms, libint2::BasisSet& shells, int batch_size) {
    const int natoms     = atoms.size();
    const int nbf        = shells.nbf();
    auto      atom2shell = shells.atom2shell(atoms);
    auto      shell2bf   = shells.shell2bf();
    const int lmax       = shells.max_l();

    lmax_at.resize(natoms);
    minalpha_at.resize(natoms);
    chi = Matrix::Zero(nbf, batch_size);
    gridxyz.resize(3, batch_size);
    x.resize(batch_size);
    y.resize(batch_size);
    z.resize(batch_size);
    z2.resize(batch_size);
    z4.resize(batch_size);
    r2.resize(batch_size);
    r4.resize(batch_size);
    radial.resize(batch_size);
    angular.resize((lmax + 1) * (lmax + 1), batch_size);
    Am.resize(lmax + 1, batch_size);
    Bm.resize(lmax + 1, batch_size);
    Pi.resize(((lmax + 1) * (lmax + 2)) / 2, batch_size);

    angular.row(0).setConstant(1.0);
    Pi.row(0).setConstant(1.0);                                    // 0,0
    if(lmax > 0) Pi.row(2).setConstant(1.0);                       // 1,1
    if(lmax > 1) Pi.row(5).setConstant(0.5 * std::sqrt(3.0));      // 2,2
    if(lmax > 2) Pi.row(9).setConstant(0.25 * std::sqrt(10.0));    // 3,3
    if(lmax > 3) Pi.row(14).setConstant(0.125 * std::sqrt(35.0));  // 4,4
    if(lmax > 4) Pi.row(20).setConstant(0.1875 * std::sqrt(14.0)); // 5,5

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
  }
};

class EC_DPLOT {
public:
  static void write_dencube(ExecutionContext& ec, ChemEnv& chem_env, Matrix& D_alpha,
                            Matrix& D_beta, std::string files_prefix) {
    libint2::BasisSet& shells = chem_env.shells;
    std::vector<Atom>& atoms  = chem_env.atoms;
    // const bool         is_uhf     = chem_env.sys_data.is_unrestricted;
    // const int          nalpha     = chem_env.sys_data.nelectrons_alpha;
    // const int          nbeta      = chem_env.sys_data.nelectrons_beta;
    const int    nshells     = shells.size();
    const int    batch_size  = 8;
    const int    batch_size3 = batch_size * batch_size * batch_size;
    const int    rank        = ec.pg().rank().value();
    const int    nranks      = ec.pg().size().value();
    const double padding     = 6.0;
    const double space       = 0.1;

    Matrix D_shblk_norm = chem_env.compute_shellblock_norm(shells, D_alpha);

    std::vector<double> spacing, minim, maxim;
    std::vector<int>    npoints, batches;
    std::tie(spacing, minim, maxim, npoints, batches) =
      define_grid(atoms, padding, space, batch_size);

    const double dV = spacing[0] * spacing[1] * spacing[2];
    double       rho_int{0.0};
    auto         shell2bf = shells.shell2bf();

    EigenDPLOT etensors;
    etensors.init(atoms, shells, batch_size3);

    std::ofstream cubefile;
    if(rank == 0) {
      std::string filename = files_prefix + ".Dtot.cube";
      std::string header   = "Total Density ";
      write_header(cubefile, filename, header, atoms, minim, spacing, npoints);
    }

    Eigen::Tensor<double, 3> rho(npoints[0], npoints[1], npoints[2]);
    rho.setZero();
    std::vector<bool>   skipshells(nshells);
    std::vector<double> local_rho(batch_size3);

    int tbatch = -1;
    for(int xbatch = 0; xbatch < batches[0]; xbatch++) {
      int ix_first = xbatch * batch_size;
      int xpoints  = std::min((xbatch + 1) * batch_size, npoints[0]) - ix_first;
      for(int ybatch = 0; ybatch < batches[1]; ybatch++) {
        int iy_first = ybatch * batch_size;
        int ypoints  = std::min((ybatch + 1) * batch_size, npoints[1]) - iy_first;
        for(int zbatch = 0; zbatch < batches[2]; zbatch++) {
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
            etensors.gridxyz(0, ipoint) = minim[0] + spacing[0] * ix;
            etensors.gridxyz(1, ipoint) = minim[1] + spacing[1] * iy;
            etensors.gridxyz(2, ipoint) = minim[2] + spacing[2] * iz;
          }

          compute_chi(local_points, skipshells, atoms, shells, etensors);

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
                    local_rho[ipoint] += factor * etensors.chi(ibf, ipoint) * D_alpha(ibf, jbf) *
                                         etensors.chi(jbf, ipoint);
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
    const int           nshells     = shells.size();
    const int           batch_size  = 8;
    const int           batch_size3 = batch_size * batch_size * batch_size;
    const int           rank        = ec.pg().rank().value();
    const int           nranks      = ec.pg().size().value();
    const double        padding     = 6.0;
    const double        space       = 0.1;
    std::vector<double> spacing, minim, maxim;
    std::vector<int>    npoints, batches;
    std::tie(spacing, minim, maxim, npoints, batches) =
      define_grid(atoms, padding, space, batch_size);

    const double dV = spacing[0] * spacing[1] * spacing[2];
    double       rho_int{0.0};

    auto shell2bf = shells.shell2bf();

    EigenDPLOT etensors;
    etensors.init(atoms, shells, batch_size3);

    std::ofstream cubefile;
    if(rank == 0) {
      std::string filename = files_prefix + ".MO" + std::to_string(iorb) + spin + ".cube";
      std::string header   = "Molecular Orbital " + std::to_string(iorb) + spin;
      write_header(cubefile, filename, header, atoms, minim, spacing, npoints);
    }

    Eigen::Tensor<double, 3> rho(npoints[0], npoints[1], npoints[2]);
    rho.setZero();
    std::vector<bool>   skipshells(nshells);
    std::vector<double> local_rho(batch_size3);

    int tbatch = -1;
    for(int xbatch = 0; xbatch < batches[0]; xbatch++) {
      int ix_first = xbatch * batch_size;
      int xpoints  = std::min((xbatch + 1) * batch_size, npoints[0]) - ix_first;
      for(int ybatch = 0; ybatch < batches[1]; ybatch++) {
        int iy_first = ybatch * batch_size;
        int ypoints  = std::min((ybatch + 1) * batch_size, npoints[1]) - iy_first;
        for(int zbatch = 0; zbatch < batches[2]; zbatch++) {
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
            etensors.gridxyz(0, ipoint) = minim[0] + spacing[0] * ix;
            etensors.gridxyz(1, ipoint) = minim[1] + spacing[1] * iy;
            etensors.gridxyz(2, ipoint) = minim[2] + spacing[2] * iz;
          }

          compute_chi(local_points, skipshells, atoms, shells, etensors);

          std::fill(local_rho.begin(), local_rho.end(), 0.0);
          for(int ish = 0; ish < nshells; ish++) {
            if(skipshells[ish]) continue;
            if(movecs.block(shell2bf[ish], iorb, shells[ish].size(), 1).norm() < 1.0e-32) continue;
            for(size_t ibf = shell2bf[ish]; ibf < shell2bf[ish] + shells[ish].size(); ibf++) {
              for(int ipoint = 0; ipoint < local_points; ipoint++) {
                local_rho[ipoint] += etensors.chi(ibf, ipoint) * movecs(ibf, iorb);
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
            rho_int += local_rho[ipoint] * local_rho[ipoint] * dV;
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
      std::cout << "  - Norm of Orbital = " << rho_tot << std::endl;
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

private:
  static void write_header(std::ofstream& cubefile, std::string& filename, std::string& header,
                           std::vector<Atom>& atoms, std::vector<double>& minim,
                           std::vector<double>& spacing, std::vector<int>& npoints) {
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);

    std::cout << std::endl << "Generation of CUBE file" << std::endl;
    cubefile.open(filename);
    cubefile << std::setprecision(6);
    cubefile << header << std::endl;
    cubefile << "Generated by ExaChem on " << std::put_time(cur_local_time, "%c") << "\n";
    cubefile << std::setw(6) << atoms.size() << " " << std::fixed << std::setw(10) << minim[0]
             << " " << std::setw(10) << minim[1] << " " << std::setw(10) << minim[2] << std::endl;
    cubefile << std::setw(6) << npoints[0] << " " << std::setw(10) << spacing[0]
             << "   0.000000   0.000000\n";
    cubefile << std::setw(6) << npoints[1] << "   0.000000 " << std::setw(10) << spacing[1]
             << "   0.000000\n";
    cubefile << std::setw(6) << npoints[2] << "   0.000000   0.000000 " << std::setw(10)
             << spacing[2] << std::endl;
    for(size_t iatom = 0; iatom < atoms.size(); iatom++) {
      cubefile << std::setw(3) << atoms[iatom].atomic_number << " " << std::setw(10)
               << static_cast<double>(atoms[iatom].atomic_number) << " " << std::setw(10)
               << atoms[iatom].x << " " << std::setw(10) << atoms[iatom].y << " " << std::setw(10)
               << atoms[iatom].z << std::endl;
    }
    cubefile << std::scientific << std::setprecision(6);
  }

  static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>,
                    std::vector<int>>
  define_grid(std::vector<Atom>& atoms, const double padding, const double space,
              const int batch_size) {
    const int natoms = atoms.size();
    double    xmax = atoms[0].x, ymax = atoms[0].y, zmax = atoms[0].z;
    double    xmin = atoms[0].x, ymin = atoms[0].y, zmin = atoms[0].z;

    for(int iatom = 1; iatom < natoms; iatom++) {
      xmax = std::max(xmax, atoms[iatom].x);
      ymax = std::max(ymax, atoms[iatom].y);
      zmax = std::max(zmax, atoms[iatom].z);
      xmin = std::min(xmin, atoms[iatom].x);
      ymin = std::min(ymin, atoms[iatom].y);
      zmin = std::min(zmin, atoms[iatom].z);
    }

    const std::vector<double> spacing{space, space, space};
    const std::vector<double> minim{xmin - padding, ymin - padding, zmin - padding};
    const std::vector<double> maxim{xmax + padding, ymax + padding, zmax + padding};
    const std::vector<int> npoints{static_cast<int>(std::ceil((maxim[0] - minim[0]) / spacing[0])),
                                   static_cast<int>(std::ceil((maxim[1] - minim[1]) / spacing[1])),
                                   static_cast<int>(std::ceil((maxim[2] - minim[2]) / spacing[2]))};
    const std::vector<int> batches{
      npoints[0] % batch_size == 0 ? npoints[0] / batch_size : npoints[0] / batch_size + 1,
      npoints[1] % batch_size == 0 ? npoints[1] / batch_size : npoints[1] / batch_size + 1,
      npoints[2] % batch_size == 0 ? npoints[2] / batch_size : npoints[2] / batch_size + 1};

    return std::tuple(spacing, minim, maxim, npoints, batches);
  }

  static void compute_chi(int local_points, std::vector<bool>& skipshells, std::vector<Atom>& atoms,
                          libint2::BasisSet& shells, EigenDPLOT& etensors) {
    double sqrt3   = std::sqrt(3.0);
    double sqrt5   = std::sqrt(5.0);
    double sqrt6   = std::sqrt(6.0);
    double sqrt10  = std::sqrt(10.0);
    double sqrt15  = std::sqrt(15.0);
    double sqrt35  = std::sqrt(35.0);
    double sqrt70  = std::sqrt(70.0);
    double sqrt105 = std::sqrt(105.0);

    const int natoms     = atoms.size();
    auto      atom2shell = shells.atom2shell(atoms);
    auto      shell2bf   = shells.shell2bf();

    for(int iatom = 0; iatom < natoms; iatom++) {
      etensors.x  = etensors.gridxyz.row(0) - atoms[iatom].x;
      etensors.y  = etensors.gridxyz.row(1) - atoms[iatom].y;
      etensors.z  = etensors.gridxyz.row(2) - atoms[iatom].z;
      etensors.r2 = etensors.x.square() + etensors.y.square() + etensors.z.square();

      // double minrsq = rsq.leftCols(local_points).minCoeff();
      double minrsq = etensors.r2.head(local_points).minCoeff();
      if(etensors.minalpha_at(iatom) * minrsq > 35.0) continue;

      etensors.Am.row(1) = etensors.x;
      etensors.Bm.row(1) = etensors.y;
      for(int m = 2; m <= etensors.lmax_at(iatom); m++) {
        etensors.Am.row(m) = etensors.Am.row(m - 1) * etensors.x.transpose() -
                             etensors.Bm.row(m - 1) * etensors.y.transpose();
        etensors.Bm.row(m) = etensors.Am.row(m - 1) * etensors.y.transpose() +
                             etensors.Bm.row(m - 1) * etensors.x.transpose();
      }

      if(etensors.lmax_at(iatom) > 0) etensors.Pi.row(1) = etensors.z; // 0,1

      if(etensors.lmax_at(iatom) > 1) {
        etensors.z2        = etensors.z.square();
        etensors.Pi.row(3) = 0.5 * (3.0 * etensors.z2 - etensors.r2); // 0,2
        etensors.Pi.row(4) = sqrt3 * etensors.z;                      // 1,2
      };

      if(etensors.lmax_at(iatom) > 2) {
        etensors.Pi.row(6) = 0.5 * etensors.z * (5.0 * etensors.z2 - 3.0 * etensors.r2); // 0,3
        etensors.Pi.row(7) = 0.25 * sqrt6 * (5.0 * etensors.z2 - etensors.r2);           // 1,3
        etensors.Pi.row(8) = 0.5 * sqrt15 * etensors.z;                                  // 2,3
      };

      if(etensors.lmax_at(iatom) > 3) {
        etensors.z4 = etensors.z2.square();
        etensors.r4 = etensors.r2.square();
        etensors.Pi.row(10) =
          0.125 * (35.0 * etensors.z4 - 30.0 * etensors.r2 * etensors.z2 + 3 * etensors.r4); // 0,4
        etensors.Pi.row(11) =
          0.25 * sqrt10 * etensors.z * (7.0 * etensors.z2 - 3 * etensors.r2);   // 1,4
        etensors.Pi.row(12) = 0.25 * sqrt5 * (7.0 * etensors.z2 - etensors.r2); // 2,4
        etensors.Pi.row(13) = 0.25 * sqrt70 * etensors.z;                       // 3,4
      };

      if(etensors.lmax_at(iatom) > 4) {
        etensors.Pi.row(15) =
          0.125 * etensors.z *
          (63 * etensors.z4 - 70 * etensors.z2 * etensors.r2 + 15.0 * etensors.r4); // 0,5
        etensors.Pi.row(16) =
          0.125 * sqrt15 * (21 * etensors.z4 - 14 * etensors.z2 * etensors.r2 + etensors.r4); // 1,5
        etensors.Pi.row(17) =
          0.25 * sqrt105 * etensors.z * (3.0 * etensors.z2 - etensors.r2);         // 2,5
        etensors.Pi.row(18) = 0.0625 * sqrt70 * (9.0 * etensors.z2 - etensors.r2); // 3,5
        etensors.Pi.row(19) = 0.375 * sqrt35 * etensors.z;                         // 4,5
      };

      for(int l = 1, iang = 1; l <= etensors.lmax_at(iatom); l++) {
        int ipi = l * (l + 1) / 2;
        // For m < 0
        for(int m = l; m > 0; m--, iang++) {
          etensors.angular.row(iang) = etensors.Bm.row(m) * etensors.Pi.row(ipi + m);
        }
        // For m == 0
        etensors.angular.row(iang) = etensors.Pi.row(ipi);
        iang++;
        // For m > 0
        for(int m = 1; m <= l; m++, iang++) {
          etensors.angular.row(iang) = etensors.Am.row(m) * etensors.Pi.row(ipi + m);
        }
      }

      auto atshells  = atom2shell[iatom];
      int  atnshells = atshells.size();
      for(int ish = atshells[0]; ish < atshells[0] + atnshells; ish++) {
        int l     = shells[ish].contr[0].l;
        int nprim = shells[ish].alpha.size();
        etensors.radial.setZero();
        int  bf_first  = shell2bf[ish];
        bool skipshell = true;
        for(int iprim = 0; iprim < nprim; iprim++) {
          if(minrsq * shells[ish].alpha[iprim] > 35.0) continue;
          skipshell = false;
          etensors.radial +=
            shells[ish].contr[0].coeff[iprim] * exp(-shells[ish].alpha[iprim] * etensors.r2);
        }
        if(skipshell) continue;
        skipshells[ish] = skipshell;
        for(size_t ibf = bf_first, iang = l * l; ibf < bf_first + shells[ish].size();
            ibf++, iang++) {
          etensors.chi.row(ibf) = etensors.radial.transpose() * etensors.angular.row(iang);
        }
      }
    }
  }
};
