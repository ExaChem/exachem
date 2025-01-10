/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/geometry_analysis.hpp"
#include <algorithm>

namespace exachem::task {

double ang_to_bohr = 1.8897259878858;
double bohr_to_ang = 1 / ang_to_bohr;

std::vector<double> atom_mass = {
  1.007825, 4.0026,    7.016,    9.01218,  11.00931, 12.0,     14.00307, 15.99491, 18.9984,
  19.99244, 22.9898,   23.98504, 26.98154, 27.97693, 30.97376, 31.97207, 34.96885, 39.9624,
  38.96371, 39.96259,  44.95592, 45.948,   50.9440,  51.9405,  54.9381,  55.9349,  58.9332,
  57.9353,  62.9298,   63.9291,  68.9257,  73.9219,  74.9216,  78.9183,  79.9165,  83.912,
  84.9117,  87.9056,   88.9054,  89.9043,  92.9060,  97.9055,  97.9072,  101.9037, 102.9048,
  105.9032, 106.90509, 113.9036, 114.9041, 117.9018, 120.9038, 129.9067, 126.9004, 131.9042,
  132.9051, 137.9050,  138.9061, 139.9053, 140.9074, 143.9099, 144.9128, 151.9195, 152.9209,
  157.9241, 159.9250,  163.9288, 164.9303, 165.9304, 168.9344, 173.9390, 174.9409, 179.9468,
  180.948,  183.9510,  186.9560, 189.9586, 192.9633, 194.9648, 196.9666, 201.9706, 204.9745,
  207.9766, 208.9804,  209.9829, 210.9875, 222.0175, 223.0198, 226.0254, 227.0278, 232.0382,
  231.0359, 238.0508,  237.0482, 244.0642, 243.0614, 247.0704, 247.0703, 251.0796, 252.0829,
  257.0950, 258.0986,  259.1009, 262.1100, 261.1087, 262.1138, 266.1219, 262.1229, 267.1318,
  268.1388, 281.0000,  280.0000, 285.0000, 284.0000, 289.0000, 287.0000, 292.0000, 293.0000,
  294.0000, 0.0,       0.0};

// bohr
std::vector<double> atom_radii = {
  0.604712316123456,  0.699198615517746,  2.45664378425154,   1.776342428612652,
  1.455089010672066,  1.41729449091435,   1.341705451398918,  1.209424632246912,
  1.209424632246912,  1.171630112489196,  3.02356158061728,   2.6456163830401196,
  2.343260224978392,  2.1542876261898116, 2.059801326795522,  1.965315027401232,
  1.8897259878858,    1.9086232477646579, 3.7794519757716,    3.288123218921292,
  3.2125341794058597, 3.02356158061728,   2.891280761465274,  2.6267191231612617,
  3.042458840496138,  2.872383501586416,  2.8345889818287,    2.343260224978392,
  2.494438304009256,  2.305465705220676,  2.324362965099534,  2.2676711854629596,
  2.2676711854629596, 2.229876665705244,  2.210979405826386,  2.192082145947528,
  4.062910873954469,  3.5904793769830197, 3.5904793769830197, 3.30702047880015,
  3.0991506201327117, 2.910178021344132,  2.777897202192126,  2.7589999423132676,
  2.683410902797836,  2.6267191231612617, 2.7401026824344097, 2.7212054225555518,
  2.683410902797836,  2.6456163830401196, 2.6456163830401196, 2.5889246034035462,
  2.570027343524688,  2.570027343524688,  4.610931410441352,  4.062910873954469,
  3.9117327949236054, 3.855041015287032,  3.8361437554081737, 3.7983492356504573,
  3.760554715892742,  3.741657456013884,  3.741657456013884,  3.7038629362561677,
  3.666068416498452,  3.6282738967407355, 3.6282738967407355, 3.5715821171041617,
  3.5904793769830197, 3.533787597346446,  3.533787597346446,  3.30702047880015,
  3.2125341794058597, 3.0613561003749963, 2.8534862417075577, 2.7212054225555518,
  2.664513642918978,  2.570027343524688,  2.570027343524688,  2.494438304009256,
  2.7401026824344097, 2.7589999423132676, 2.796794462070984,  2.6456163830401196,
  2.8345889818287,    2.8345889818287};

// prints atomic coordinates
void print_geometry(ExecutionContext& ec, ChemEnv& chem_env) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(70, '-') << std::endl;
    int i = 1;
    for(auto ecatom: chem_env.ec_atoms) {
      ss << std::setw(6) << std::left << i++ << std::setw(4) << ecatom.esymbol << std::right
         << std::fixed << std::setprecision(10) << std::setw(16) << ecatom.atom.x << std::setw(16)
         << ecatom.atom.y << std::setw(16) << ecatom.atom.z << std::endl;
    }
    std::cout << ss.str();
  }
}

void print_bond_lengths(ExecutionContext& ec, ChemEnv& chem_env,
                        std::vector<std::vector<double>>& bonds, int& num_atoms) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(60, '-') << std::endl;
    ss << std::setw(25) << "Bond Lengths" << std::endl << std::endl;

    ss << std::setw(3) << std::left << "i"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
       << " " << std::right << std::setw(20) << std::setw(3) << std::left << "Length (Angstroms)"
       << " " << std::right << std::setw(20) << std::setw(3) << std::left << "Length (Bohr)"
       << " " << std::right << std::setw(20) << std::endl;

    for(int i = 0; i < num_atoms; i++) {
      for(int j = i; j < num_atoms; j++) {
        if(i != j && bonds[j][i] != 0.0) {
          ss << std::setw(3) << std::left << i << " " << std::right << std::setw(14) << std::setw(3)
             << std::left << j << " " << std::right << std::setw(12) << std::fixed
             << std::setprecision(10) << bonds[i][j] * bohr_to_ang << " " << std::right
             << std::setw(18) << std::fixed << std::setprecision(10) << bonds[j][i] << " "
             << std::right << std::setw(12) << std::endl;
        }
      }
    }
    std::cout << ss.str();
  }
}

// unused for now
void print_oop_angles(ExecutionContext& ec, ChemEnv& chem_env,
                      std::vector<std::vector<std::vector<std::vector<double>>>>& oop_angles,
                      int&                                                        num_atoms) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(60, '-') << std::endl;
    ss << std::setw(20) << "Out-of-plane Angles" << std::endl << std::endl;

    ss << std::setw(3) << std::left << "i"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "l"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "Angle (degrees)"
       << " " << std::right << std::setw(14) << std::endl;

    for(int i = 0; i < num_atoms; i++) {
      for(int j = 0; j < i; j++) {
        for(int k = 0; k < j; k++) {
          for(int l = 0; l < k; l++) {
            if(i != j && i != k && i != l && j != k && j != l && k != l) {
              ss << std::setw(3) << std::left << i << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << j << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << k << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << l << " " << std::right << std::setw(14)
                 << std::fixed << std::setprecision(10) << oop_angles[i][j][k][l] * 180 / acos(-1.0)
                 << " " << std::right << std::setw(14) << std::endl; // j center, i0, j1
            }
          }
        }
      }
    }
    std::cout << ss.str();
  }
}

void print_torsional_angles(
  ExecutionContext& ec, ChemEnv& chem_env,
  std::vector<std::vector<std::vector<std::vector<double>>>>& torsional_angles, int& num_atoms) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(60, '-') << std::endl;
    ss << std::setw(20) << "Torsional Angles" << std::endl << std::endl;

    ss << std::setw(3) << std::left << "i"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "l"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "Angle (degrees)"
       << " " << std::right << std::setw(14) << std::endl;

    for(int i = 0; i < num_atoms; i++) {
      for(int j = i; j < num_atoms; j++) {
        for(int k = j; k < num_atoms; k++) {
          for(int l = k; l < num_atoms; l++) {
            if(i != j && i != k && i != l && j != k && j != l && k != l &&
               torsional_angles[i][j][k][l] != 0.0) {
              ss << std::setw(3) << std::left << i << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << j << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << k << " " << std::right << std::setw(14)
                 << std::setw(3) << std::left << l << " " << std::right << std::setw(14)
                 << std::fixed << std::setprecision(10)
                 << torsional_angles[i][j][k][l] * 180 / acos(-1.0) << " " << std::right
                 << std::setw(14) << std::endl; // j center, i0, j1
            }
          }
        }
      }
    }
    std::cout << ss.str();
  }
}

void print_com(ExecutionContext& ec, std::vector<double>& com) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(70, '-') << std::endl;
    ss << std::setw(40) << "Center of Mass" << std::endl << std::endl;
    ss << std::right << std::setw(10) << "x" << std::setw(16) << "y" << std::setw(16) << "z"
       << std::right << std::setw(18) << "Unit" << std::endl;
    ss << std::right << std::setprecision(10) << std::setw(16) << com[0] * bohr_to_ang
       << std::setw(16) << com[1] * bohr_to_ang << std::setw(16) << com[2] * bohr_to_ang
       << std::setw(14) << "Angstrom" << std::endl;
    ss << std::right << std::setprecision(10) << std::setw(16) << com[0] << std::setw(16) << com[1]
       << std::setw(16) << com[2] << std::setw(14) << "Bohr" << std::endl;
    std::cout << ss.str();
  }
}

void print_mot(ExecutionContext& ec, const std::vector<std::vector<double>>& mot) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(70, '-') << std::endl;
    ss << std::setw(40) << "Moments of Inertia Tensor" << std::endl << std::endl;

    for(int i = 0; i < 3; i++) {
      ss << std::setw(18) << std::right << mot[i][0] << std::setw(18) << mot[i][1] << std::setw(18)
         << mot[i][2] << std::endl;
    }
    std::cout << ss.str();
  }
}

void print_pmots(ExecutionContext& ec, std::vector<double>& pmots, const int& num_atoms) {
  if(ec.print()) {
    std::stringstream ss;
    ss << std::endl << std::string(70, '-') << std::endl;
    ss << std::setw(50) << "Principle Moments of Inertia" << std::endl << std::endl;
    ss << std::right << std::setw(12) << "x" << std::setw(18) << "y" << std::setw(18) << "z"
       << std::setw(20) << "Unit" << std::endl;
    ss << std::right << std::setw(18) << std::setprecision(10)
       << pmots[0] * bohr_to_ang * bohr_to_ang << std::setw(18)
       << pmots[1] * bohr_to_ang * bohr_to_ang << std::setw(18)
       << pmots[2] * bohr_to_ang * bohr_to_ang << std::setw(18) << "amu * ang^2" << std::endl;
    ss << std::right << std::setw(18) << std::setprecision(10) << pmots[0] << std::setw(18)
       << pmots[1] << std::setw(18) << pmots[2] << std::setw(18) << "amu * bohr^2" << std::endl;
    ss << std::right << std::setw(18) << std::scientific << std::setprecision(10)
       << pmots[0] * 1.6605402E-24 * 0.529177249E-8 * 0.529177249E-8 << std::setw(18)
       << pmots[1] * 1.6605402E-24 * 0.529177249E-8 * 0.529177249E-8 << std::setw(18)
       << pmots[2] * 1.6605402E-24 * 0.529177249E-8 * 0.529177249E-8 << std::setw(18) << "g * cm^2"
       << std::endl;
    ss << std::endl;

    ss << " - ";
    if(num_atoms == 2) ss << "Molecule is diatomic." << std::endl;
    else if(pmots[0] < 1e-4) ss << "Molecule is linear." << std::endl;
    else if((fabs(pmots[0] - pmots[1]) < 1e-4) && (fabs(pmots[1] - pmots[2]) < 1e-4))
      ss << "Molecule is a spherical top." << std::endl;
    else if((fabs(pmots[0] - pmots[1]) < 1e-4) && (fabs(pmots[1] - pmots[2]) > 1e-4))
      ss << "Molecule is an oblate symmetric top." << std::endl;
    else if((fabs(pmots[0] - pmots[1]) > 1e-4) && (fabs(pmots[1] - pmots[2]) < 1e-4))
      ss << "Molecule is a prolate symmetric top." << std::endl;
    else ss << "Molecule is an asymmetric top." << std::endl;

    double _pi  = acos(-1.0);
    double conv = 6.6260755E-34 / (8.0 * _pi * _pi);
    conv /= 1.6605402E-27 * 0.529177249E-10 * 0.529177249E-10;
    conv *= 1e-6;
    ss << std::endl << std::string(70, '-') << std::endl;
    ss << std::endl << std::setw(35) << "Rotational constants (MHz)" << std::endl << std::endl;
    ss << std::setprecision(4) << std::setw(8) << "A = " << conv / pmots[0] << std::setw(12)
       << "B = " << conv / pmots[1] << std::setw(12) << "C = " << conv / pmots[2] << std::endl;

    conv = 6.6260755E-34 / (8.0 * _pi * _pi);
    conv /= 1.6605402E-27 * 0.529177249E-10 * 0.529177249E-10;
    conv /= 2.99792458E10;
    ss << std::endl << std::string(70, '-') << std::endl;
    ss << std::endl << std::setw(35) << "Rotational constants (cm-1)" << std::endl << std::endl;
    ss << std::setprecision(4) << std::setw(8) << "A = " << conv / pmots[0] << std::setw(12)
       << "B = " << conv / pmots[1] << std::setw(12) << "C = " << conv / pmots[2] << std::endl;

    std::cout << ss.str();
  }
}

std::vector<std::vector<double>> process_geometry(ExecutionContext& ec, ChemEnv& chem_env) {
  std::vector<std::vector<double>> data_mat;

  // for each atom
  for(const auto& ec_atom: chem_env.ec_atoms) {
    std::vector<double> temp_data(4);
    temp_data[0] = ec_atom.atom.atomic_number;
    temp_data[1] = ec_atom.atom.x; // always is initially in bohr
    temp_data[2] = ec_atom.atom.y; // thus all calcs are natively in bohr
    temp_data[3] = ec_atom.atom.z;
    data_mat.push_back(temp_data);
  }

  return data_mat;
}

double single_bond_length(ExecutionContext& ec, int num_atoms,
                          std::vector<std::vector<double>>& data_mat, int i, int j) {
  double sqx          = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
  double sqy          = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
  double sqz          = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
  double length       = sqrt(sqx + sqy + sqz);
  double atom0_radius = atom_radii[data_mat[i][0] - 1];
  double atom1_radius = atom_radii[data_mat[j][0] - 1];
  double sum          = atom0_radius + atom1_radius;
  if(length < 1.3 * sum) { return length; }
  else { return 0.0; }
}

double single_bond_length_optimize(ExecutionContext& ec, int num_atoms,
                                   std::vector<std::vector<double>>& data_mat, int i, int j,
                                   double threshold, std::vector<double> atom_radii_arg) {
  double sqx          = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
  double sqy          = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
  double sqz          = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
  double length       = sqrt(sqx + sqy + sqz);
  double atom0_radius = atom_radii_arg[data_mat[i][0] - 1];
  double atom1_radius = atom_radii_arg[data_mat[j][0] - 1];
  double sum          = atom0_radius + atom1_radius;
  if(length < threshold * sum) { return length; }
  else { return 0.0; }
}

std::vector<std::vector<double>> process_bond_lengths(ExecutionContext& ec, int num_atoms,
                                                      std::vector<std::vector<double>>& data_mat) {
  std::vector<std::vector<double>> bond_lengths(num_atoms, std::vector<double>(num_atoms));

  for(int i = 0; i < num_atoms; i++) {
    for(int j = i; j < num_atoms; j++) {
      if(i != j) {
        double sqx          = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
        double sqy          = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
        double sqz          = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
        double length       = sqrt(sqx + sqy + sqz);
        double atom0_radius = atom_radii[data_mat[i][0] - 1];
        double atom1_radius = atom_radii[data_mat[j][0] - 1];

        double sum = atom0_radius + atom1_radius;
        if(length < 1.3 * sum) {
          bond_lengths[i][j] = length;
          bond_lengths[j][i] = length;
        }
        else {
          bond_lengths[i][j] = 0.0;
          bond_lengths[j][i] = 0.0;
        }
      }
    }
  }

  // loop to add bond lengths for fragmented atoms
  for(int i = 0; i < num_atoms; i++) {
    bool bonded = false;
    for(int j = 0; j < num_atoms; j++) {
      if(bond_lengths[i][j] != 0.0) { bonded = true; }
    }
    // if atom i is not bonded to any other atom
    if(!bonded) {
      double bond_threshold = 0.1;
      while(!bonded) {
        for(int j = 0; j < num_atoms; j++) {
          if(i != j) {
            double sqx    = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
            double sqy    = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
            double sqz    = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
            double length = sqrt(sqx + sqy + sqz);
            double atom0_radius = atom_radii[data_mat[i][0] - 1];
            double atom1_radius = atom_radii[data_mat[j][0] - 1];
            double sum          = atom0_radius + atom1_radius;
            if(length < (1.3 + bond_threshold) * sum) {
              bond_lengths[i][j] = length;
              bond_lengths[j][i] = length;
              bonded             = true;
            }
            else {
              bond_lengths[i][j] = 0.0;
              bond_lengths[j][i] = 0.0;
            }
          }
        }
        bond_threshold += 0.1;
      }
    }
  }

  return bond_lengths;
}

std::vector<std::vector<double>> process_bond_lengths(ExecutionContext& ec, int num_atoms,
                                                      std::vector<std::vector<double>>& data_mat,
                                                      std::vector<double> atom_radii_arg) {
  std::vector<std::vector<double>> bond_lengths(num_atoms, std::vector<double>(num_atoms));

  for(int i = 0; i < num_atoms; i++) {
    for(int j = i; j < num_atoms; j++) {
      if(i != j) {
        double sqx          = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
        double sqy          = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
        double sqz          = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
        double length       = sqrt(sqx + sqy + sqz);
        double atom0_radius = atom_radii_arg[data_mat[i][0] - 1];
        double atom1_radius = atom_radii_arg[data_mat[j][0] - 1];

        double sum = atom0_radius + atom1_radius;
        if(length < 1.3 * sum) {
          bond_lengths[i][j] = length;
          bond_lengths[j][i] = length;
        }
        else {
          bond_lengths[i][j] = 0.0;
          bond_lengths[j][i] = 0.0;
        }
      }
    }
  }

  // loop to add bond lengths for fragmented atoms
  for(int i = 0; i < num_atoms; i++) {
    bool bonded = false;
    for(int j = 0; j < num_atoms; j++) {
      if(bond_lengths[i][j] != 0.0) { bonded = true; }
    }
    // if atom i is not bonded to any other atom
    if(!bonded) {
      double bond_threshold = 0.1;
      while(!bonded) {
        for(int j = 0; j < num_atoms; j++) {
          if(i != j) {
            double sqx    = (data_mat[i][1] - data_mat[j][1]) * (data_mat[i][1] - data_mat[j][1]);
            double sqy    = (data_mat[i][2] - data_mat[j][2]) * (data_mat[i][2] - data_mat[j][2]);
            double sqz    = (data_mat[i][3] - data_mat[j][3]) * (data_mat[i][3] - data_mat[j][3]);
            double length = sqrt(sqx + sqy + sqz);
            double atom0_radius = atom_radii_arg[data_mat[i][0] - 1];
            double atom1_radius = atom_radii_arg[data_mat[j][0] - 1];
            double sum          = atom0_radius + atom1_radius;
            if(length < (1.3 + bond_threshold) * sum) {
              bond_lengths[i][j] = length;
              bond_lengths[j][i] = length;
              bonded             = true;
            }
            else {
              bond_lengths[i][j] = 0.0;
              bond_lengths[j][i] = 0.0;
            }
          }
        }
        bond_threshold += 0.1;
      }
    }
  }

  return bond_lengths;
}

// function to calculate the unit vectors for a pair of atoms
auto atom_pair_unit_vector(std::vector<std::vector<double>>& data, int& idx0, int& idx1,
                           std::vector<std::vector<double>>& bonds, int& num_atoms) {
  std::vector<double> vect(3);

  vect[0]    = -(data[idx0][1] - data[idx1][1]); // x displacement
  vect[1]    = -(data[idx0][2] - data[idx1][2]);
  vect[2]    = -(data[idx0][3] - data[idx1][3]);
  double sq0 = vect[0] * vect[0];
  double sq1 = vect[1] * vect[1];
  double sq2 = vect[2] * vect[2];
  double n   = sqrt(sq0 + sq1 + sq2);
  vect[0] /= n;
  vect[1] /= n;
  vect[2] /= n;

  return vect;
}

// function to precalculate all possible unit vectors for all pairs of atoms
std::vector<std::vector<std::vector<double>>>
calculate_atom_pair_unit_vector(std::vector<std::vector<double>>& data,
                                std::vector<std::vector<double>>& bonds, int& num_atoms) {
  std::vector<std::vector<std::vector<double>>> atom_pair_unit_vectors(
    num_atoms, std::vector<std::vector<double>>(num_atoms, std::vector<double>(3, 0)));

  // not a symmetric matrix
  for(int i = 0; i < num_atoms; i++) {
    for(int j = 0; j < num_atoms; j++) {
      auto vect                    = atom_pair_unit_vector(data, i, j, bonds, num_atoms);
      atom_pair_unit_vectors[i][j] = vect;
    }
  }

  return atom_pair_unit_vectors;
}

double dot_product(std::vector<double>& atom0, std::vector<double>& atom1) {
  return atom0[0] * atom1[0] + atom0[1] * atom1[1] + atom0[2] * atom1[2];
}

double specific_bond_angle(ExecutionContext& ec, const int& num_atoms,
                           const std::vector<std::vector<double>>&              bonds,
                           const std::vector<std::vector<std::vector<double>>>& apuv, const int& i,
                           const int& j, const int& k) {
  std::vector<double> atom0 = apuv[j][i];
  std::vector<double> atom1 = apuv[j][k];
  double              angle = acos(atom0[0] * atom1[0] + atom0[1] * atom1[1] + atom0[2] * atom1[2]);

  return angle;
}

std::vector<double> cross_product(std::vector<double>& atom0, std::vector<double>& atom1) {
  std::vector<double> product(3);
  product[0] = atom0[1] * atom1[2] - atom0[2] * atom1[1];
  product[1] = -(atom0[0] * atom1[2] - atom0[2] * atom1[0]);
  product[2] = atom0[0] * atom1[1] - atom0[1] * atom1[0];
  return product;
}

// auto out_of_plane_angles(ExecutionContext& ec, std::vector<std::vector<double>>& data,
//                          int& num_atoms, std::vector<std::vector<double>>& bonds,
//                          std::vector<std::vector<std::vector<double>>>& angles,
//                          std::vector<std::vector<std::vector<double>>>& apuv) {
//   std::vector<std::vector<std::vector<std::vector<double>>>> oop_angles(
//     num_atoms,
//     std::vector<std::vector<std::vector<double>>>(
//       num_atoms, std::vector<std::vector<double>>(
//                    num_atoms, std::vector<double>(num_atoms, 0)))); // hmm ill need to fix these

//   for(int i = 0; i < num_atoms; i++) {
//     for(int j = 0; j < i; j++) {
//       for(int k = 0; k < j; k++) {
//         for(int l = 0; l < k; l++) {
//           if(i != j && i != k && i != l && j != k && j != l && k != l) {
//             auto                atom0 = apuv[k][j];
//             auto                atom1 = apuv[k][l];
//             auto                cross = cross_product(atom0, atom1);
//             std::vector<double> angle(3);
//             auto                atom2 = apuv[k][i];                                 // k i
//             angle[0]                  = cross[0] * atom2[0] / sin(angles[j][k][l]); // jkl
//             angle[1]                  = cross[1] * atom2[1] / sin(angles[j][k][l]);
//             angle[2]                  = cross[2] * atom2[2] / sin(angles[j][k][l]);
//             auto resultant            = asin(angle[0] + angle[1] + angle[2]);
//             oop_angles[i][j][k][l]    = resultant;
//           }
//         }
//       }
//     }
//   }

//   return oop_angles;
// }

auto single_apuv(const std::vector<std::vector<double>>& data,
                 const std::vector<std::vector<double>>& bonds, const int& num_atoms, const int& i,
                 const int& j) {
  std::vector<double> vect(3);

  vect[0]    = -(data[i][1] - data[j][1]); // x displacement
  vect[1]    = -(data[i][2] - data[j][2]);
  vect[2]    = -(data[i][3] - data[j][3]);
  double sq0 = vect[0] * vect[0];
  double sq1 = vect[1] * vect[1];
  double sq2 = vect[2] * vect[2];
  double n   = sqrt(sq0 + sq1 + sq2);
  vect[0] /= n;
  vect[1] /= n;
  vect[2] /= n;

  return vect;
}

auto specific_bond_angles(const ExecutionContext& ec, const int& num_atoms,
                          const std::vector<std::vector<double>>& data,
                          const std::vector<std::vector<double>>& bonds, const int& i, const int& j,
                          const int& k) {
  std::vector<double> atom0 = single_apuv(data, bonds, num_atoms, j, i); // ji
  std::vector<double> atom1 = single_apuv(data, bonds, num_atoms, j, k); // jk
  double              angle = acos(atom0[0] * atom1[0] + atom0[1] * atom1[1] + atom0[2] * atom1[2]);

  return angle;
}

auto bond_angle(const ExecutionContext& ec, const int& num_atoms,
                const std::vector<std::vector<double>>& data,
                const std::vector<std::vector<double>>& bonds, const int& i, const int& j,
                const int& k) {
  std::vector<double> atom0 = single_apuv(data, bonds, num_atoms, j, i); // ji
  std::vector<double> atom1 = single_apuv(data, bonds, num_atoms, j, k); // jk
  double              angle = acos(atom0[0] * atom1[0] + atom0[1] * atom1[1] + atom0[2] * atom1[2]);

  return angle;
}

auto single_torsional_angle(const ExecutionContext&                 ec,
                            const std::vector<std::vector<double>>& data, const int& num_atoms,
                            const std::vector<std::vector<double>>& bonds, const int& i,
                            const int& j, const int& k, const int& l) {
  std::vector<double> e_ij      = single_apuv(data, bonds, num_atoms, i, j); // ij
  std::vector<double> e_jk      = single_apuv(data, bonds, num_atoms, j, k); // jk
  std::vector<double> e_kl      = single_apuv(data, bonds, num_atoms, k, l); // kl
  auto                cproduct0 = cross_product(e_ij, e_jk);
  auto                cproduct1 = cross_product(e_jk, e_kl);
  double              numerator = dot_product(cproduct0, cproduct1);
  double              divisor0  = sin(specific_bond_angles(ec, num_atoms, data, bonds, i, j, k));
  double              divisor1  = sin(specific_bond_angles(ec, num_atoms, data, bonds, j, k, l));

  double divisor   = divisor0 * divisor1;
  double cos_value = numerator / divisor;
  double value;
  if(cos_value < -1.0) value = acos(-1.0);
  else if(cos_value > 1.0) value = acos(1);
  else value = acos(cos_value);

  // Compute the sign of the torsion

  double cross_x = cproduct0[1] * cproduct1[2] - cproduct0[2] * cproduct1[1];
  double cross_y = cproduct0[2] * cproduct1[0] - cproduct0[0] * cproduct1[2];
  double cross_z = cproduct0[0] * cproduct1[1] - cproduct0[1] * cproduct1[0];
  double norm    = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
  cross_x /= norm;
  cross_y /= norm;
  cross_z /= norm;
  double sign = 1.0;
  double dot  = cross_x * e_jk[0] + cross_y * e_jk[1] + cross_z * e_jk[2];
  if(dot < 0.0) sign = -1.0;

  return sign * value * 180 / acos(-1.0);
}

void print_bond_angles(ExecutionContext& ec, ChemEnv& chem_env, int& num_atoms,
                       std::vector<std::vector<double>>&                   data,
                       const std::vector<std::vector<double>>&             bonds,
                       const std::vector<std::vector<std::vector<double>>> apuv) {
  if(ec.print()) {
    std::stringstream ss;

    ss << std::endl << std::string(60, '-') << std::endl;
    ss << std::setw(18) << "Bond Angles" << std::endl << std::endl;

    ss << std::setw(3) << std::left << "i"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
       << " " << std::right << std::setw(13) << std::setw(3) << std::left << "Angle (degrees)"
       << " " << std::right << std::setw(14) << std::endl;

    int num_bonds = 0;

    for(int i = 0; i < num_atoms; i++) {     // center atom
      for(int j = 0; j < num_atoms; j++) {   // atom 1
        for(int k = i; k < num_atoms; k++) { // atom 2
          if(i != j && i != k && j != k && bonds[i][j] != 0.0 && bonds[j][k] != 0.0 &&
             specific_bond_angles(ec, num_atoms, data, bonds, i, j, k) != 0.0) {
            ss << std::setw(3) << std::left << i << " " << std::right << std::setw(14)
               << std::setw(3) << std::left << j << " " << std::right << std::setw(14)
               << std::setw(3) << std::left << k << " " << std::left << std::setw(14) << std::fixed
               << std::setprecision(10)
               << specific_bond_angles(ec, num_atoms, data, bonds, i, j, k) * 180 / acos(-1.0)
               << " " << std::right << std::setw(14) << std::endl; // j center, i0, j1
            num_bonds++;
          }
        }
      }
    }
    ss << std::endl << " - Number of angles printed: " << num_bonds << std::endl;
    std::cout << ss.str();
  }
}

// this function computs the angle then prints it
auto torsional_angle(ExecutionContext& ec, const std::vector<std::vector<double>>& data,
                     const int& num_atoms, const std::vector<std::vector<double>>& bonds,
                     const std::vector<std::vector<std::vector<double>>>& apuv) {
  std::stringstream ss;
  if(ec.print()) {
    ss << std::endl << std::string(60, '-') << std::endl;
    ss << std::setw(20) << "Torsional Angles" << std::endl << std::endl;

    ss << std::setw(3) << std::left << "i"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
       << " " << std::right << std::setw(14) << std::setw(3) << std::left << "l"
       << " " << std::right << std::setw(13) << std::setw(3) << std::left << "Angle (degrees)"
       << " " << std::right << std::setw(14) << std::endl;

    int printed = 0;

    std::set<std::vector<int>> used_indices;

    for(int i = 0; i < num_atoms; i++) {
      for(int j = 0; j < num_atoms; j++) {
        if(i != j && bonds[i][j] != 0) {
          for(int k = 0; k < num_atoms; k++) {
            if(i != k && j != k && bonds[j][k] != 0.0 && bonds[i][j] != 0.0 && bonds[j][k] != 0.0 &&
               specific_bond_angles(ec, num_atoms, data, bonds, i, j, k) != acos(-1.0)) {
              for(int l = 0; l < num_atoms; l++) {
                std::vector<int> current = {i, j, k, l};
                used_indices.insert(current);
                std::vector<int> reverse = {l, k, j, i};

                if(i != l && j != l && k != l && bonds[k][l] != 0.0 &&
                   specific_bond_angles(ec, num_atoms, data, bonds, j, k, l) != acos(-1.0) &&
                   used_indices.find(reverse) != used_indices.end()) {
                  std::vector<double> e_ij      = apuv[i][j];
                  std::vector<double> e_jk      = apuv[j][k];
                  std::vector<double> e_kl      = apuv[k][l];
                  auto                cproduct0 = cross_product(e_ij, e_jk);
                  auto                cproduct1 = cross_product(e_jk, e_kl);
                  double              numerator = dot_product(cproduct0, cproduct1);
                  double divisor0 = sin(specific_bond_angles(ec, num_atoms, data, bonds, i, j, k));
                  double divisor1 = sin(specific_bond_angles(ec, num_atoms, data, bonds, j, k, l));

                  double divisor   = divisor0 * divisor1;
                  double cos_value = numerator / divisor;
                  double value;
                  if(cos_value < -1.0) value = acos(-1.0);
                  else if(cos_value > 1.0) value = acos(1);
                  else value = acos(cos_value);

                  double cross_x = cproduct0[1] * cproduct1[2] - cproduct0[2] * cproduct1[1];
                  double cross_y = cproduct0[2] * cproduct1[0] - cproduct0[0] * cproduct1[2];
                  double cross_z = cproduct0[0] * cproduct1[1] - cproduct0[1] * cproduct1[0];
                  double norm    = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
                  cross_x /= norm;
                  cross_y /= norm;
                  cross_z /= norm;
                  double sign = 1.0;
                  double dot  = cross_x * e_jk[0] + cross_y * e_jk[1] + cross_z * e_jk[2];
                  if(dot < 0.0) sign = -1.0;

                  double current_torsional_angle = value * sign;

                  if(current_torsional_angle != 0.0 && current_torsional_angle != acos(-1.0)) {
                    ss << std::setw(3) << std::left << i << " " << std::right << std::setw(14)
                       << std::setw(3) << std::left << j << " " << std::right << std::setw(14)
                       << std::setw(3) << std::left << k << " " << std::right << std::setw(14)
                       << std::setw(3) << std::left << l << " " << std::left << std::setw(14)
                       << std::fixed << std::setprecision(10)
                       << current_torsional_angle * 180 / acos(-1.0) << " " << std::right
                       << std::setw(14) << std::endl; // j center, i0, j1
                    printed++;
                  }
                }
              }
            }
          }
        }
      }
    }

    if(ec.print()) {
      ss << std::endl << " - Number of torsional angles printed: " << printed << std::endl;
      std::cout << ss.str();
    }
  }
}

std::vector<double> center_of_mass(ExecutionContext&                       ec,
                                   const std::vector<std::vector<double>>& data,
                                   const int&                              num_atoms) {
  double den   = 0;
  double x_num = 0;
  double y_num = 0;
  double z_num = 0;
  for(int i = 0; i < num_atoms; i++) {
    x_num += atom_mass[data[i][0] - 1] * data[i][1];
    y_num += atom_mass[data[i][0] - 1] * data[i][2];
    z_num += atom_mass[data[i][0] - 1] * data[i][3];
    den += atom_mass[data[i][0] - 1];
  }

  std::vector<double> com = {x_num / den, y_num / den, z_num / den};
  return com;
}

std::vector<std::vector<double>>
moment_of_inertia(ExecutionContext& ec, const std::vector<std::vector<double>>& base_data,
                  const int& num_atoms, const std::vector<double>& com) {
  std::vector<std::vector<double>> mot(3, std::vector<double>(3, 0));
  std::vector<std::vector<double>> data = base_data;

  for(int i = 0; i < num_atoms; i++) {
    data[i][1] -= com[0];
    data[i][2] -= com[1];
    data[i][3] -= com[2];
  }

  for(int i = 0; i < num_atoms; i++) {
    mot[0][0] += atom_mass[data[i][0] - 1] * (data[i][2] * data[i][2] + data[i][3] * data[i][3]);
    mot[1][1] += atom_mass[data[i][0] - 1] * (data[i][1] * data[i][1] + data[i][3] * data[i][3]);
    mot[2][2] += atom_mass[data[i][0] - 1] * (data[i][1] * data[i][1] + data[i][2] * data[i][2]);
    mot[0][1] -= atom_mass[data[i][0] - 1] * data[i][1] * data[i][2];
    mot[0][2] -= atom_mass[data[i][0] - 1] * data[i][1] * data[i][3];
    mot[1][2] -= atom_mass[data[i][0] - 1] * data[i][2] * data[i][3];
  }
  mot[1][0] = mot[0][1];
  mot[2][0] = mot[0][2];
  mot[2][1] = mot[1][2];

  return mot;
}

std::vector<double> principle_moments_of_inertia(ExecutionContext&                       ec,
                                                 const std::vector<std::vector<double>>& mot) {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
  Matrix                                                                         I(3, 3);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) { I(i, j) = mot[i][j]; }
  }

  Eigen::SelfAdjointEigenSolver<Matrix> solver(I);
  Matrix                                evals = solver.eigenvalues();
  std::vector<double>                   pmots = {evals(0), evals(1), evals(2)};
  return pmots;
}

double bond_length(const std::vector<double>& atom0, const std::vector<double>& atom1) {
  double sqx    = (atom0[1] - atom1[1]) * (atom0[1] - atom1[1]);
  double sqy    = (atom0[2] - atom1[2]) * (atom0[2] - atom1[2]);
  double sqz    = (atom0[3] - atom1[3]) * (atom0[3] - atom1[3]);
  double length = sqrt(sqx + sqy + sqz);

  return length;
}

auto z_matrix(ExecutionContext& ec, ChemEnv& chem_env) {
  std::stringstream ss;
  auto              atoms     = process_geometry(ec, chem_env);
  int               num_atoms = atoms.size();
  auto              bonds     = process_bond_lengths(ec, num_atoms, atoms);

  std::vector<int>  zmat_idx;
  std::vector<int>  frag_idx;
  std::vector<int>  connectivity;
  std::vector<bool> included_atoms(num_atoms, false);

  // adding the initial atom
  zmat_idx.push_back(0);
  connectivity.push_back(0);
  included_atoms[0] = true;

  // internal switch between printed zmatrix and geometry optimization matrix
  // if set to true, it will only show find connections between bonded atoms
  // if set to false, it will generate the matrix by iterating over all atoms
  bool true_bonded = false;
  if(true_bonded) {
    // while not all atoms have been included
    double increased_threshold       = 0;
    int    prev_it_included_atoms    = 0;
    int    current_it_included_atoms = 1;

    while(std::find(included_atoms.begin(), included_atoms.end(), false) != included_atoms.end()) {
      // for each atom that is not attached
      for(int i = 1; i < num_atoms; i++) {
        if(included_atoms[i] == false) {
          // for each atom that is attached
          for(int j = 0; j < num_atoms; j++) {
            if(included_atoms[j] == true) {
              double atom0_radius = atom_radii[atoms[i][0] - 1];
              double atom1_radius = atom_radii[atoms[j][0] - 1];
              double sum          = atom0_radius + atom1_radius;

              // if the atom that is not attached is bonded to an atom that is attached
              if(bond_length(atoms[i], atoms[j]) < (1.3 + increased_threshold) * sum) {
                zmat_idx.push_back(i);
                connectivity.push_back(j); // con[j] is bonded to [i]
                included_atoms[i] = true;
                break;
              }
            }
          }
        }
      }
      prev_it_included_atoms    = current_it_included_atoms;
      current_it_included_atoms = std::count(included_atoms.begin(), included_atoms.end(), true);
      if(prev_it_included_atoms == current_it_included_atoms) { increased_threshold += 0.1; }
      else { increased_threshold = 0.0; }
    }
  }
  else {
    for(int i = 1; i < num_atoms; i++) {
      zmat_idx.push_back(i);
      connectivity.push_back(i - 1);
    }
  }

  // next
  // build the zmatrix

  // shape num_atoms, [#, ref., stretch, ref., angle, ref., torsion] 7
  std::vector<std::vector<double>> zmatrix(num_atoms, std::vector<double>(7));

  if(ec.print()) {
    ss << std::endl << std::string(60, '-') << std::endl << std::endl;
    ss << std::setw(20) << "Z-Matrix" << std::endl;
  }

  // vector to store atomic numbers based on position
  std::vector<int> atomic_numbers;
  for(const auto& ec_atom: chem_env.ec_atoms) {
    atomic_numbers.push_back(ec_atom.atom.atomic_number);
  }

  for(int i = 0; i < num_atoms; i++) {
    if(i == 0) {
      if(ec.print()) { ss << std::fixed << chem_env.ec_atoms[i].esymbol << std::endl; }
      zmatrix[i][0] = atomic_numbers[i];
      zmatrix[i][1] = -1;
      zmatrix[i][2] = -1;
      zmatrix[i][3] = -1;
      zmatrix[i][4] = -1;
      zmatrix[i][5] = -1;
      zmatrix[i][6] = -1;
    }
    else if(i == 1) {
      zmatrix[i][0] = atomic_numbers[i];
      zmatrix[i][1] = connectivity[i];
      zmatrix[i][2] = bond_length(atoms[zmat_idx[i]], atoms[connectivity[i]]);
      zmatrix[i][3] = -1;
      zmatrix[i][4] = -1;
      zmatrix[i][5] = -1;
      zmatrix[i][6] = -1;

      if(ec.print()) {
        ss << std::fixed << chem_env.ec_atoms[i].esymbol << " " << connectivity[i] + 1 << " "
           << zmatrix[i][2] << std::endl;
      }
    }
    else if(i == 2) {
      int temp = 1;
      while(!(zmat_idx[i] != connectivity[i] && connectivity[i] != zmat_idx[i - temp] &&
              zmat_idx[i] != zmat_idx[i - temp])) {
        temp++;
      }

      zmatrix[i][0] = atomic_numbers[i];
      zmatrix[i][1] = connectivity[i];
      zmatrix[i][2] = bond_length(atoms[zmat_idx[i]], atoms[connectivity[i]]);
      zmatrix[i][3] = zmat_idx[i - temp];
      zmatrix[i][4] =
        bond_angle(ec, num_atoms, atoms, bonds, zmat_idx[i], connectivity[i], zmat_idx[i - temp]);
      // zmatrix[i][4] = bond_angle(ec, atoms, zmat_idx[i], zmat_idx[i-temp], connectivity[i]);
      zmatrix[i][5] = -1;
      zmatrix[i][6] = -1;

      if(ec.print()) {
        ss << std::fixed << chem_env.ec_atoms[i].esymbol << " " << connectivity[i] + 1 << " "
           << zmatrix[i][2] << " " << zmat_idx[i - temp] + 1 << " " << zmatrix[i][4] << std::endl;
      }
    }
    else {
      int temp = 1;
      while(!(zmat_idx[i] != connectivity[i] && connectivity[i] != zmat_idx[i - temp] &&
              zmat_idx[i] != zmat_idx[i - temp])) {
        temp++;
      }

      int temp2 = 1;
      while(!(connectivity[i] != zmat_idx[i - temp - temp2] &&
              zmat_idx[i] != zmat_idx[i - temp - temp2] &&
              zmat_idx[i - temp] != zmat_idx[i - temp - temp2] && temp != temp2)) {
        temp2++;
      }

      zmatrix[i][0] = atomic_numbers[i];
      zmatrix[i][1] = connectivity[i];
      zmatrix[i][2] = bond_length(atoms[zmat_idx[i]], atoms[connectivity[i]]);
      zmatrix[i][3] = zmat_idx[i - temp];
      zmatrix[i][4] = bond_angle(ec, num_atoms, atoms, bonds, zmat_idx[i], connectivity[i],
                                 zmat_idx[i - temp]); //  init, center, alt
      // zmatrix[i][4] = bond_angle(ec, atoms, zmat_idx[i], zmat_idx[i-temp], connectivity[i]);
      zmatrix[i][5] = zmat_idx[i - temp - temp2];
      // i suspect torsional angles are off
      // hmm how can i test it
      // angle ijk
      // torsion ijkl
      zmatrix[i][6] = single_torsional_angle(ec, atoms, num_atoms, bonds, zmat_idx[i],
                                             connectivity[i], zmat_idx[i - temp],
                                             zmat_idx[i - temp - temp2]);

      if(ec.print()) {
        ss << std::fixed << chem_env.ec_atoms[i].esymbol << " " << connectivity[i] + 1 << " "
           << zmatrix[i][2] << " " << zmat_idx[i - temp] + 1 << " " << zmatrix[i][4] << " "
           << zmat_idx[i - temp - temp2] + 1 << " " << zmatrix[i][6] << std::endl;
      }
    }
  }

  if(ec.print()) { std::cout << ss.str(); }

  return zmatrix;
}

Eigen::Vector3d nerf(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                     double l, double theta, double chi) {
  // Calculate unit vectors AB and BC
  Eigen::Vector3d ab_unit = (b - a).normalized();
  Eigen::Vector3d bc_unit = (c - b).normalized();

  // Calculate unit normals n = AB x BC and p = n x BC
  Eigen::Vector3d n_unit = ab_unit.cross(bc_unit).normalized();
  Eigen::Vector3d p_unit = n_unit.cross(bc_unit);

  // Create rotation matrix [BC; p; n] (3x3)
  Eigen::Matrix3d M;
  M.col(0) = bc_unit;
  M.col(1) = p_unit;
  M.col(2) = n_unit;

  // Convert degrees to radians
  theta = theta * acos(-1.0) / 180;
  chi   = chi * acos(-1.0) / 180;

  // Calculate coord pre rotation matrix
  Eigen::Vector3d d2(-l * std::cos(theta), l * std::sin(theta) * std::cos(chi),
                     l * std::sin(theta) * std::sin(chi));

  // Apply rotation and return the final coordinate
  return c + M * d2;
}

auto cartesian_from_z_matrix(ExecutionContext& ec, const ChemEnv& chem_env,
                             const std::vector<std::vector<double>> zmatrix) {
  // takes a z matrix
  // computes cartesian coordinates from z-matrix
  std::vector<std::vector<double>> cartesian(zmatrix.size(),
                                             std::vector<double>(4)); // atomic number, x, y, z
  std::stringstream                ss;

  for(size_t i = 0; i < zmatrix.size(); i++) {
    // handling for first three atoms
    // setting first atom to origin
    if(i == 0) {
      cartesian[i][0] = zmatrix[i][0];
      cartesian[i][1] = 0.0; // setting position to origin
      cartesian[i][2] = 0.0;
      cartesian[i][3] = 0.0;
      // setting second atom to offset on z-axis
    }
    else if(i == 1) {
      cartesian[i][0] = zmatrix[i][0];
      cartesian[i][1] = 0.0; // setting x and y to zero
      cartesian[i][2] = 0.0;
      cartesian[i][3] = zmatrix[i][2]; // considering the first distance as being in the z direction
      // setting third atom based on angle between first and third atom
    }
    else if(i == 2) {
      cartesian[i][0] = zmatrix[i][0];
      cartesian[i][1] = 0.0; // setting x to zero

      double theta = zmatrix[i][4]; // angle

      cartesian[i][2] = zmatrix[i][2] * cos((theta - 90) * acos(-1.0) / 180);
      cartesian[i][3] = cartesian[1][3] + (zmatrix[i][2] * sin((theta - 90) * acos(-1.0) / 180));
      // computing remaining atoms using the nerf function
    }
    else {
      Eigen::Vector3d atom0(cartesian[i - 1][1], cartesian[i - 1][2], cartesian[i - 1][3]);
      Eigen::Vector3d atom1(cartesian[i - 2][1], cartesian[i - 2][2], cartesian[i - 2][3]);
      Eigen::Vector3d atom2(cartesian[i - 3][1], cartesian[i - 3][2], cartesian[i - 3][3]);
      auto eigen_coords = nerf(atom2, atom1, atom0, zmatrix[i][2], zmatrix[i][4], zmatrix[i][6]);
      cartesian[i][0]   = zmatrix[i][0];
      cartesian[i][1]   = eigen_coords[0];
      cartesian[i][2]   = eigen_coords[1];
      cartesian[i][3]   = eigen_coords[2];
    }
  }

  if(ec.print()) {
    ss << std::endl << std::string(60, '-') << std::endl << std::endl;
    ss << std::setw(20) << "Converted back to Cartesian" << std::endl;
    for(size_t i = 0; i < cartesian.size(); i++) {
      ss << std::setprecision(0) << cartesian[i][0] << std::setprecision(10) << " "
         << cartesian[i][1] << " " << cartesian[i][2] << " " << cartesian[i][3] << std::endl;
    }
    std::cout << ss.str();
  }
}

void geometry_analysis(ExecutionContext& ec, ChemEnv& chem_env) {
  auto data_mat     = process_geometry(ec, chem_env);
  int  num_atoms    = data_mat.size();
  auto bond_lengths = process_bond_lengths(ec, num_atoms, data_mat);
  auto apuv         = calculate_atom_pair_unit_vector(data_mat, bond_lengths, num_atoms);
  // auto oop_angles       = out_of_plane_angles(ec, data_mat, num_atoms, bond_lengths, angles,
  // apuv);
  auto com   = center_of_mass(ec, data_mat, num_atoms);
  auto mot   = moment_of_inertia(ec, data_mat, num_atoms, com);
  auto pmots = principle_moments_of_inertia(ec, mot);
  print_bond_lengths(ec, chem_env, bond_lengths, num_atoms);
  ec.pg().barrier();
  print_bond_angles(ec, chem_env, num_atoms, data_mat, bond_lengths, apuv);
  ec.pg().barrier();
  torsional_angle(ec, data_mat, num_atoms, bond_lengths, apuv);
  ec.pg().barrier();
  // print_oop_angles(ec, chem_env, oop_angles, num_atoms);
  print_com(ec, com);
  ec.pg().barrier();
  print_mot(ec, mot);
  ec.pg().barrier();
  print_pmots(ec, pmots, num_atoms);
  ec.pg().barrier();
  auto zmatrix = z_matrix(ec, chem_env);
  cartesian_from_z_matrix(ec, chem_env, zmatrix);

  std::cout.flags(std::ios::fmtflags());
  ec.pg().barrier();
}

} // namespace exachem::task
