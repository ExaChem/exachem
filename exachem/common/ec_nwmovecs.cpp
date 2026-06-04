/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/ec_nwmovecs.hpp"
#include "exachem/common/txt_utils.hpp"

bool ECNWChem::check_nwmovecs(std::string nwmovecsfile) {
  nwmovecs_exists = !nwmovecsfile.empty();
  if(nwmovecs_exists) {
    nwmovecs_file_valid = std::filesystem::exists(nwmovecsfile);
    if(!nwmovecs_file_valid)
      tamm_terminate("ERROR: nwmovecsfile provided: " + nwmovecsfile + " does not exist");
  }
  return nwmovecs_file_valid;
}

void ECNWChem::write_nwmovecs(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a_vec,
                              std::string files_prefix) {
  Matrix nw_movecs = C_a;
  reorder_ec_orbitals(true, chem_env.shells, nw_movecs, C_a);

  int32_t       length;
  std::ofstream movecs(files_prefix + ".nwmovecs", std::ios::binary);

  std::vector<char> date(26);
  std::time_t currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm*  localTime   = std::localtime(&currentTime);
  const char* format      = "%a %b %d %H:%M:%S %Y";
  std::strftime(date.data(), date.size(), format, localTime);

  char* header = new char[142];
  memset(header, 0, 142);
  memcpy(header + 96, "dft", 3);
  memcpy(header + 116, date.data(), 26);
  write_record<char>(movecs, header, 142);

  length = 20;
  memset(header, 0, 20);
  memcpy(header, "dft", 3);
  write_record<char>(movecs, header, 20);

  int64_t lentit = 7;
  length         = sizeof(lentit);
  memcpy(header, "ExaChem", 7);
  write_record<int64_t>(movecs, &lentit, length);
  write_record<char>(movecs, header, 7);

  int64_t lenbas = 8;
  length         = sizeof(lenbas);
  memcpy(header, "ao basis", 8);
  write_record<int64_t>(movecs, &lenbas, length);
  write_record<char>(movecs, header, 8);

  int64_t nsets = 1;
  int64_t nbf   = C_a.rows();
  int64_t nmo   = C_a.cols();
  length        = sizeof(nsets);
  write_record<int64_t>(movecs, &nsets, length);
  write_record<int64_t>(movecs, &nbf, length);
  write_record<int64_t>(movecs, &nmo, length);

  std::vector<double> occs(nbf, 0.0);
  for(int64_t imo = 0; imo < chem_env.sys_data.nelectrons_alpha; imo++) occs[imo] = 2.0;
  length = nbf * sizeof(double);
  write_record<double>(movecs, occs.data(), length);
  write_record<double>(movecs, eps_a_vec.data(), length);

  std::vector<double> _buffer(nbf);
  for(int64_t imo = 0; imo < nmo; imo++) {
    for(int64_t ibf = 0; ibf < nbf; ibf++) _buffer[ibf] = nw_movecs(ibf, imo);
    write_record<double>(movecs, _buffer.data(), length);
  }

  std::vector<double> zero = {0.0, 0.0};
  length                   = 2 * sizeof(double);
  write_record<double>(movecs, zero.data(), length);

  movecs.close();
  delete[] header;
}

void ECNWChem::write_nwmovecs(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a_vec,
                              Matrix& C_b, std::vector<double>& eps_b_vec,
                              std::string files_prefix) {
  Matrix nw_movecs_a = C_a;
  Matrix nw_movecs_b = C_b;
  reorder_ec_orbitals(true, chem_env.shells, nw_movecs_a, C_a);
  reorder_ec_orbitals(true, chem_env.shells, nw_movecs_b, C_b);

  int32_t       length;
  std::ofstream movecs(files_prefix + ".nwmovecs", std::ios::binary);

  std::vector<char> date(26);
  std::time_t currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm*  localTime   = std::localtime(&currentTime);
  const char* format      = "%a %b %d %H:%M:%S %Y";
  std::strftime(date.data(), date.size(), format, localTime);

  char* header = new char[142];
  memset(header, 0, 142);
  memcpy(header + 96, "dft", 3);
  memcpy(header + 116, date.data(), 26);
  write_record<char>(movecs, header, 142);

  length = 20;
  memset(header, 0, 20);
  memcpy(header, "dft", 3);
  write_record<char>(movecs, header, 20);

  int64_t lentit = 7;
  length         = sizeof(lentit);
  memcpy(header, "ExaChem", 7);
  write_record<int64_t>(movecs, &lentit, length);
  write_record<char>(movecs, header, 7);

  int64_t lenbas = 8;
  length         = sizeof(lenbas);
  memcpy(header, "ao basis", 8);
  write_record<int64_t>(movecs, &lenbas, length);
  write_record<char>(movecs, header, 8);

  int64_t              nsets = 2;
  int64_t              nbf   = C_a.rows();
  std::vector<int64_t> nmo   = {C_a.cols(), C_a.cols()};
  length                     = sizeof(nsets);
  write_record<int64_t>(movecs, &nsets, length);
  write_record<int64_t>(movecs, &nbf, length);
  write_record<int64_t>(movecs, nmo.data(), sizeof(nmo));

  std::vector<double> occs(nbf, 0.0);
  for(int64_t imo = 0; imo < chem_env.sys_data.nelectrons_alpha; imo++) occs[imo] = 1.0;
  length = nbf * sizeof(double);
  write_record<double>(movecs, occs.data(), length);
  write_record<double>(movecs, eps_a_vec.data(), length);

  std::vector<double> _buffer(nbf);
  for(int64_t imo = 0; imo < nmo[0]; imo++) {
    for(int64_t ibf = 0; ibf < nbf; ibf++) _buffer[ibf] = nw_movecs_a(ibf, imo);
    write_record<double>(movecs, _buffer.data(), length);
  }

  for(int64_t imo = 0; imo < chem_env.sys_data.nelectrons_alpha; imo++) occs[imo] = 0.0;
  for(int64_t imo = 0; imo < chem_env.sys_data.nelectrons_beta; imo++) occs[imo] = 1.0;
  length = nbf * sizeof(double);
  write_record<double>(movecs, occs.data(), length);
  write_record<double>(movecs, eps_b_vec.data(), length);

  for(int64_t imo = 0; imo < nmo[0]; imo++) {
    for(int64_t ibf = 0; ibf < nbf; ibf++) _buffer[ibf] = nw_movecs_b(ibf, imo);
    write_record<double>(movecs, _buffer.data(), length);
  }

  std::vector<double> zero = {0.0, 0.0};
  length                   = 2 * sizeof(double);
  write_record<double>(movecs, zero.data(), length);

  movecs.close();
  delete[] header;
}

void ECNWChem::reorder_nwchem_orbitals(const bool is_spherical, const libint2::BasisSet& shells,
                                       Matrix& nw_mat, Matrix& ec_mat) {
  auto   shell2bf = shells.shell2bf();
  size_t nsh      = shells.size();
  size_t k        = 0;
  for(size_t ish = 0; ish < nsh; ish++) {
    int l   = shells[ish].contr[0].l;
    int ibf = shell2bf[ish];
    if(l == 0) {
      ec_mat.row(ibf) = nw_mat.row(ibf);
      k++;
    }
    else if(l == 1) {
      ec_mat.row(ibf)     = nw_mat.row(ibf + 1);
      ec_mat.row(ibf + 1) = nw_mat.row(ibf + 2);
      ec_mat.row(ibf + 2) = nw_mat.row(ibf);
      k += 3;
    }
    else {
      for(int m = -l, i = 0; m <= l; m++, i++) {
        double phase        = m % 2 > 0 ? -1.0 : 1.0;
        ec_mat.row(ibf + i) = phase * nw_mat.row(ibf + i);
        k++;
      }
    }
  }
}

void ECNWChem::reorder_ec_orbitals(const bool is_spherical, const libint2::BasisSet& shells,
                                   Matrix& nw_mat, Matrix& ec_mat) {
  auto   shell2bf = shells.shell2bf();
  size_t nsh      = shells.size();
  size_t k        = 0;
  for(size_t ish = 0; ish < nsh; ish++) {
    int l   = shells[ish].contr[0].l;
    int ibf = shell2bf[ish];
    if(l == 0) {
      nw_mat.row(ibf) = nw_mat.row(ibf);
      k++;
    }
    else if(l == 1) {
      nw_mat.row(ibf)     = ec_mat.row(ibf + 2);
      nw_mat.row(ibf + 1) = ec_mat.row(ibf);
      nw_mat.row(ibf + 2) = ec_mat.row(ibf + 1);
      k += 3;
    }
    else {
      for(int m = -l, i = 0; m <= l; m++, i++) {
        double phase        = m % 2 > 0 ? -1.0 : 1.0;
        nw_mat.row(ibf + i) = phase * ec_mat.row(ibf + i);
        k++;
      }
    }
  }
}

void ECNWChem::read_nwmovecs(ChemEnv& chem_env, Matrix& C_alpha, Matrix& C_beta,
                             std::vector<double>& eps_a, std::vector<double>& eps_b) {
  SystemData&              sys_data    = chem_env.sys_data;
  const libint2::BasisSet& shells      = chem_env.shells;
  SCFOptions&              scf_options = chem_env.ioptions.scf_options;

  std::ifstream movecs(scf_options.nwmovecsfile, std::ios::in | std::ios::binary);
  if(!movecs.is_open()) { tamm_terminate("Could not open nwmovecsfile"); }

  int64_t           lentit, lenbas, nsets, nbf;
  std::vector<char> buffer(512);

  // Read Header information
  read_record<char>(movecs, buffer.data());

  // SCFTYPE
  read_record<char>(movecs, buffer.data());

  // TITLE
  read_record<int64_t>(movecs, &lentit);
  read_record<char>(movecs, buffer.data());

  // BASIS
  read_record<int64_t>(movecs, &lenbas);
  read_record<char>(movecs, buffer.data());

  // NSETS
  read_record<int64_t>(movecs, &nsets);

  // NBF
  read_record<int64_t>(movecs, &nbf);
  if(nbf != C_alpha.rows()) {
    std::cout << std::endl
              << std::endl
              << "Expected Nbf = " << C_alpha.rows() << ", got Nbf = " << nbf << std::endl;
    tamm_terminate("Error reading NWChem movecs file");
  }

  // NMO
  std::vector<int64_t> nmo(nsets);
  read_record<int64_t>(movecs, nmo.data());

  // MOVECS
  std::vector<Eigen::VectorXd> evals, occs;
  std::vector<Matrix>          nw_movecs;
  for(int64_t iset = 0; iset < nsets; iset++) {
    std::vector<double> _buffer(nbf);
    Eigen::VectorXd     _evals(nbf), _occs(nbf);
    Matrix              _movecs(nbf, nmo[iset]);
    read_record<double>(movecs, _buffer.data());
    for(int64_t ibf = 0; ibf < nbf; ibf++) _occs(ibf) = _buffer[ibf];
    occs.push_back(_occs);

    read_record<double>(movecs, _buffer.data());
    if(iset == 0) {
      eps_a.resize(nmo[iset], 0.0);
      for(int64_t ibf = 0; ibf < nbf; ibf++) eps_a[ibf] = _buffer[ibf];
    }
    else if(iset == 1) {
      eps_b.resize(nmo[iset], 0.0);
      for(int64_t ibf = 0; ibf < nbf; ibf++) eps_b[ibf] = _buffer[ibf];
    }

    for(int64_t imo = 0; imo < nmo[iset]; imo++) {
      read_record<double>(movecs, _buffer.data());
      for(int64_t ibf = 0; ibf < nbf; ibf++) _movecs(ibf, imo) = _buffer[ibf];
    }
    nw_movecs.push_back(_movecs);
  }
  movecs.close();

  reorder_nwchem_orbitals(true, shells, nw_movecs[0], C_alpha);
  if(sys_data.is_unrestricted) {
    if(nsets < 2) {
      C_beta = C_alpha;
      eps_b  = eps_a;
    }
    else { reorder_nwchem_orbitals(true, shells, nw_movecs[1], C_beta); }
  }
}

template<typename T>
void ECNWChem::write_record(std::ofstream& file, T* data, int32_t length) {
  file.write(reinterpret_cast<char*>(&length), sizeof(length));
  file.write(reinterpret_cast<char*>(data), length);
  file.write(reinterpret_cast<char*>(&length), sizeof(length));
}

template<typename T>
void ECNWChem::read_record(std::ifstream& file, T* data) {
  int32_t length;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));
  file.read(reinterpret_cast<char*>(data), length);
  file.read(reinterpret_cast<char*>(&length), sizeof(length));
}
