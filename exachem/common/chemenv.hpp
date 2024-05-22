#pragma once
#include "common/ec_basis.hpp"
#include "common/system_data.hpp"
#include "ecatom.hpp"
#include "options/input_options.hpp"
// #include "libint2_includes.hpp"
#include "tamm/tamm.hpp"
#include "txt_utils.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

class ChemEnv {
public:
  using TensorType = double;

  void write_sinfo();
  // void write_json_data(const std::string cmodule);
  void sinfo();
  int  get_nfcore();

  ChemEnv() = default;

  json        jinput; //!< container for the parsed input data.
  std::string input_file;
  SystemData  sys_data;
  ECOptions   ioptions;
  ECBasis     ec_basis;

  std::vector<Atom>   atoms;
  std::vector<ECAtom> ec_atoms;

  double                   hf_energy{0.0};
  libint2::BasisSet        shells;
  std::vector<size_t>      shell_tile_map;
  tamm::Tensor<TensorType> C_AO;
  tamm::Tensor<TensorType> F_AO;
  tamm::Tensor<TensorType> C_beta_AO;
  tamm::Tensor<TensorType> F_beta_AO;
  TiledIndexSpace          AO_opt;
  TiledIndexSpace          AO_tis;
  TiledIndexSpace          AO_ortho;
  bool                     no_scf;

  std::string workspace_dir{};

  void write_json_data(const std::string cmodule);

  Matrix compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A);

  void update(double hf_energy, libint2::BasisSet shells, std::vector<size_t> shell_tile_map,
              tamm::Tensor<TensorType> C_AO, tamm::Tensor<TensorType> F_AO,
              tamm::Tensor<TensorType> C_beta_AO, tamm::Tensor<TensorType> F_beta_AO,
              TiledIndexSpace AO_opt, TiledIndexSpace AO_tis, TiledIndexSpace AO_ortho,
              bool no_scf) {
    this->hf_energy      = hf_energy;
    this->shells         = shells;
    this->shell_tile_map = shell_tile_map;
    this->C_AO           = C_AO;
    this->F_AO           = F_AO;
    this->C_beta_AO      = C_beta_AO;
    this->F_beta_AO      = F_beta_AO;
    this->AO_opt         = AO_opt;
    this->AO_tis         = AO_tis;
    this->AO_ortho       = AO_ortho;
    this->no_scf         = no_scf;
  }
};
