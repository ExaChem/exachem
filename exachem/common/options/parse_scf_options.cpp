#include "parse_scf_options.hpp"

ParseSCFOptions::ParseSCFOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseSCFOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseSCFOptions::parse_check(json& jinput) {
  // clang-format off
  const std::vector<std::string> valid_scf{"charge", "multiplicity", "lshift", "tol_int", "tol_sch",
    "tol_lindep", "conve", "convd", "diis_hist","force_tilesize","tilesize","df_tilesize",
    "damp","writem","nnodes","restart","noscf","moldenfile", "guess",
    "debug","scf_type","xc_type", "xc_grid_type", "n_lindep","restart_size","scalapack_nb","riscf",
    "scalapack_np_row","scalapack_np_col","ext_data_path","PRINT",
    "qed_omegas","qed_lambdas","qed_volumes","qed_polvecs","comments"};
  // clang-format on

  for(auto& el: jinput["SCF"].items()) {
    if(std::find(valid_scf.begin(), valid_scf.end(), el.key()) == valid_scf.end())
      tamm_terminate("INPUT FILE ERROR: Invalid SCF option [" + el.key() + "] in the input file");
  }
}

void ParseSCFOptions::parse(ChemEnv& chem_env) {
  json        jscf        = chem_env.jinput["SCF"];
  SCFOptions& scf_options = chem_env.ioptions.scf_options;
  parse_option<int>(scf_options.charge, jscf, "charge");
  parse_option<int>(scf_options.multiplicity, jscf, "multiplicity");
  parse_option<double>(scf_options.lshift, jscf, "lshift");
  parse_option<double>(scf_options.tol_int, jscf, "tol_int");
  parse_option<double>(scf_options.tol_sch, jscf, "tol_sch");
  parse_option<double>(scf_options.tol_lindep, jscf, "tol_lindep");
  parse_option<double>(scf_options.conve, jscf, "conve");
  parse_option<double>(scf_options.convd, jscf, "convd");
  parse_option<int>(scf_options.diis_hist, jscf, "diis_hist");
  parse_option<bool>(scf_options.force_tilesize, jscf, "force_tilesize");
  parse_option<uint32_t>(scf_options.AO_tilesize, jscf, "tilesize");
  parse_option<uint32_t>(scf_options.dfAO_tilesize, jscf, "df_tilesize");
  parse_option<int>(scf_options.damp, jscf, "damp");
  parse_option<int>(scf_options.writem, jscf, "writem");
  parse_option<int>(scf_options.nnodes, jscf, "nnodes");
  parse_option<bool>(scf_options.restart, jscf, "restart");
  parse_option<bool>(scf_options.noscf, jscf, "noscf");
  parse_option<bool>(scf_options.debug, jscf, "debug");
  parse_option<std::string>(scf_options.moldenfile, jscf, "moldenfile");
  parse_option<std::string>(scf_options.scf_type, jscf, "scf_type");
  parse_option<std::string>(scf_options.xc_grid_type, jscf, "xc_grid_type");
  parse_option<std::vector<std::string>>(scf_options.xc_type, jscf, "xc_type");
  parse_option<int>(scf_options.n_lindep, jscf, "n_lindep");
  parse_option<int>(scf_options.restart_size, jscf, "restart_size");
  parse_option<int>(scf_options.scalapack_nb, jscf, "scalapack_nb");
  parse_option<int>(scf_options.scalapack_np_row, jscf, "scalapack_np_row");
  parse_option<int>(scf_options.scalapack_np_col, jscf, "scalapack_np_col");
  parse_option<std::string>(scf_options.ext_data_path, jscf, "ext_data_path");
  parse_option<std::vector<double>>(scf_options.qed_omegas, jscf, "qed_omegas");
  parse_option<std::vector<double>>(scf_options.qed_lambdas, jscf, "qed_lambdas");
  parse_option<std::vector<double>>(scf_options.qed_volumes, jscf, "qed_volumes");
  parse_option<std::vector<std::vector<double>>>(scf_options.qed_polvecs, jscf, "qed_polvecs");

  json jscf_guess          = jscf["guess"];
  json jguess_atom_options = jscf_guess["atom_options"];
  parse_option<bool>(scf_options.sad, jscf_guess, "sad");

  for(auto& [element_symbol, atom_opt]: jguess_atom_options.items()) {
    scf_options.guess_atom_options[element_symbol] = atom_opt;
  }

  if(scf_options.nnodes < 1 || scf_options.nnodes > 100) {
    tamm_terminate("INPUT FILE ERROR: SCF option nnodes should be a number between 1 and 100");
  }
  {
    auto xc_grid_str = scf_options.xc_grid_type;
    xc_grid_str.erase(remove_if(xc_grid_str.begin(), xc_grid_str.end(), isspace),
                      xc_grid_str.end());
    scf_options.xc_grid_type = xc_grid_str;
    std::transform(xc_grid_str.begin(), xc_grid_str.end(), xc_grid_str.begin(), ::tolower);
    if(xc_grid_str != "fine" && xc_grid_str != "ultrafine" && xc_grid_str != "superfine")
      tamm_terminate("INPUT FILE ERROR: SCF option xc_grid_type should be one of [Fine, "
                     "UltraFine, SuperFine]");
  }
  update_common_options(chem_env);
}

void ParseSCFOptions::update_common_options(ChemEnv& chem_env) {
  SCFOptions&    scf_options    = chem_env.ioptions.scf_options;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  scf_options.debug         = common_options.debug;
  scf_options.maxiter       = common_options.maxiter;
  scf_options.basis         = common_options.basis;
  scf_options.dfbasis       = common_options.dfbasis;
  scf_options.basisfile     = common_options.basisfile;
  scf_options.gaussian_type = common_options.gaussian_type;
  scf_options.geom_units    = common_options.geom_units;
  scf_options.file_prefix   = common_options.file_prefix;
  scf_options.ext_data_path = common_options.ext_data_path;
}

void ParseSCFOptions::print(ChemEnv& chem_env) {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "SCF COptions" << std::endl;
  std::cout << "{" << std::endl;

  SCFOptions& scf_options = chem_env.ioptions.scf_options;
  std::cout << " charge       = " << scf_options.charge << std::endl;
  std::cout << " multiplicity = " << scf_options.multiplicity << std::endl;
  std::cout << " level shift  = " << scf_options.lshift << std::endl;
  std::cout << " tol_int      = " << scf_options.tol_int << std::endl;
  std::cout << " tol_sch      = " << scf_options.tol_sch << std::endl;
  std::cout << " tol_lindep   = " << scf_options.tol_lindep << std::endl;
  std::cout << " conve        = " << scf_options.conve << std::endl;
  std::cout << " convd        = " << scf_options.convd << std::endl;
  std::cout << " diis_hist    = " << scf_options.diis_hist << std::endl;
  std::cout << " AO_tilesize  = " << scf_options.AO_tilesize << std::endl;
  std::cout << " writem       = " << scf_options.writem << std::endl;
  std::cout << " damp         = " << scf_options.damp << std::endl;
  if(!scf_options.moldenfile.empty()) {
    std::cout << " moldenfile   = " << scf_options.moldenfile << std::endl;
    // std::cout << " n_lindep = " << n_lindep <<  std::endl;
  }

  std::cout << " scf_type     = " << scf_options.scf_type << std::endl;

  // QED
  if(!scf_options.qed_omegas.empty()) {
    std::cout << " qed_omegas  = [";
    for(auto x: scf_options.qed_omegas) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!scf_options.qed_lambdas.empty()) {
    std::cout << " qed_lambdas  = [";
    for(auto x: scf_options.qed_lambdas) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!scf_options.qed_volumes.empty()) {
    std::cout << " qed_volumes  = [";
    for(auto x: scf_options.qed_volumes) { std::cout << x << ","; }
    std::cout << "\b]" << std::endl;
  }

  if(!scf_options.qed_polvecs.empty()) {
    std::cout << " qed_polvecs  = [";
    for(auto x: scf_options.qed_polvecs) {
      std::cout << "[";
      for(auto y: x) { std::cout << y << ","; }
      std::cout << "\b],";
    }
    std::cout << "\b]" << std::endl;
  }

  if(!scf_options.xc_type.empty()) {
    std::cout << " xc_type      = [ ";
    for(auto xcfunc: scf_options.xc_type) { std::cout << " \"" << xcfunc << "\","; }
    std::cout << "\b ]" << std::endl;
    std::cout << " xc_grid_type = " << scf_options.xc_grid_type << std::endl;
  }

  if(scf_options.scalapack_np_row > 0 && scf_options.scalapack_np_col > 0) {
    std::cout << " scalapack_np_row = " << scf_options.scalapack_np_row << std::endl;
    std::cout << " scalapack_np_col = " << scf_options.scalapack_np_col << std::endl;
    if(scf_options.scalapack_nb > 1)
      std::cout << " scalapack_nb = " << scf_options.scalapack_nb << std::endl;
  }
  std::cout << " restart_size = " << scf_options.restart_size << std::endl;
  txt_utils::print_bool(" restart     ", scf_options.restart);
  txt_utils::print_bool(" debug       ", scf_options.debug);
  if(scf_options.restart) txt_utils::print_bool(" noscf       ", scf_options.noscf);
  // txt_utils::print_bool(" sad         ", scf_options.sad);
  if(scf_options.mulliken_analysis || scf_options.mos_txt ||
     scf_options.mo_vectors_analysis.first) {
    std::cout << " PRINT {" << std::endl;
    if(scf_options.mos_txt)
      std::cout << std::boolalpha << "  mos_txt             = " << scf_options.mos_txt << std::endl;
    if(scf_options.mulliken_analysis)
      std::cout << std::boolalpha << "  mulliken_analysis   = " << scf_options.mulliken_analysis
                << std::endl;
    if(scf_options.mo_vectors_analysis.first) {
      std::cout << "  mo_vectors_analysis = [" << std::boolalpha
                << scf_options.mo_vectors_analysis.first;
      std::cout << "," << scf_options.mo_vectors_analysis.second << "]" << std::endl;
    }
    std::cout << " }" << std::endl;
  }
  std::cout << "}" << std::endl;
}