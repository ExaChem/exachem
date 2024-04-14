#include "parse_task_options.hpp"

ParseTaskOptions::ParseTaskOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseTaskOptions::parse_check(json& jinput) {
  const std::vector<string> valid_tasks{
    "sinfo",  "scf",         "fci",          "fcidump",   "mp2",        "gw",      "cd_2e",
    "cc2",    "dlpno_ccsd",  "dlpno_ccsd_t", "ducc",      "ccsd",       "ccsd_sf", "ccsd_t",
    "gfccsd", "ccsd_lambda", "eom_ccsd",     "rteom_cc2", "rteom_ccsd", "comments"};

  for(auto& el: jinput["TASK"].items()) {
    if(std::find(valid_tasks.begin(), valid_tasks.end(), el.key()) == valid_tasks.end())
      tamm_terminate("INPUT FILE ERROR: Invalid TASK option [" + el.key() + "] in the input file");
  }
}

void ParseTaskOptions::parse(ChemEnv& chem_env) {
  json         jtask        = chem_env.jinput["TASK"];
  TaskOptions& task_options = chem_env.ioptions.task_options;

  // clang-format off
  parse_option<bool>(task_options.sinfo, jtask, "sinfo");
  parse_option<bool>(task_options.scf, jtask, "scf");
  parse_option<bool>(task_options.mp2, jtask, "mp2");
  parse_option<bool>(task_options.gw, jtask, "gw");
  parse_option<bool>(task_options.cc2, jtask, "cc2");
  parse_option<bool>(task_options.fci, jtask, "fci");
  parse_option<bool>(task_options.fcidump, jtask, "fcidump");
  parse_option<bool>(task_options.cd_2e, jtask, "cd_2e");
  parse_option<bool>(task_options.ducc, jtask, "ducc");
  parse_option<bool>(task_options.ccsd, jtask, "ccsd");
  parse_option<bool>(task_options.ccsd_sf, jtask, "ccsd_sf");
  parse_option<bool>(task_options.ccsd_t, jtask, "ccsd_t");
  parse_option<bool>(task_options.ccsd_lambda, jtask, "ccsd_lambda");
  parse_option<bool>(task_options.eom_ccsd, jtask, "eom_ccsd");
  parse_option<bool>(task_options.rteom_cc2, jtask, "rteom_cc2");
  parse_option<bool>(task_options.rteom_ccsd, jtask, "rteom_ccsd");
  parse_option<bool>(task_options.gfccsd, jtask, "gfccsd");

  parse_option<std::pair<bool, std::string>>(task_options.dlpno_ccsd, jtask, "dlpno_ccsd");
  parse_option<std::pair<bool, std::string>>(task_options.dlpno_ccsd_t, jtask, "dlpno_ccsd_t");
  // clang-format on
  update_common_options(chem_env);
}

void ParseTaskOptions::update_common_options(ChemEnv& chem_env) {
  TaskOptions&   task_options   = chem_env.ioptions.task_options;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  task_options.debug         = common_options.debug;
  task_options.maxiter       = common_options.maxiter;
  task_options.basis         = common_options.basis;
  task_options.dfbasis       = common_options.dfbasis;
  task_options.basisfile     = common_options.basisfile;
  task_options.gaussian_type = common_options.gaussian_type;
  task_options.geom_units    = common_options.geom_units;
  task_options.file_prefix   = common_options.file_prefix;
  task_options.ext_data_path = common_options.ext_data_path;
}

void ParseTaskOptions::print(ChemEnv& chem_env) {
  std::cout << std::endl << "Task Options" << std::endl;
  std::cout << "{" << std::endl;

  TaskOptions& task_options = chem_env.ioptions.task_options;
  txt_utils::print_bool(" sinfo        ", task_options.sinfo);
  txt_utils::print_bool(" scf          ", task_options.scf);
  txt_utils::print_bool(" mp2          ", task_options.mp2);
  txt_utils::print_bool(" gw           ", task_options.gw);
  txt_utils::print_bool(" cc2          ", task_options.cc2);
  txt_utils::print_bool(" fci          ", task_options.fci);
  txt_utils::print_bool(" fcidump      ", task_options.fcidump);
  txt_utils::print_bool(" cd_2e        ", task_options.cd_2e);
  txt_utils::print_bool(" ducc         ", task_options.ducc);
  txt_utils::print_bool(" ccsd         ", task_options.ccsd);
  txt_utils::print_bool(" ccsd_sf      ", task_options.ccsd_sf);
  txt_utils::print_bool(" ccsd_lambda  ", task_options.ccsd_lambda);
  txt_utils::print_bool(" eom_ccsd     ", task_options.eom_ccsd);
  txt_utils::print_bool(" rteom_cc2    ", task_options.rteom_cc2);
  txt_utils::print_bool(" rteom_ccsd   ", task_options.rteom_ccsd);
  txt_utils::print_bool(" gfccsd       ", task_options.gfccsd);
  std::cout << " dlpno_ccsd:  " << task_options.dlpno_ccsd.first << ", "
            << task_options.dlpno_ccsd.second << "\n";
  std::cout << " dlpno_ccsd_t " << task_options.dlpno_ccsd_t.first << ", "
            << task_options.dlpno_ccsd_t.second << "\n";
  std::cout << "}" << std::endl;
}