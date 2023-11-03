/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

// clang-format off
#include "cc/ccsd/cd_ccsd_os_ann.hpp"
#include "cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "exachem/cc/lambda/ccsd_lambda.hpp"
#include "exachem/cc/eom/eomccsd_opt.hpp"
// clang-format on

#define EC_COMPLEX

#if !defined(USE_UPCXX) and defined(EC_COMPLEX)
void gfccsd_driver(std::string filename, OptionsMap options_map);
void rt_eom_cd_ccsd_driver(std::string filename, OptionsMap options_map);
#include "exachem/fci/fci.hpp"
#endif

void cd_cc2_driver(std::string filename, OptionsMap options_map);
void ducc_driver(std::string filename, OptionsMap options_map);

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 2) tamm_terminate("Please provide an input file!");

  std::string   filename = fs::canonical(std::string(argv[1]));
  std::ifstream testinput(filename);
  if(!testinput) tamm_terminate("Input file provided [" + filename + "] does not exist!");

  auto current_time   = std::chrono::system_clock::now();
  auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
  auto cur_local_time = localtime(&current_time_t);

  const auto       rank = ProcGroup::world_rank();
  ProcGroup        pg   = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  // read geometry from a json file
  json jinput;
  check_json(filename);
  auto is = std::ifstream(filename);

  OptionsMap options_map;
  std::tie(options_map, jinput) = parse_input(is);
  if(options_map.options.output_file_prefix.empty())
    options_map.options.output_file_prefix = getfilename(filename);

  if(rank == 0) {
    cout << endl << "date: " << std::put_time(cur_local_time, "%c") << endl;
    cout << "program: " << fs::canonical(argv[0]) << endl;
    cout << "input: " << filename << endl;
    cout << "nnodes: " << ec.nnodes() << ", ";
    cout << "nproc: " << ec.nnodes() * ec.ppn() << endl;
    cout << "prefix: " << options_map.options.output_file_prefix << endl << endl;
    ec.print_mem_info();
    cout << endl << endl;
    cout << "Input file provided" << endl << std::string(20, '-') << endl;
    std::cout << jinput.dump(2) << std::endl;
  }

  // if(rank == 0) {
  //   std::ofstream res_file(getfilename(filename)+".json");
  //   res_file << std::setw(2) << jinput << std::endl;
  // }

  const auto              task = options_map.task_options;
  const std::vector<bool> tvec = {task.scf,
                                  task.mp2,
                                  task.gw,
                                  task.fci,
                                  task.cd_2e,
                                  task.ducc,
                                  task.ccsd,
                                  task.ccsd_t,
                                  task.ccsd_lambda,
                                  task.eom_ccsd,
                                  task.fcidump,
                                  task.rteom_cc2,
                                  task.rteom_ccsd,
                                  task.gfccsd,
                                  task.dlpno_ccsd.first,
                                  task.dlpno_ccsd_t.first};
  if(std::count(tvec.begin(), tvec.end(), true) > 1)
    tamm_terminate("[INPUT FILE ERROR] only a single task can be enabled at once!");

  std::string ec_arg2{};
  if(argc == 3) {
    ec_arg2 = std::string(argv[2]);
    if(!fs::exists(ec_arg2))
      tamm_terminate("Input file provided [" + ec_arg2 + "] does not exist!");
  }

#if !defined(USE_MACIS)
  if(task.fci) tamm_terminate("Full CI integration not enabled!");
#endif

  if(task.scf) scf(filename, options_map);
  else if(task.mp2) cd_mp2(filename, options_map);
  else if(task.cd_2e) cd_2e_driver(filename, options_map);
  else if(task.ccsd) cd_ccsd(filename, options_map);
  else if(task.ccsd_t) ccsd_t_driver(filename, options_map);
  else if(task.cc2) cd_cc2_driver(filename, options_map);
  else if(task.ccsd_lambda) ccsd_lambda_driver(filename, options_map);
  else if(task.eom_ccsd) eom_ccsd_driver(filename, options_map);
  else if(task.ducc) ducc_driver(filename, options_map);
#if !defined(USE_UPCXX) and defined(EC_COMPLEX)
  else if(task.fci || task.fcidump) fci_driver(filename, options_map);
  else if(task.gfccsd) gfccsd_driver(filename, options_map);
  else if(task.rteom_ccsd) rt_eom_cd_ccsd_driver(filename, options_map);
#endif

  else
    tamm_terminate(
      "[ERROR] Unsupported task specified (or) code for the specified task is not built");

  tamm::finalize();

  return 0;
}
