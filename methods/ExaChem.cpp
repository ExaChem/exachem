/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include <exachem/exachem_git.hpp>
#include <tamm/tamm_git.hpp>

#define EC_CC
#define EC_COMPLEX

// clang-format off
#if defined(EC_CC)
#include "cc/ccsd/cd_ccsd_os_ann.hpp"
#include "cc/ccsd_t/ccsd_t_fused_driver.hpp"
#include "exachem/cc/lambda/ccsd_lambda.hpp"
#include "exachem/cc/eom/eomccsd_opt.hpp"
#endif

#include "exachem/common/chemenv.hpp"
#include "exachem/common/options/parse_options.hpp"
#include "scf/scf_main.hpp"
// clang-format on
using namespace exachem;

#if !defined(USE_UPCXX) and defined(EC_COMPLEX) and defined(EC_CC)
namespace exachem::cc::gfcc {
void gfccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);
}
namespace exachem::rteom_cc::ccsd {
void rt_eom_cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);
}
#include "exachem/fci/fci.hpp"
#endif

#if defined(EC_CC)
namespace exachem::cc2 {
void cd_cc2_driver(ExecutionContext& ec, ChemEnv& chem_env);
}
namespace exachem::cc::ducc {
void ducc_driver(ExecutionContext& ec, ChemEnv& chem_env);
}
#endif

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 2) tamm_terminate("Please provide an input file or folder!");

  auto                     input_fpath = std::string(argv[1]);
  std::vector<std::string> inputfiles;

  if(fs::is_directory(input_fpath)) {
    for(auto const& dir_entry: std::filesystem::directory_iterator{input_fpath}) {
      if(fs::path(dir_entry.path()).extension() == ".json") inputfiles.push_back(dir_entry.path());
    }
  }
  else {
    if(!fs::exists(input_fpath))
      tamm_terminate("Input file or folder path provided [" + input_fpath + "] does not exist!");
    inputfiles.push_back(input_fpath);
  }

  if(inputfiles.empty()) tamm_terminate("No input files provided");

  const auto       rank = ProcGroup::world_rank();
  ProcGroup        pg   = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  for(auto ifile: inputfiles) {
    std::string   input_file = fs::canonical(ifile);
    std::ifstream testinput(input_file);
    if(!testinput) tamm_terminate("Input file provided [" + input_file + "] does not exist!");

    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);

    // read geometry from a json file
    ChemEnv chem_env;
    chem_env.input_file = input_file;

    // This call should update all input options and SystemData object
    std::unique_ptr<ECOptionParser> iparse = std::make_unique<ECOptionParser>(chem_env);

    ECOptions& ioptions              = chem_env.ioptions;
    chem_env.sys_data.input_molecule = ParserUtils::getfilename(input_file);

    if(chem_env.ioptions.common_options.file_prefix.empty()) {
      chem_env.ioptions.common_options.file_prefix = chem_env.sys_data.input_molecule;
    }

    chem_env.sys_data.output_file_prefix =
      chem_env.ioptions.common_options.file_prefix + "." + chem_env.ioptions.common_options.basis;
    chem_env.workspace_dir = chem_env.sys_data.output_file_prefix + "_files/";

    if(rank == 0) {
      std::cout << exachem_git_info() << std::endl;
      std::cout << tamm_git_info() << std::endl;
    }

    if(rank == 0) {
      std::ostringstream cur_date;
      cur_date << std::put_time(cur_local_time, "%c");
      cout << endl << "date: " << cur_date.str() << endl;
      cout << "program: " << fs::canonical(argv[0]) << endl;
      cout << "input: " << input_file << endl;
      std::cout << "nnodes: " << ec.nnodes() << ", ";
      std::cout << "nproc_per_node: " << ec.ppn() << ", ";
      std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
      std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
      std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
      cout << "prefix: " << chem_env.sys_data.output_file_prefix << endl << endl;
      ec.print_mem_info();
      cout << endl << endl;
      cout << "Input file provided" << endl << std::string(20, '-') << endl;
      std::cout << chem_env.jinput.dump(2) << std::endl;
      chem_env.sys_data.results["output"]["machine_info"]["date"]           = cur_date.str();
      chem_env.sys_data.results["output"]["machine_info"]["nnodes"]         = ec.nnodes();
      chem_env.sys_data.results["output"]["machine_info"]["nproc_per_node"] = ec.ppn();
      chem_env.sys_data.results["output"]["machine_info"]["nproc_total"] = ec.nnodes() * ec.ppn();
      auto meminfo                                                       = ec.mem_info();
      chem_env.sys_data.results["output"]["machine_info"]["cpu"]["name"] = meminfo.cpu_name;
      chem_env.sys_data.results["output"]["machine_info"]["cpu"]["cpu_memory_per_node_gib"] =
        meminfo.cpu_mem_per_node;
      chem_env.sys_data.results["output"]["machine_info"]["cpu"]["total_cpu_memory_gib"] =
        meminfo.total_cpu_mem;
      if(ec.has_gpu()) {
        chem_env.sys_data.results["output"]["machine_info"]["ngpus_per_node"] = ec.gpn();
        chem_env.sys_data.results["output"]["machine_info"]["ngpus_total"] = ec.nnodes() * ec.gpn();
        chem_env.sys_data.results["output"]["machine_info"]["gpu"]["name"] = meminfo.gpu_name;
        chem_env.sys_data.results["output"]["machine_info"]["gpu"]["memory_per_gpu_gib"] =
          meminfo.gpu_mem_per_device;
        chem_env.sys_data.results["output"]["machine_info"]["gpu"]["gpu_memory_per_node_gib"] =
          meminfo.gpu_mem_per_node;
        chem_env.sys_data.results["output"]["machine_info"]["gpu"]["total_gpu_memory_gib"] =
          meminfo.total_gpu_mem;
      }
    }

    const auto              task = ioptions.task_options;
    const std::vector<bool> tvec = {task.sinfo,
                                    task.scf,
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

    SCFOptions& scf_options   = chem_env.ioptions.scf_options;
    chem_env.ec_basis         = ECBasis(ec, scf_options.basis, scf_options.basisfile,
                                        scf_options.gaussian_type, chem_env.atoms, chem_env.ec_atoms);
    chem_env.shells           = chem_env.ec_basis.shells;
    chem_env.sys_data.has_ecp = chem_env.ec_basis.has_ecp;

    if(task.sinfo) chem_env.sinfo();
    else if(task.scf) scf::scf_driver(ec, chem_env);
#if defined(EC_CC)
    else if(task.mp2) mp2::cd_mp2(ec, chem_env);
    else if(task.cd_2e) cd::cd_2e_driver(ec, chem_env);
    else if(task.ccsd) cc::ccsd::cd_ccsd(ec, chem_env);
    else if(task.ccsd_t) cc::ccsd_t::ccsd_t_driver(ec, chem_env);
    else if(task.cc2) cc2::cd_cc2_driver(ec, chem_env);
    else if(task.ccsd_lambda) cc::ccsd_lambda::ccsd_lambda_driver(ec, chem_env);
    else if(task.eom_ccsd) cc::eom::eom_ccsd_driver(ec, chem_env);
    else if(task.ducc) cc::ducc::ducc_driver(ec, chem_env);
#if !defined(USE_UPCXX) and defined(EC_COMPLEX)
    else if(task.fci || task.fcidump) fci::fci_driver(ec, chem_env);
    else if(task.gfccsd) cc::gfcc::gfccsd_driver(ec, chem_env);
    else if(task.rteom_ccsd) rteom_cc::ccsd::rt_eom_cd_ccsd_driver(ec, chem_env);
#endif

#endif

    else
      tamm_terminate(
        "[ERROR] Unsupported task specified (or) code for the specified task is not built");
  }

  tamm::finalize();

  return 0;
}
