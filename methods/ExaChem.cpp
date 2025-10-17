/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include <exachem/exachem_git.hpp>
// #include <exachem/task/ec_task.hpp>
#include <exachem/task/geometry_optimizer.hpp>
#include <tamm/tamm_git.hpp>

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 2) tamm_terminate("Please provide an input file or folder!");

  const auto       rank = ProcGroup::world_rank();
  ProcGroup        pg   = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  if(rank == 0) {
    std::cout << exachem_git_info() << std::endl;
    std::cout << tamm_git_info() << std::endl;
  }

  std::ostringstream cur_date;
  if(rank == 0) {
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    cur_date << std::put_time(cur_local_time, "%c");
    cout << endl << "date: " << cur_date.str() << endl;
    cout << "program: " << fs::canonical(argv[0]) << endl;
    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc_per_node: " << ec.ppn() << ", ";
    std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
    if(ec.has_gpu()) {
      std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
      std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << endl;
    }
    std::cout << std::endl;
    ec.print_mem_info();
  }

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

  for(auto ifile: inputfiles) {
    std::string   input_file = fs::canonical(ifile);
    std::ifstream testinput(input_file);
    if(!testinput) tamm_terminate("Input file provided [" + input_file + "] does not exist!");

    // read geometry from a json file
    ChemEnv chem_env;
    chem_env.input_file = input_file;

    if(rank == 0) {
      cout << endl << std::string(60, '-') << endl;
      cout << endl << "Input file provided: " << input_file << endl << endl;
    }

    // This call should update all input options and SystemData object
    std::unique_ptr<ECOptionParser> iparse = std::make_unique<ECOptionParser>(chem_env);

    ECOptions& ioptions              = chem_env.ioptions;
    chem_env.sys_data.input_molecule = ParserUtils::getfilename(input_file);

    std::string output_dir = chem_env.ioptions.common_options.output_dir;
    if(chem_env.ioptions.common_options.file_prefix.empty()) {
      chem_env.ioptions.common_options.file_prefix = chem_env.sys_data.input_molecule;
    }
    if(!output_dir.empty()) {
      output_dir += "/";
      const auto    test_file = output_dir + "ec_test_file.tmp";
      std::ofstream ofs(test_file);
      if(!ofs) {
        tamm_terminate("[ERROR] Path provided as output_dir [" +
                       chem_env.ioptions.common_options.output_dir +
                       "] is not writable (or) does not exist");
      }
      ofs.close();
      fs::remove(test_file);
    }

    chem_env.sys_data.output_file_prefix =
      chem_env.ioptions.common_options.file_prefix + "." + chem_env.ioptions.common_options.basis;
    chem_env.workspace_dir = output_dir + chem_env.sys_data.output_file_prefix + "_files/";

    if(rank == 0) {
      std::cout << chem_env.jinput.dump(2) << std::endl;
      cout << endl
           << "Output folder & files prefix: " << chem_env.sys_data.output_file_prefix << endl
           << endl;

      // Store machine info in results
      auto& machine_info = chem_env.sys_data.results["output"]["machine_info"];
      auto  meminfo      = ec.mem_info();

      machine_info["date"]                           = cur_date.str();
      machine_info["nnodes"]                         = ec.nnodes();
      machine_info["nproc_per_node"]                 = ec.ppn();
      machine_info["nproc_total"]                    = ec.nnodes() * ec.ppn();
      machine_info["cpu"]["name"]                    = meminfo.cpu_name;
      machine_info["cpu"]["cpu_memory_per_node_gib"] = meminfo.cpu_mem_per_node;
      machine_info["cpu"]["total_cpu_memory_gib"]    = meminfo.total_cpu_mem;
      if(ec.has_gpu()) {
        machine_info["ngpus_per_node"]                 = ec.gpn();
        machine_info["ngpus_total"]                    = ec.nnodes() * ec.gpn();
        machine_info["gpu"]["name"]                    = meminfo.gpu_name;
        machine_info["gpu"]["memory_per_gpu_gib"]      = meminfo.gpu_mem_per_device;
        machine_info["gpu"]["gpu_memory_per_node_gib"] = meminfo.gpu_mem_per_node;
        machine_info["gpu"]["total_gpu_memory_gib"]    = meminfo.total_gpu_mem;
      }
    }

    const auto task = ioptions.task_options;

    std::string ec_arg2{};
    if(argc == 3) {
      ec_arg2 = std::string(argv[2]);
      if(!fs::exists(ec_arg2))
        tamm_terminate("Input file provided [" + ec_arg2 + "] does not exist!");
    }

    chem_env.read_run_context();

    const auto          task_op  = task.operation;
    std::vector<Atom>   atoms    = chem_env.atoms;
    std::vector<ECAtom> ec_atoms = chem_env.ec_atoms;

    if(task_op[0] == "gradient") {
      exachem::task::NumericalGradients::compute_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);
    }
    else if(task_op[0] == "optimize") {
      exachem::task::GeometryOptimizer::geometry_optimizer(ec, chem_env, atoms, ec_atoms, ec_arg2);
    }
    else exachem::task::NumericalGradients::compute_energy(ec, chem_env, ec_arg2);

    if(ec.print()) chem_env.write_run_context();

  } // loop over input files

  ec.flush_and_sync();
  ec.pg().destroy_coll();
  tamm::finalize();

  return 0;
}
