/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include <exachem/exachem_git.hpp>
#include <exachem/task/ec_task.hpp>
#include <tamm/tamm_config.hpp>
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

  auto ec_t1 = std::chrono::high_resolution_clock::now();

  if(rank == 0) {
    cout << endl << "program: " << fs::canonical(argv[0]) << endl;
    std::cout << std::endl;
    ec.print_execution_environment();
    std::cout << std::endl << tamm_build_config() << std::endl;
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

      // Store the execution environment and build configuration in results, as printed above
      auto& output = chem_env.sys_data.results["output"];

      output["execution_environment"] = json::parse(ec.execution_environment_json());

      json build_configuration;
      build_configuration["git"]["exachem"] = json::parse(exachem_git_json());
      build_configuration["git"]["tamm"]    = json::parse(tamm_git_json());
      build_configuration.update(json::parse(tamm_build_config_json()));
      output["build_configuration"] = build_configuration;
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

    if(txt_utils::strequal_case(task_op[0], "gradient")) {
      std::string grad_type = (task_op.size() > 1) ? task_op.at(1) : "analytical";
      if(!task.scf) grad_type = "numerical"; // force numerical for any non-scf task for now
      if(grad_type == "numerical") chem_env.sys_data.gradient_type = GradientType::Numerical;
      else chem_env.sys_data.gradient_type = GradientType::Analytical;
      exachem::gradients::ECGradients::compute_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);
    }
    else if(txt_utils::strequal_case(task_op[0], "optimize")) {
      std::string grad_type = (task_op.size() > 1) ? task_op.at(1) : "analytical";
      if(!task.scf) grad_type = "numerical"; // force numerical for any non-scf task for now
      if(txt_utils::strequal_case(grad_type, "numerical"))
        chem_env.sys_data.gradient_type = GradientType::Numerical;
      else chem_env.sys_data.gradient_type = GradientType::Analytical;
#if defined(TAMM_USE_PYTHON)
      const std::string geom_opt = (task_op.size() == 3) ? task_op.at(2) : "geometric";
      if(txt_utils::strequal_case(geom_opt, "pyberny")) {
        exachem::optimizers::PyBerny::optimize(ec, chem_env, atoms, ec_atoms, ec_arg2);
      }
      else { // geomeTRIC
        exachem::optimizers::GeomeTRIC::optimize(ec, chem_env, atoms, ec_atoms, ec_arg2);
        ec.pg().barrier();
        exachem::optimizers::finalize_python();
      }
#else
      // PyBerny
      exachem::optimizers::PyBerny::optimize(ec, chem_env, atoms, ec_atoms, ec_arg2);
#endif
    }
    else if(txt_utils::strequal_case(task_op[0], "ipi") ||
            txt_utils::strequal_case(task_op[0], "ase")) {
      // Run as a persistent force/energy server for an external simulation
      // driver. The optional 2nd element selects the gradient method.
      std::string grad_type = (task_op.size() > 1) ? task_op.at(1) : "analytical";
      if(!task.scf) grad_type = "numerical"; // force numerical for any non-scf task for now
      if(txt_utils::strequal_case(grad_type, "numerical"))
        chem_env.sys_data.gradient_type = GradientType::Numerical;
      else chem_env.sys_data.gradient_type = GradientType::Analytical;
      if(txt_utils::strequal_case(task_op[0], "ipi")) {
        exachem::integrations::ipi::run("localhost", 31415, 1, "", ec, chem_env, atoms, ec_atoms,
                                        ec_arg2);
      }
      else { exachem::integrations::ase::run(ec, chem_env, atoms, ec_atoms, ec_arg2); }
    }
    else exachem::task::compute_energy(ec, chem_env, ec_arg2);

    if(ec.print()) chem_env.write_run_context();

  } // loop over input files

  auto                          ec_t2       = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> ec_duration = ec_t2 - ec_t1;
  if(rank == 0) {
    std::cout << std::endl
              << "Total ExaChem runtime: " << std::fixed << std::setprecision(2)
              << ec_duration.count() << " secs" << std::endl
              << std::endl;
  }

  ec.flush_and_sync();
  ec.pg().destroy_coll();
  tamm::finalize();

  return 0;
}
