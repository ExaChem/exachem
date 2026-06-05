// geometric.cpp

#include "exachem/optimizers/geometric.hpp"
#include "exachem/gradients/ec_gradients.hpp"
#include "exachem/task/task_interface.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace exachem::optimizers {

std::unique_ptr<py::scoped_interpreter> g_python;
std::once_flag                          g_python_once;

ExecutionContext*    g_ec                 = nullptr;
ChemEnv*             g_chem_env           = nullptr;
std::vector<Atom>*   g_atoms              = nullptr;
std::vector<ECAtom>* g_ec_atoms           = nullptr;
const std::string*   g_ec_arg2            = nullptr;
bool                 g_python_initialized = false;

// --------------------
// Helpers
// --------------------

Eigen::RowVectorXd
numpy_to_rowvec(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  auto buf = arr.request();
  if(buf.ndim != 1) throw std::runtime_error("Expected 1D NumPy array");

  Eigen::RowVectorXd v(buf.size);
  const auto*        ptr = static_cast<const double*>(buf.ptr);
  for(ssize_t i = 0; i < buf.size; ++i) v(i) = ptr[i];
  return v;
}

py::array_t<double> rowvec_to_numpy(const Eigen::RowVectorXd& v) {
  py::array_t<double> arr(v.size());
  auto                r = arr.mutable_unchecked<1>();
  for(Eigen::Index i = 0; i < v.size(); ++i) r(i) = v(i);
  return arr;
}

Eigen::RowVectorXd flatten_gradient(const Eigen::MatrixXd& grad_mat, size_t natoms, size_t ncart) {
  Eigen::RowVectorXd grad_flat(ncart);

  if(grad_mat.rows() == static_cast<Eigen::Index>(natoms) && grad_mat.cols() == 3) {
    Eigen::Index c = 0;
    for(Eigen::Index i = 0; i < grad_mat.rows(); ++i) {
      grad_flat(c++) = grad_mat(i, 0);
      grad_flat(c++) = grad_mat(i, 1);
      grad_flat(c++) = grad_mat(i, 2);
    }
    return grad_flat;
  }

  if(grad_mat.rows() == 1 && grad_mat.cols() == static_cast<Eigen::Index>(ncart)) {
    return grad_mat.row(0);
  }

  if(grad_mat.cols() == 1 && grad_mat.rows() == static_cast<Eigen::Index>(ncart)) {
    return grad_mat.col(0).transpose();
  }

  std::ostringstream oss;
  oss << "Unsupported gradient shape from ExaChem: " << grad_mat.rows() << " x " << grad_mat.cols()
      << ", expected natoms x 3 or flat vector length 3*natoms";
  throw std::runtime_error(oss.str());
}

std::string make_xyz_string(const std::vector<Atom>& atoms, const std::vector<ECAtom>& ec_atoms) {
  std::ostringstream oss;
  oss << atoms.size() << "\n";
  oss << "ExaChem geometry\n";
  for(size_t i = 0; i < atoms.size(); ++i) {
    oss << ec_atoms[i].get_symbol(atoms[i].atomic_number) << " "
        << atoms[i].x * exachem::constants::bohr2ang << " "
        << atoms[i].y * exachem::constants::bohr2ang << " "
        << atoms[i].z * exachem::constants::bohr2ang << "\n";
  }
  return oss.str();
}

fs::path get_workspace_dir(const ChemEnv& chem_env) {
  if(chem_env.workspace_dir.empty()) return fs::current_path();
  return fs::path(chem_env.workspace_dir);
}

// --------------------
// Python callbacks
// --------------------

py::dict
exachem_compute(py::array_t<double, py::array::c_style | py::array::forcecast> coords_bohr) {
  if(g_ec == nullptr || g_chem_env == nullptr || g_atoms == nullptr || g_ec_atoms == nullptr ||
     g_ec_arg2 == nullptr) {
    throw std::runtime_error("ExaChem geomeTRIC backend not initialized");
  }

  Eigen::RowVectorXd geom_bohr = numpy_to_rowvec(coords_bohr);

  g_chem_env->update_geometry(*g_atoms, *g_ec_atoms, geom_bohr);

  auto t_atoms    = g_chem_env->atoms;
  auto t_ec_atoms = g_chem_env->ec_atoms;

  g_chem_env->atoms    = *g_atoms;
  g_chem_env->ec_atoms = *g_ec_atoms;

  const double energy = exachem::task::compute_energy(*g_ec, *g_chem_env, *g_ec_arg2);

  const Eigen::MatrixXd grad_mat = exachem::gradients::ECGradients::compute_gradients(
    *g_ec, *g_chem_env, *g_atoms, *g_ec_atoms, *g_ec_arg2);

  const Eigen::RowVectorXd grad_flat =
    flatten_gradient(grad_mat, g_atoms->size(), geom_bohr.size());

  g_chem_env->atoms    = t_atoms;
  g_chem_env->ec_atoms = t_ec_atoms;

  py::dict result;
  result["energy"]   = py::float_(energy);
  result["gradient"] = rowvec_to_numpy(grad_flat);
  return result;
}

bool exachem_detect_dft() { return false; }

py::object
exachem_calc_bondorder(py::array_t<double, py::array::c_style | py::array::forcecast> coords_bohr) {
  (void) coords_bohr;
  return py::none();
}

void exachem_clear_calcs() {}

void exachem_save_guess_files(const std::string& dirname) { (void) dirname; }

void exachem_load_guess_files(const std::string& dirname) { (void) dirname; }

PYBIND11_EMBEDDED_MODULE(exachem_geometric_backend, m) {
  m.def("compute", &exachem_compute);
  m.def("detect_dft", &exachem_detect_dft);
  m.def("calc_bondorder", &exachem_calc_bondorder);
  m.def("clear_calcs", &exachem_clear_calcs);
  m.def("save_guess_files", &exachem_save_guess_files);
  m.def("load_guess_files", &exachem_load_guess_files);
}

// --------------------
// Python setup
// --------------------

void define_python_adapter() {
  py::exec(R"PYCODE(
import numpy as np
import geometric.prepare
import geometric.molecule
from geometric.engine import Engine
import exachem_geometric_backend

class ExaChemEngine(Engine):
    def __init__(self, molecule):
        super().__init__(molecule)

    def calc_new(self, coords, dirname):
        res = exachem_geometric_backend.compute(np.asarray(coords, dtype=float).reshape(-1))
        return {
            "energy": float(res["energy"]),
            "gradient": np.asarray(res["gradient"], dtype=float).reshape(-1),
        }

    def clearCalcs(self):
        exachem_geometric_backend.clear_calcs()
        if hasattr(self, "stored_calcs"):
            self.stored_calcs = {}

    def save_guess_files(self, dirname):
        exachem_geometric_backend.save_guess_files(dirname)

    def load_guess_files(self, dirname):
        exachem_geometric_backend.load_guess_files(dirname)

    def detect_dft(self):
        return bool(exachem_geometric_backend.detect_dft())

    def calc_bondorder(self, coords, dirname):
        res = exachem_geometric_backend.calc_bondorder(np.asarray(coords, dtype=float).reshape(-1))
        if res is None:
            raise NotImplementedError("Bond order not available")
        return np.asarray(res, dtype=float)

_original_get_molecule_engine = geometric.prepare.get_molecule_engine

def _exachem_get_molecule_engine(**kwargs):
    engine_name = kwargs.get("engine", None)

    inputf = kwargs.get("input", None)
    if inputf is None:
        raise RuntimeError("ExaChem geomeTRIC interface requires 'input' to be an XYZ file")

    Molecule = geometric.molecule.Molecule
    M = Molecule(inputf, fragment=False)

    eng = ExaChemEngine(M)
    return M, eng

geometric.prepare.get_molecule_engine = _exachem_get_molecule_engine
geometric.optimize.get_molecule_engine = _exachem_get_molecule_engine
)PYCODE");
}

void ensure_python_initialized() {
  std::call_once(g_python_once, []() {
    g_python = std::make_unique<py::scoped_interpreter>();
    py::gil_scoped_acquire gil;
    define_python_adapter();
    g_python_initialized = true;
  });
}

void finalize_python() {
  if(g_python) {
    py::gil_scoped_acquire gil;
    g_python.reset();
    g_python_initialized = false;
  }
}

Eigen::RowVectorXd GeomeTRIC::current_geometry(const std::vector<Atom>& atoms) {
  Eigen::RowVectorXd geom(3 * atoms.size());
  Eigen::Index       c = 0;

  for(const auto& a: atoms) {
    geom(c++) = a.x;
    geom(c++) = a.y;
    geom(c++) = a.z;
  }

  return geom;
}

void GeomeTRIC::optimize(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                         std::vector<ECAtom>& ec_atoms, const std::string& ec_arg2) {
  g_ec       = &ec;
  g_chem_env = &chem_env;
  g_atoms    = &atoms;
  g_ec_atoms = &ec_atoms;
  g_ec_arg2  = &ec_arg2;

  try {
    ensure_python_initialized();
    py::gil_scoped_acquire gil;

    const int rank = ec.pg().rank().value();

    const fs::path base_workdir = chem_env.get_files_dir();
    const fs::path workdir =
      base_workdir / "geometric" / ("geometric_rank_" + std::to_string(rank));
    fs::create_directories(workdir);

    const fs::path xyz_path = workdir / "geometric_input.xyz";
    {
      std::ofstream ofs(xyz_path);
      if(!ofs) {
        throw std::runtime_error("Failed to write temporary XYZ file: " + xyz_path.string());
      }
      ofs << make_xyz_string(atoms, ec_atoms);
    }

    const fs::path logini_path = workdir / "geometric_logging.ini";
    {
      std::ofstream logini(logini_path);
      if(!logini) {
        throw std::runtime_error("Failed to write logging config file: " + logini_path.string());
      }

      if(rank == 0) {
        logini << "[loggers]\n";
        logini << "keys=root\n\n";
        logini << "[handlers]\n";
        logini << "keys=consoleHandler\n\n";
        logini << "[formatters]\n";
        logini << "keys=simpleFormatter\n\n";
        logini << "[logger_root]\n";
        logini << "level=INFO\n";
        logini << "handlers=consoleHandler\n\n";
        logini << "[handler_consoleHandler]\n";
        logini << "class=StreamHandler\n";
        logini << "level=INFO\n";
        logini << "formatter=simpleFormatter\n";
        logini << "args=(sys.stdout,)\n\n";
        logini << "[formatter_simpleFormatter]\n";
        logini << "format=%(message)s\n";
      }
      else {
        logini << "[loggers]\n";
        logini << "keys=root\n\n";
        logini << "[handlers]\n";
        logini << "keys=nullHandler\n\n";
        logini << "[formatters]\n";
        logini << "keys=simpleFormatter\n\n";
        logini << "[logger_root]\n";
        logini << "level=CRITICAL\n";
        logini << "handlers=nullHandler\n\n";
        logini << "[handler_nullHandler]\n";
        logini << "class=logging.NullHandler\n";
        logini << "level=CRITICAL\n";
        logini << "args=()\n\n";
        logini << "[formatter_simpleFormatter]\n";
        logini << "format=%(message)s\n";
      }
    }

    py::module_ optimize_m = py::module_::import("geometric.optimize");

    py::dict kwargs;
    kwargs["engine"] = "ExaChemEngine";
    kwargs["input"]  = xyz_path.string();

    std::string file_prefix = chem_env.ioptions.common_options.file_prefix.empty()
                                ? "geometric"
                                : chem_env.ioptions.common_options.file_prefix;

    fs::path prefix_name = fs::path(file_prefix).filename();
    fs::path prefix_path = workdir / (prefix_name.string() + "_rank_" + std::to_string(rank));

    fs::create_directories(fs::path(prefix_path.string() + ".tmp"));

    kwargs["prefix"]  = prefix_path.string();
    kwargs["verbose"] = 0;
    kwargs["logIni"]  = logini_path.string();

    // kwargs["coordsys"] = ...;
    kwargs["maxiter"] = 300;
    // kwargs["transition"] = ...;
    // kwargs["hessian"] = ...;
    // kwargs["constraints"] = ...;
    // kwargs["conmethod"] = ...;
    // kwargs["irc"] = ...;
    // kwargs["irc_direction"] = ...;
    // kwargs["trust"] = ...;
    // kwargs["tmax"] = ...;
    // kwargs["tmin"] = ...;
    // kwargs["qccnv"] = ...;
    // kwargs["molcnv"] = ...;
    // kwargs["write_cart_hess"] = ...;
    // kwargs["frequency"] = ...;
    // kwargs["bothre"] = ...;
    // kwargs["subfrctor"] = ...;
    // kwargs["rigid"] = ...;

    py::object progress;

    if(rank == 0) { progress = optimize_m.attr("run_optimizer")(**kwargs); }
    else {
      py::module_ sys = py::module_::import("sys");
      py::module_ os  = py::module_::import("os");

      py::object old_stdout = sys.attr("stdout");
      py::object old_stderr = sys.attr("stderr");
      py::object devnull =
        os.attr("fdopen")(os.attr("open")(os.attr("devnull"), os.attr("O_WRONLY")), "w");

      try {
        sys.attr("stdout") = devnull;
        sys.attr("stderr") = devnull;
        progress           = optimize_m.attr("run_optimizer")(**kwargs);
      } catch(...) {
        try {
          sys.attr("stdout") = old_stdout;
          sys.attr("stderr") = old_stderr;
          devnull.attr("close")();
        } catch(...) {}
        throw;
      }

      sys.attr("stdout") = old_stdout;
      sys.attr("stderr") = old_stderr;
      devnull.attr("close")();
    }

    py::sequence  xyzs    = progress.attr("xyzs").cast<py::sequence>();
    const ssize_t nframes = py::len(xyzs);
    if(nframes < 1) { throw std::runtime_error("geomeTRIC returned no optimized geometries"); }

    py::array final_xyz_ang = xyzs[py::int_(nframes - 1)].cast<py::array>();
    auto      buf           = final_xyz_ang.request();

    if(buf.ndim != 2 || static_cast<size_t>(buf.shape[0]) != atoms.size() || buf.shape[1] != 3) {
      throw std::runtime_error("Unexpected final geometry shape from geomeTRIC trajectory");
    }

    const auto*        ptr = static_cast<const double*>(buf.ptr);
    Eigen::RowVectorXd final_geom(3 * atoms.size());
    for(size_t i = 0; i < atoms.size(); ++i) {
      final_geom(3 * i + 0) = ptr[3 * i + 0] * exachem::constants::ang2bohr;
      final_geom(3 * i + 1) = ptr[3 * i + 1] * exachem::constants::ang2bohr;
      final_geom(3 * i + 2) = ptr[3 * i + 2] * exachem::constants::ang2bohr;
    }

    chem_env.update_geometry(atoms, ec_atoms, final_geom);
    chem_env.atoms    = atoms;
    chem_env.ec_atoms = ec_atoms;

    if(ec.print()) {
      std::cout << std::endl << std::setw(34) << "Optimized geometry" << std::endl;
      exachem::geometry::print_geometry(ec, chem_env);
    }
  } catch(const py::error_already_set& e) {
    throw std::runtime_error(std::string("Python/geomeTRIC error: ") + e.what());
  }
}

} // namespace exachem::optimizers
