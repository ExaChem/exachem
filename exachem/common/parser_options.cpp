#include "parser_options.hpp"
ECParse::ECParse(std::string_view filename) {
  check_json(std::string(filename));
  auto is = std::ifstream(std::string(filename));

  // auto jsax         = ECParse::json_sax_no_exception(jinput);
  auto jsax         = nlohmann::detail::json_sax_dom_parser<json>(jinput, false);
  bool parse_result = json::sax_parse(is, &jsax);
  if(!parse_result) tamm_terminate("Error parsing input file");

  parse_option<string>(geom_units, jinput["geometry"], "units");
  parse_option<std::vector<string>>(geometry, jinput["geometry"], "coordinates", false);
  size_t natom = geometry.size();

  ec_atoms.resize(natom);
  atoms.resize(natom);
  geom_bohr.resize(natom);

  for(size_t i = 0; i < natom; i++) {
    std::string        line = geometry[i];
    std::istringstream iss(line);
    std::string        element_symbol;
    double             x, y, z;
    iss >> element_symbol >> x >> y >> z;
    geom_bohr[i] = element_symbol;

    const auto Z = ECAtom::get_atomic_number(element_symbol);

    atoms[i].atomic_number = Z;
    atoms[i].x             = x;
    atoms[i].y             = y;
    atoms[i].z             = z;

    ec_atoms[i].atom    = atoms[i];
    ec_atoms[i].esymbol = element_symbol;
  }
  convertUnits();
} // end of parse

void ECParse::convertUnits() {
  const double angstrom_to_bohr = 1.8897259878858;
  // json         jgeom_bohr;
  bool nw_units_bohr = true;
  // If geometry units specified are angstrom, convert to bohr
  if(geom_units == "angstrom") nw_units_bohr = false;

  if(!nw_units_bohr) {
    // .xyz files report Cartesian coordinates in angstroms;
    // convert to bohr
    for(auto i = 0U; i < atoms.size(); i++) {
      std::ostringstream ss_bohr;
      atoms[i].x *= angstrom_to_bohr;
      atoms[i].y *= angstrom_to_bohr;
      atoms[i].z *= angstrom_to_bohr;
      ss_bohr << std::setw(3) << std::left << geom_bohr[i] << " " << std::right << std::setw(14)
              << std::fixed << std::setprecision(10) << atoms[i].x << " " << std::right
              << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].y << " "
              << std::right << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].z;
      geom_bohr[i]     = ss_bohr.str();
      ec_atoms[i].atom = atoms[i];
    }
    // jgeom_bohr["geometry_bohr"] = geom_bohr;
  }
} // end of convertUnits

//     static nlohmann::detail::json_sax_dom_parser<json> ECParse::json_sax_no_exception(json& j) {
//     return nlohmann::detail::json_sax_dom_parser<json>(j, false);
//   } // END of json_sax_no_exception

void ECParse::check_json(std::string filename) {
  namespace fs        = std::filesystem;
  std::string get_ext = fs::path(filename).extension();
  const bool  is_json = (get_ext == ".json");
  if(!is_json) tamm_terminate("ERROR: Input file provided [" + filename + "] must be a json file");
} // END of static inline void check_json(std::string filename)

json ECParse::json_from_file(std::string jfile) {
  json jdata;
  check_json(jfile);

  auto is = std::ifstream(jfile);

  auto jsax         = nlohmann::detail::json_sax_dom_parser<json>(jdata, false);
  bool parse_result = json::sax_parse(is, &jsax);
  if(!parse_result) tamm_terminate("Error parsing file: " + jfile);

  return jdata;
} // END of static inline json json_from_file(std::string jfile)

void ECParse::json_to_file(json jdata, std::string jfile) {
  std::ofstream res_file(jfile);
  res_file << std::setw(2) << jdata << std::endl;
} // END of static inline void json_to_file(json jdata, std::string jfile)

std::string ECParse::getfilename(std::string filename) {
  size_t lastindex = filename.find_last_of(".");
  auto   fname     = filename.substr(0, lastindex);
  return fname.substr(fname.find_last_of("/") + 1, fname.length());
} // END of static inline std::string getfilename(std::string filename)