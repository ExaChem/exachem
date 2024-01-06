#pragma once

// clang-format off
#include <any>
#include <cctype>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
#include "txtutils.hpp"
using namespace tamm;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

#include <nlohmann/json.hpp>
#include "libint2_includes.hpp"
#include "ecatom.hpp"

using json = nlohmann::ordered_json;
// clang-format on

template<typename T>
void parse_option(T& val, json j, std::string key, bool optional = true) {
  if(j.contains(key)) val = j[key].get<T>();
  else if(!optional) {
    tamm_terminate("INPUT FILE ERROR: " + key + " not specified. Please specify the " + key +
                   " option!");
  }
}

class ECParse {
public:
  std::vector<string> geometry;
  std::vector<ECAtom> ec_atoms;
  std::vector<Atom>   atoms;
  std::vector<string> geom_bohr;
  const double        angstrom_to_bohr = 1.8897259878858;
  json                jinput;
  std::string         geom_units;
  ECParse(std::string_view filename);
  void               convertUnits();
  static void        check_json(std::string filename);
  static json        json_from_file(std::string jfile);
  static void        json_to_file(json jdata, std::string jfile);
  static std::string getfilename(std::string filename);
};