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
#include <nlohmann/json.hpp>

#include "common/libint2_includes.hpp"
#include "common/ecatom.hpp"
#include "common/txt_utils.hpp"
#include "common/chemenv.hpp"

using namespace tamm;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using json = nlohmann::ordered_json;
// clang-format on

class ParserUtils {
public:
  template<typename T>
  void parse_option(T& val, json j, std::string key, bool optional = true) {
    if(j.contains(key)) val = j[key].get<T>();
    else if(!optional) {
      tamm_terminate("INPUT FILE ERROR: " + key + " not specified. Please specify the " + key +
                     " option!");
    }
  }

  static std::string getfilename(std::string filename) {
    size_t lastindex = filename.find_last_of(".");
    auto   fname     = filename.substr(0, lastindex);
    return fname.substr(fname.find_last_of("/") + 1, fname.length());
  }

  static void json_to_file(json jdata, std::string jfile) {
    std::ofstream res_file(jfile);
    res_file << std::setw(2) << jdata << std::endl;
  }

  static json json_from_file(std::string jfile) {
    json jdata;
    auto check_json = [](const std::string& filename) {
      if(std::filesystem::path(filename).extension() != ".json") {
        tamm_terminate("ERROR: Input file provided [" + filename + "] must be a json file");
      }
    };
    check_json(jfile);

    auto is = std::ifstream(jfile);

    auto jsax         = nlohmann::detail::json_sax_dom_parser<json>(jdata, false);
    bool parse_result = json::sax_parse(is, &jsax);
    if(!parse_result) tamm_terminate("Error parsing file: " + jfile);

    return jdata;
  }
};
