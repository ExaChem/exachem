/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/txt_utils.hpp"
#include <iostream>

void txt_utils::to_upper(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
}

void txt_utils::to_lower(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

std::string txt_utils::str_upper(const std::string str) {
  std::string ustr = str;
  std::transform(ustr.begin(), ustr.end(), ustr.begin(), ::toupper);
  return ustr;
}

bool txt_utils::strequal_case(const std::string& a, const std::string& b) {
  return a.size() == b.size() and
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const char a, const char b) { return std::tolower(a) == std::tolower(b); });
}

void txt_utils::print_bool(const std::string str, bool val) {
  std::cout << str << " = " << std::boolalpha << val << std::endl;
}