/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <string>

#include <cctype> // transform

#include <algorithm> // for ::toupper and ::tolower

class txtutils {
public:
  inline void        to_upper(std::string& str);
  inline void        to_lower(std::string& str);
  inline std::string str_upper(const std::string str);
  inline bool        strequal_case(const std::string& a, const std::string& b);
  inline void        print_bool(const std::string str, bool val);
};

void txtutils::to_upper(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
}

void txtutils::to_lower(std::string& str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

std::string txtutils::str_upper(const std::string str) {
  std::string ustr = str;
  std::transform(ustr.begin(), ustr.end(), ustr.begin(), ::toupper);
  return ustr;
}

bool txtutils::strequal_case(const std::string& a, const std::string& b) {
  return a.size() == b.size() and
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const char a, const char b) { return std::tolower(a) == std::tolower(b); });
}

void txtutils::print_bool(const std::string str, bool val) {
  std::cout << str << " = " << std::boolalpha << val << std::endl;
}