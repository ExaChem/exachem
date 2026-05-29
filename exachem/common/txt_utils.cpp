/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/txt_utils.hpp"
#include <iostream>

std::string txt_utils::to_upper(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return str;
}

std::string txt_utils::to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str;
}

bool txt_utils::strequal_case(const std::string& a, const std::string& b) {
  return a.size() == b.size() and
         std::equal(a.begin(), a.end(), b.begin(), [](unsigned char a, unsigned char b) {
           return std::tolower(a) == std::tolower(b);
         });
}
