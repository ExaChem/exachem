/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm> // for ::toupper and ::tolower
#include <cctype>    // transform
#include <string>

namespace txt_utils {
void        to_upper(std::string& str);
void        to_lower(std::string& str);
std::string str_upper(const std::string str);
bool        strequal_case(const std::string& a, const std::string& b);
void        print_bool(const std::string str, bool val);
} // namespace txt_utils
