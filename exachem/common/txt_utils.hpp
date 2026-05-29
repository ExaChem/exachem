/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm> // for ::toupper and ::tolower
#include <cctype>    // transform
#include <string>

namespace txt_utils {
std::string to_upper(std::string str);
std::string to_lower(std::string str);
bool        strequal_case(const std::string& a, const std::string& b);
} // namespace txt_utils
