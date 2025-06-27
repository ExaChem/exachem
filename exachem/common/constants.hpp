/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

namespace exachem::constants {
inline constexpr double bohr2ang = 0.529177210544;
inline double ang2bohr = 1.8897261259077822; // modified if user provides ang2au in input file
} // namespace exachem::constants
