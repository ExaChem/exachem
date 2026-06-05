
include(TargetMacros)

set(GEOMETRY_SRCDIR geometry)

set(GEOMETRY_INCLUDES
    ${GEOMETRY_SRCDIR}/geometry_analysis.hpp
    ${GEOMETRY_SRCDIR}/internal_coordinates.hpp
    )

set(GEOMETRY_SRCS
    ${GEOMETRY_SRCDIR}/geometry_analysis.cpp
    ${GEOMETRY_SRCDIR}/internal_coordinates.cpp
   )
