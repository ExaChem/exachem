
#include "libint2_includes.hpp"
#include "tamm/tamm.hpp"
#include "txtutils.hpp"
#include <iostream>
#include <vector>
using namespace tamm;
using libint2::Atom;
class ECAtom {
public:
  Atom                atom;
  std::string         esymbol;
  std::string         basis;
  bool                has_ecp{false};
  int                 ecp_nelec{};
  std::vector<double> ecp_coeffs{};
  std::vector<double> ecp_exps{};
  std::vector<int>    ecp_ams{};
  std::vector<int>    ecp_ns{};
  static int          get_atomic_number(std::string element_symbol);
};
