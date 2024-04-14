#include "common/chemenv.hpp"
#include "common/options/parser_utils.hpp"

class ParseCOptions: public ParserUtils {
private:
  void parse(ChemEnv& chem_env);

public:
  ParseCOptions() = default;
  ParseCOptions(ChemEnv& chem_env);
  void operator()(ChemEnv& chem_env);
  void print(ChemEnv& chem_env);
};
