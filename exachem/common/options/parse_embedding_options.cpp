/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_embedding_options.hpp"

ParseEmbeddingOptions::ParseEmbeddingOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseEmbeddingOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseEmbeddingOptions::parse_check(json& jinput) {
  if(!jinput.contains("EMBEDDING")) return;
  // clang-format off
  const std::vector<std::string> valid_embedding{"high_level", "active_atoms", "lambda",
     "partition", "nactive_mos", "freeze_projected", "projector", "iterative_vembedding",
     "pao_thresh1", "pao_thresh2", "use_ksref"};
  // clang-format on

  for(auto& el: jinput["EMBEDDING"].items()) {
    if(std::find(valid_embedding.begin(), valid_embedding.end(), el.key()) == valid_embedding.end())
      tamm_terminate("INPUT FILE ERROR: Invalid EMBEDDING option [" + el.key() +
                     "] in the input file");
  }
}

void ParseEmbeddingOptions::parse(ChemEnv& chem_env) {
  if(!chem_env.jinput.contains("EMBEDDING")) return;
  json              jemb              = chem_env.jinput["EMBEDDING"];
  EmbeddingOptions& embedding_options = chem_env.ioptions.embedding_options;

  parse_option<std::vector<std::string>>(embedding_options.high_level, jemb, "high_level");
  parse_option<std::string>(embedding_options.partition, jemb, "partition");
  parse_option<std::string>(embedding_options.projector, jemb, "projector");
  parse_option<std::vector<int>>(embedding_options.active_atoms, jemb, "active_atoms");
  parse_option<std::vector<int>>(embedding_options.nactive_orbitals, jemb, "nactive_mos");
  parse_option<double>(embedding_options.lambda, jemb, "lambda");
  parse_option<double>(embedding_options.pao_thresh1, jemb, "pao_thresh1");
  parse_option<double>(embedding_options.pao_thresh2, jemb, "pao_thresh2");
  parse_option<bool>(embedding_options.use_ksref, jemb, "use_ksref");
  parse_option<bool>(embedding_options.freeze_projected, jemb, "freeze_projected");
  parse_option<bool>(embedding_options.iterative_vembedding, jemb, "iterative_vembedding");

  bool                           found_ks  = false;
  bool                           found_wft = false;
  const std::vector<std::string> dft{"PBE", "SCAN", "R2SCAN", "BLYP", "PBE0"};
  for(auto& ihigh: embedding_options.high_level) {
    std::transform(ihigh.begin(), ihigh.end(), ihigh.begin(), ::toupper);
    if(ihigh.rfind("XC_", 0) == 0) { found_ks = true; }
    else if(std::find(dft.begin(), dft.end(), ihigh) == dft.end()) { found_wft = true; }
    else { found_ks = true; }
  }
  if(found_ks && found_wft) {
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option high_level cannot mix DFT and WFT");
  }
  else if(found_wft && embedding_options.high_level.size() > 1) {
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option high_level cannot mix different WFTs");
  }

  if(embedding_options.lambda < 0.0) {
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option lambda should be greater than 0.0");
  }

  if(embedding_options.active_atoms.empty()) {
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option active_atoms is empty");
  }

  std::string& projector = embedding_options.projector;
  std::transform(projector.begin(), projector.end(), projector.begin(), ::toupper);
  const std::vector<std::string> projector_valid{"HUZINAGA", "LEVEL"};
  if(std::find(projector_valid.begin(), projector_valid.end(), projector) == projector_valid.end())
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option projector is not HUZINAGA or LEVEL");

  std::string& partition = embedding_options.partition;
  std::transform(partition.begin(), partition.end(), partition.begin(), ::toupper);
  const std::vector<std::string> partition_valid{"SPADE"};
  if(std::find(partition_valid.begin(), partition_valid.end(), partition) == partition_valid.end())
    tamm_terminate("INPUT FILE ERROR: EMBEDDING option partition is not SPADE");

  for(auto& i_acc_mos: embedding_options.nactive_orbitals) {
    if(i_acc_mos < 1)
      tamm_terminate("INPUT FILE ERROR: EMBEDDING option nactive_mos should be > 0");
  }
}
