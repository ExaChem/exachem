"""
ExaChem: Open Source Exascale Computational Chemistry Software.

Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.

See LICENSE.txt for details
"""

"""
Parses a single xyz file or a folder with multiple xyz files and returns ExaChem input (json) file(s).

Args:
  input_file: The path to a single xyz file (OR) path to a folder contatining xyz files.
  optional_args: units, task-name, operation

Returns:
  json input file(s) for ExaChem.
"""

import os
import sys
import json
from pathlib import Path 


def get_next_word(full_string, target_string, all_rem_words=False):
    words = full_string.split()
    tloc=-1
    for i,w in enumerate(words):
      if w==target_string: 
        tloc=i
        break

    if tloc==-1: return None
    if all_rem_words: return words[tloc+1:]
    else: return words[tloc+1]   

def parse_xyz_file(input_file):

  lines = []
  with open(input_file, 'r') as f:
    lines = list(line for line in (l.strip() for l in f) if line)   

  geometry = []
  natoms = int(lines[0])
  charge = 0
  mult = 1
  cline = (lines[1]).lower()
  cline_exists = False

  if "charge" in cline or "mult" in cline: cline_exists = True
  clist = cline.split(" ")
  clist = [i for i in clist if i]
  
  if len(clist) == 4 and cline_exists:
    clist = clist[1:]
    clist = [x.replace(".", "") for x in clist]
    clist = [x.replace("-", "") for x in clist]
    clist = [x.replace("+", "") for x in clist]
    cline_exists = not all(x.isnumeric() for x in clist)

  glno = 1

  if cline_exists:
    glno = 2
    if "charge" in cline:
      charge = int(get_next_word(cline,"charge").strip(";"))
    if "mult" in cline:
      mult = int(get_next_word(cline,"mult").strip(";"))
  
  # print("natoms = %s" %(natoms))

  # Parse the geometry.
  for line in lines[glno:]:
    geometry.append(line)

  assert(natoms == len(geometry))
  return [natoms,charge,mult,geometry]


def dict_to_json(dictname):
  json_objects = json.dumps(dictname, indent = 4) 
  # print(json_objects)
  return json_objects

if __name__ == '__main__':
  input_fpath = sys.argv[1]
  input_fpath = os.path.abspath(input_fpath)

  if not os.path.exists(input_fpath): 
      print("ERROR: [" + input_fpath + "] does not exist!")
      sys.exit(1)

  taskname = "sinfo"
  units = "angstrom"
  if len(sys.argv) >= 3: units = sys.argv[2]
  if len(sys.argv) >= 4: taskname = sys.argv[3]
  if len(sys.argv) == 5: operation = sys.argv[4]
  else: operation = ["energy"]

  input_files = []

  if os.path.isdir(input_fpath):
    cwd = os.path.abspath(input_fpath)
    jwd = cwd+"/json"
    if not os.path.exists(jwd): os.mkdir(jwd)

    for _, _, files in os.walk(cwd):
      for file in files:
        if file.endswith(".xyz"): input_files.append(file) 

  elif os.path.isfile(input_fpath):
    cwd = os.path.dirname(input_fpath)
    jwd = cwd    
    input_files.append(input_fpath)
  
  os.chdir(cwd)

  # print(input_files)
     
  for input_file in input_files:

    file_prefix = Path(input_file).stem

    natoms,charge,mult,geometry = parse_xyz_file(input_file)

    exachem_opt = {}

    exachem_opt["geometry"] = { "coordinates": geometry, "units": units}

    exachem_opt["common"] = {}
    exachem_opt["common"]["maxiter"] = 50
    exachem_opt["basis"] = {"basisset":"cc-pvdz"}

    scf_type = "unrestricted"
    if charge == 0 and mult==1: scf_type = "restricted"

    exachem_opt["SCF"] = {}
    scf_opt = exachem_opt["SCF"]
    scf_opt["conve"] = 1e-8
    scf_opt["convd"] = 1e-7
    scf_opt["diis_hist"] = 10
    scf_opt["charge"] = charge
    scf_opt["multiplicity"] = mult
    scf_opt["lshift"] = 0
    scf_opt["damp"] = 100
    scf_opt["scf_type"] = scf_type
    scf_opt["debug"] = False
    scf_opt["restart"] = False
    scf_opt["noscf"] = False

    exachem_opt["CC"] = {}
    cc_opt = exachem_opt["CC"]
    cc_opt["ndiis"] = 5
    cc_opt["lshift"] = 0
    cc_opt["threshold"] = 1e-6
    cc_opt["ccsd_maxiter"] = 50
    cc_opt["writet"] = False

    cc_opt["CCSD(T)"] = {}
    ccsd_pt = cc_opt["CCSD(T)"]
    ccsd_pt["cache_size"] = 8
    ccsd_pt["skip_ccsd"] = False
    ccsd_pt["ccsdt_tilesize"] = 40

    exachem_opt["TASK"] = {}
    exachem_opt["TASK"][taskname] = True
    exachem_opt["TASK"]["operation"] = [operation]

    exachem_opt = dict_to_json(exachem_opt)

    ec_json_file = jwd + "/" + file_prefix + ".json"
    with open(ec_json_file, "w") as ecfile:
      ecfile.write(exachem_opt)
