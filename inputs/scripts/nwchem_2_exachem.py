"""
ExaChem: Open Source Exascale Computational Chemistry Software.

Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.

See LICENSE.txt for details
"""

"""
Parses an NWChem input file and returns an ExaChem input (json) file

Args:
  input_file: The path to the NWChem input file.

Returns:
  A (json) input file for ExaChem.
"""

import re
import sys
import json
from pathlib import Path

def remove_words(full_string, remove_list):
  words = full_string.split()
  new_str = []
  for w in words:
    if not w in remove_list: new_str.append(w)

  return ' '.join(new_str)

def get_first_word(full_string):
    words = full_string.split()
    return words[0].strip()

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

def parse_nwchem_input(input_file):
  nwchem_opt = {}
  unsupported = []
  unsupported_rest = []
  geom_block=False

  with open(input_file, 'r') as f:
    # Parse the geometry.
    geometry = []

    nwchem_opt['start'] = None

    for line in f:
      line=line.strip()
      oline=line
      line=line.lower()
      
      if line.startswith('start'):
        nwchem_opt['start'] = get_next_word(line,"start")

      if line.startswith('geometry'):
        geom_block=True
        #TODO: assume units show up in the same line if present
        geom_units="angstrom"
        if "units" in line: geom_units = get_next_word(line,"units")
        nwchem_opt['geom_units'] = geom_units
        ugeom = remove_words(line,["units",geom_units,"geometry"])
        if ugeom: unsupported.append("geometry keywords: %s are not supported (ignored)" %(ugeom))
        continue      
      if geom_block and line.startswith('end'): break
      if not geom_block: continue
      
      #symmetry [group] <name>
      nwchem_opt['symmetry']="c1"
      if line.startswith('symmetry'): 
        sym_grp = get_next_word(line,"symmetry")
        if "group" in line: sym_grp = get_next_word(line,"group")
        nwchem_opt['symmetry']=sym_grp
        continue

      geometry.append(oline)

  nwchem_opt['geometry'] = geometry

  # Parse rest of the input file

  atom_except_list=[]
  atom_basis={}
  geom_block=False
  basis_block=False
  scf_block=False
  dft_block=False
  tce_block=False

  nwchem_opt["TASK"] = {}
  nwchem_opt["task_tce"] = False
  nw_task = nwchem_opt["TASK"]
  # nwchem_opt["gaussian_type"] = "spherical"
  nwchem_opt["basisset"] = "sto-3g"

  nwchem_opt["SCF"] = {}
  nwchem_opt["SCF"]["PRINT"] = {}
  scf_opt = nwchem_opt["SCF"]
  scf_opt["tol_int"] = 1e-12
  scf_opt["tol_lindep"] = 1e-5
  scf_opt["conve"] = 1e-8
  scf_opt["convd"] = 1e-7
  scf_opt["diis_hist"] = 10
  scf_opt["multiplicity"] = 1
  scf_opt["lshift"] = 0
  scf_opt["alpha"] = 1.0
  scf_opt["type"] = "restricted"
  scf_opt["maxiter"] = 50
  scf_opt["PRINT"]["mulliken"] = False
  scf_opt["PRINT"]["mo_vectors"] = [False,0.15]

  #DFT    
  nwchem_opt["DFT"] = {}
  nwchem_opt["DFT"]["PRINT"] = {}
  dft_opt = nwchem_opt["DFT"]
  dft_opt["hfexch"] = False
  dft_opt["tol_int"] = 1e-12
  dft_opt["tol_lindep"] = 1e-5
  dft_opt["conve"] = 1e-8
  dft_opt["convd"] = 1e-7
  dft_opt["diis_hist"] = 10
  dft_opt["multiplicity"] = 1
  dft_opt["lshift"] = 0
  dft_opt["alpha"] = 1.0
  dft_opt["type"] = "restricted"
  dft_opt["maxiter"] = 50
  dft_opt["PRINT"]["mulliken"] = False
  dft_opt["PRINT"]["mo_vectors"] = [False,0.15]

  nwchem_opt["TCE"] = {}
  nwchem_opt["TCE"]["PRINT"] = {}
  tce_opt = nwchem_opt["TCE"]
  tce_opt["threshold"] = 1e-6
  tce_opt["ndiis"] = 5
  tce_opt["lshift"] = 0 
  tce_opt["ccsd_maxiter"] = 50
  tce_opt["freeze_core"] = 0 
  tce_opt["freeze_virtual"] = 0 

  with open(input_file, 'r') as f:
    #Collect all tasks
    for line in f:
      line=line.strip()
      line=line.lower()

      if line.startswith('#'): continue

      if line.startswith('task'):
        if 'property' in line: unsupported.append("property keyword is unsupported in the task line: %s" %(line))

        tname = get_next_word(line,'task')
        if tname=="scf" in line: nw_task["scf"] = True
        #TODO
        elif tname=="dft" in line: nw_task["scf"] = True
        elif tname=="mp2" in line: nw_task["mp2"] = True
        elif tname=="tce": 
          nw_task["scf"] = False
          nwchem_opt["task_tce"] = True
        else: unsupported.append("unsupported task line specified: %s" %(line))

    #Parse basis,scf,tce blocks
    f.seek(0)
    for line in f:
      oline = line
      line=line.strip()
      line=line.lower()

      if line.startswith('#'): continue
      #TODO: can charge be free floating anywhere? any other places using "charge" keyword?
      if line.startswith("charge"):
        nwchem_opt["SCF"]["charge"] = int(get_next_word(line,"charge"))
        continue

      #Skip geometry section
      if geom_block and line.startswith('end'):
        geom_block=False
        continue
      if geom_block: continue
      if line.startswith('geometry'):
        geom_block=True
        continue
      
      #parse BASIS block
      #TODO Not parsed: string name, basis file paths, REL, BSE, full symbol name, basis dumps
      if line.startswith("basis"):
        basis_block=True
        if "cartesian" in line:
          # nwchem_opt["gaussian_type"] = "cartesian"
          unsupported.append("cartesian not supported (ignored), spherical will be used")
        continue

      if basis_block and "library" in line:
        if line.startswith("#"): continue
        elif line.startswith("*"):
          nwchem_opt["basisset"] = get_next_word(line,"library")
          if "except" in line:
            atom_except_list = get_next_word(line,"except",True)
        else: 
          atom_symbol = get_first_word(line)
          atom_basis[atom_symbol] = get_next_word(line,"library")
        continue

      if basis_block and line.startswith('end'):
        basis_block=False
        if atom_basis: nwchem_opt["atom_basis"] = atom_basis
        continue

      #SCF options
      if line.startswith("scf"):
        scf_block=True
        continue

      if scf_block:
        if line.startswith('end'):
          scf_block=False
          continue
        if 'thresh' in line:
          conve = float(get_next_word(line,"thresh"))
          scf_opt["conve"] = conve
          #TODO: no convd in nwchem ?
          scf_opt["convd"] = "{:.1e}".format(conve * 10.0)
        if 'tol2e' in line:
          scf_opt["tol_int"] = float(get_next_word(line,"tol2e"))
        if 'uhf' in line:
          scf_opt["type"] = "unrestricted"
        elif 'rohf' in line: 
          unsupported.append("ROHF not supported (will use UHF)")
          scf_opt["type"] = "unrestricted"
        if 'singlet' in line: None
        if 'doublet' in line: scf_opt["multiplicity"] = 2
        if 'triplet' in line: scf_opt["multiplicity"] = 3
        if 'quartet' in line: scf_opt["multiplicity"] = 4
        if 'quintet' in line: scf_opt["multiplicity"] = 5
        if 'sextet'  in line: scf_opt["multiplicity"] = 6
        if 'septet'  in line: scf_opt["multiplicity"] = 7
        if 'octet'   in line: scf_opt["multiplicity"] = 8
        if 'nopen'   in line: scf_opt["multiplicity"] = int(get_next_word(line,"nopen"))
        if 'maxiter' in line: scf_opt["maxiter"]      = int(get_next_word(line,"maxiter"))

        if 'level' in line:
          if 'nr' in line: scf_opt["lshift"] = float(get_next_word(line,"nr"))
          if "pcg" in line: unsupported.append("SCF level shift pcg option not supported (ignored)")

        if 'mulliken' in line: 
          if not "noprint" in line: scf_opt["PRINT"]["mulliken"] = True
        if "final vectors analysis" in line: 
          if not "noprint" in line: scf_opt["PRINT"]["mo_vectors"] = [True,0.15]
        continue

      #DFT options
      if line.startswith("dft"):
        dft_block=True
        continue

      if dft_block:
        if line.startswith('end'):
          dft_block=False
          continue
        if line.startswith("xc") and "hfexch" in line: dft_opt["hfexch"] = True
        if 'convergence' in line:
          if 'damp' in line: dft_opt["alpha"]    = float(100-int(get_next_word(line,"damp")))/100
          if 'lshift' in line: dft_opt["lshift"]  = float(get_next_word(line,"lshift"))

          if 'energy' in line:
            conve = float(get_next_word(line,"energy"))
            dft_opt["conve"] = conve
            #TODO: no convd in nwchem ?
            dft_opt["convd"] = "{:.1e}".format(conve * 10.0)
          if 'density' in line:
            dft_opt["convd"] = float(get_next_word(line,"density"))
          if 'diis' in line:
            dft_opt["diis_hist"] = int(get_next_word(line,"ndiis"))

          if 'acccoul' in line:
            dft_opt["tol_int"] = float(get_next_word(line,"acccoul"))

        #TODO
        # elif 'rohf' in line: 
        #   unsupported.append("ROHF not supported (will use UHF)")
        #   dft_opt["type"] = "unrestricted"

        if 'mult' in line and line.startswith("mult"): 
          mult = int(get_next_word(line,"mult"))
          dft_opt["multiplicity"] = mult
          if mult > 1: dft_opt["type"] = "unrestricted"
        if 'maxiter' in line: dft_opt["maxiter"]   = int(get_next_word(line,"maxiter"))

        if 'mulliken' in line: 
          if not "noprint" in line: dft_opt["PRINT"]["mulliken"] = True
        if "final vectors analysis" in line: 
          if not "noprint" in line: dft_opt["PRINT"]["mo_vectors"] = [True,0.15]
        continue        

      #TCE options
      if line.startswith("tce"):
        tce_block=True
        continue

      if tce_block:
        if line.startswith('end'):
          tce_block=False
          continue
        if 'thresh' in line:
          tce_opt["threshold"] = float(get_next_word(line,"thresh"))
        if 'diis' in line:
          tce_opt["ndiis"] = int(get_next_word(line,"ndiis"))
        if 'lshift' in line:
          tce_opt["lshift"] = float(get_next_word(line,"lshift"))
        if 'maxiter' in line:
          tce_opt["ccsd_maxiter"] = int(get_next_word(line,"maxiter"))

        #TODO: double-check freeze syntax
        #freeze atomic, freeze core 10 (or freeze 10), freeze virtual 5
        if "freeze" in line:
          if "atomic" in line: unsupported.append("tce freeze atomic not supported (ignored)")
          elif "core" in line: tce_opt["freeze_core"] = int(get_next_word(line,"core"))
          elif "virtual" in line: tce_opt["freeze_virtual"] = int(get_next_word(line,"virtual"))
          else: tce_opt["freeze_core"] = int(get_next_word(line,"freeze"))

        #TODO: eomccsd
        if nwchem_opt["task_tce"]:
          if 'cc2' in line: nw_task["cc2"] = True
          elif 'mp2' in line: nw_task["mp2"] = True
          elif 'ccsd' in line or 'icsd' in line: nw_task["ccsd"] = True
          elif 'ccsd(t)' in line: nw_task["ccsd_t"] = True
          elif 'dipole' in line: nw_task["ccsd_lambda"] = True

        continue

      # match = re.match(r'^(.*)\s+(.*)$', line)
      # if match:
      #   key, value = match.groups()
      #   nwchem_opt[key] = value
      if line: unsupported_rest.append(oline)

  return [nwchem_opt,unsupported,unsupported_rest]


def dict_to_json(dictname):
  json_objects = json.dumps(dictname, indent = 4) 
  print(json_objects)
  return json_objects

if __name__ == '__main__':
  input_file = sys.argv[1]
  file_prefix = Path(input_file).stem
  nwchem_opt,unsupported,unsupported_rest = parse_nwchem_input(input_file)
  # dict_to_json(nwchem_opt)

  exachem_opt = {}

  """
    Common block

    Top-level directives
    start file_prefix -> output_file_prefix

    Geometries
      geometry units <angstroms or an | au or atomic or bohr>

    Basis Sets
      basis name (spherical || cartesian) default cartesian
      *  library 3-21g <except atom1>
      atom1 symbol library basis-name
      ...
      atomN symbol library basis-name


    Ignore:
      permanent_dir, memory line
      geometry name center autoz autosym <symmetry groupname> <zmatrix|zcoord|system>
      geometry load xyz from input file
  """

  #Check for geometry units
  geom_units = nwchem_opt["geom_units"]
  if "angstrom"==geom_units or "angstroms"==geom_units or "an"==geom_units: geom_units = "angstrom"
  elif "au"==geom_units or "atomic"==geom_units or "bohr"==geom_units: geom_units = "bohr"
  else: 
    sym_msg="geometry units %s are not supported. Only bohr and angstrom are allowed (defaulting to angstrom)" %(geom_units)
    geom_units="angstrom"
    unsupported.append(sym_msg)

  exachem_opt["geometry"] = { "coordinates": nwchem_opt["geometry"], "units": geom_units}

  #check for C1 symmetry 
  nwsym=nwchem_opt["symmetry"]
  sym_msg="symmetry %s is not supported. Only C1 symmetry is allowed (defaulting to C1)." %(nwsym)
  if nwsym != "c1": unsupported.append(sym_msg)

  #start directive
  if not nwchem_opt["start"]: nwchem_opt["start"] = file_prefix
  exachem_opt["common"] = { "output_file_prefix": nwchem_opt["start"] }

  #BASIS
  #TODO: atom basis is not parsed 
  exachem_opt["basis"] = {"basisset":nwchem_opt["basisset"]}

  if "atom_basis" in nwchem_opt.keys():
    exachem_opt["basis"]["atom_basis"] = nwchem_opt["atom_basis"]


  """
    SCF block

    charge N
    maxiter 50
    SINGLET, DOUBLET, TRIPLET, QUARTET, QUINTET, SEXTET, SEPTET, OCTET (mult=1-8)
    uhf, rhf
    thresh 1e-4 (convergenece thershold)
    TOL2E <real tol2e default min(10e-7 , 0.01*thresh)>
    task scf
    set tolguess 1e-7
    LEVEL: level-shifting the orbital Hessian
    level pcg 20 0.3 0.0 nr 0.2 0.005 0.0
      - we only check for level nr 0.2

    PRINT: mulliken ao, final vectors analysis

    Ignore:
      ROHF
      DIIS 
      DIRECT, SEMIDIRECT
      NR: Newton-Raphson
      SYM <string (ON||OFF) default ON>
      ADAPT <string (ON||OFF) default ON>
      orbital localization (we can port this from dlpno code later)
      gradients
      VECTORS [[input] ....
  """

  exachem_opt["SCF"] = {}
  exachem_opt["SCF"]["PRINT"] = {}
  ec_scf = exachem_opt["SCF"]

  scf_opt = nwchem_opt["SCF"]
  ec_scf["charge"] = nwchem_opt["SCF"]["charge"]
  ec_scf["multiplicity"] = scf_opt["multiplicity"]
  ec_scf["lshift"] = scf_opt["lshift"]
  ec_scf["tol_int"] = scf_opt["tol_int"]
  ec_scf["tol_lindep"] = scf_opt["tol_lindep"]
  ec_scf["conve"] = scf_opt["conve"]
  ec_scf["convd"] = scf_opt["convd"]
  ec_scf["alpha"] = scf_opt["alpha"]
  ec_scf["scf_type"] = scf_opt["type"]
  ec_scf["diis_hist"] = scf_opt["diis_hist"]
  exachem_opt["common"]["maxiter"] = scf_opt["maxiter"]
  ec_scf["PRINT"]["mulliken"] = scf_opt["PRINT"]["mulliken"]
  ec_scf["PRINT"]["mo_vectors"] = scf_opt["PRINT"]["mo_vectors"]

  """
    DFT block
  """
  dft_opt = nwchem_opt["DFT"]
  if dft_opt["hfexch"]:
    ec_scf["multiplicity"] = dft_opt["multiplicity"]
    ec_scf["lshift"] = dft_opt["lshift"]
    ec_scf["tol_int"] = dft_opt["tol_int"]
    ec_scf["tol_lindep"] = dft_opt["tol_lindep"]
    ec_scf["conve"] = dft_opt["conve"]
    ec_scf["convd"] = dft_opt["convd"]
    ec_scf["alpha"] = dft_opt["alpha"]
    ec_scf["scf_type"] = dft_opt["type"]
    ec_scf["diis_hist"] = dft_opt["diis_hist"]
    exachem_opt["common"]["maxiter"] = dft_opt["maxiter"]
    ec_scf["PRINT"]["mulliken"] = dft_opt["PRINT"]["mulliken"]
    ec_scf["PRINT"]["mo_vectors"] = dft_opt["PRINT"]["mo_vectors"]    

  """
    TCE block

    task tce energy/ener
    cuda 1
    thresh, maxiter, diis, DIPOLE
    freeze atomic, freeze core 10 (or freeze 10), freeze virtual 5
    
    CC2,CCSD,CCSD(T), ignore the rest

    TODO: <eaccsd|iaccsd|ccsd> <EOMSOL> nroots 2
    set tce:maxeorb 0.1
    set tce:threshl 1.0d-3
    set tce:thresheom 1.0d-4

    ignore:
    DFT|HF|SCF
    2eorb,2emet,tce:nts,tilesize,targetsym,symmetry,attilesize
    task tce <gradient|optimize|frequencies>

    response properties
    set tce:lineresp <logical lineresp default: F>
    set tce:afreq <double precision afreq(9) default: 0.0> 
    set tce:respaxis <logical respaxis(3) default: T T T>
  """

  tce_opt = nwchem_opt["TCE"]
  exachem_opt["CC"] = {}
  # exachem_opt["CC"]["PRINT"] = {}
  cc_opt = exachem_opt["CC"]
  cc_opt["threshold"]      = tce_opt["threshold"]
  cc_opt["ndiis"]           = tce_opt["ndiis"]
  cc_opt["lshift"]         = tce_opt["lshift"]
  cc_opt["ccsd_maxiter"]   = tce_opt["ccsd_maxiter"]
  cc_opt["freeze_core"]    = tce_opt["freeze_core"]
  cc_opt["freeze_virtual"] = tce_opt["freeze_virtual"]

  """
    MISC

    task mp2

    ignore: direct_mp2, rimp2, scs
    freeze atomic, freeze core 10 (or freeze 10), freeze virtual 5
    property, gradients

  """

  exachem_opt["TASK"] = nwchem_opt["TASK"]

  exachem_opt = dict_to_json(exachem_opt)

  print("\n\n=============================")
  print("UNSUPPORTED OPTIONS\n=============================")
  note_str = "NOTE: Only start line, geometry, basis, SCF and TCE blocks are parsed currently."
  note_str += "The memory, permanent_dir lines, load xyz file, all restart directives are ignored if present."
  print(note_str)
  for uopt in unsupported: print("\n - " + uopt.strip())
  print("\n\n--------------------------------------")
  print(" Lines ignored in %s\n--------------------------------------\n" %(input_file))
  for uopt in unsupported_rest: print(uopt,end='')

  ec_json_file = file_prefix + ".json"
  with open(ec_json_file, "w") as ecfile:
    ecfile.write(exachem_opt)
