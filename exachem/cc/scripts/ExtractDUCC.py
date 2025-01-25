# Note: This is only set up for closed shell systems !!!
#
# Produced 4 files:
# info - Contains information about the extracted Hamiltonian
# FCIDUMP - Not the normal FCIDUMP, but one for DMRG calculations
# yaml - YAML file based on Broombridge version 0.3
# xacc - format needed for the XACC program
#
# Usage:
# python3 ExtractDUCC.py <output> <integralpath> <frzncore>
# <output> is the output file from the DUCC calculation
# <integralpath> is the path to the integral file
# <frzncore> is the number of additional frozen core orbitals 
#            beyond those dropped in the DUCC calculation.

# Example:
# python3 ExtractDUCC.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt 0
# ---------------------------------------------------------------------------------------
import sys
import numpy as np
from numpy import linalg as LA
np.set_printoptions(threshold=sys.maxsize)

preamble=""""$schema": https://raw.githubusercontent.com/Microsoft/Quantum/master/Chemistry/Schema/broombridge-0.3.schema.json \n"""

emitter_yaml = False
emitter_ruamel = False
try:
    try:
        import ruamel.yaml as ruamel
    except ImportError:
        import ruamel_yaml as ruamel
    emitter_ruamel = True
except ImportError:
    import yaml
    emitter_yaml = True

if len(sys.argv) < 4:
    print("Insufficient arguments provided.")
    sys.exit(1)

output = sys.argv[1]
integralpath = sys.argv[2]
frzncore = int(sys.argv[3])
yamlpath = open(output+".yaml", 'w')

printthresh = 0.00000000005

def extract_fields():
    data = {}
    data['format'] = {'version' : '0.3'}
    data['bibliography'] = [{'url' : 'https://doi.org/10.48550/arXiv.2201.01257'}]
    reader_mode = ""
    geometry = None
    coulomb_repulsion = None
    scf_energy = None
    n_occ_alpha = None
    n_occ_beta = None
    n_virt = None
    n_frozen_core = None
    n_orbitals = None
    one_electron_integrals = None
    two_electron_integrals = None
    scf_energy_offset = None
    energy_offset = None
    fci_energy = None
    note = None
    xacc_hamiltonian = []

    with open(output, 'r') as f:
        for line in f:
            ln = line.strip()
            ln_segments = ln.split()
            if len(ln) == 0 or ln[0]=="#": #blank or comment line
                continue
            if reader_mode == "":
                if ln_segments[0] == '"coordinates":':
                    reader_mode = "cartesian_geometry"
                    geometry = {'coordinate_system': 'cartesian'}
                    geometry['atoms'] = []
                    geometry['symmetry'] = "C1"
#                elif ln_segments[:2] == ["Common", "Options"]:
#                    reader_mode = "basis"
                elif ln_segments[:4] == ["Nuclear", "repulsion", "energy", "="]:
                    coulomb_repulsion = {
                        'units' : 'hartree',
                        'value' : float(ln_segments[4])
                    }
                elif ln_segments[:5] == ['**', 'Total', 'SCF', 'energy', '=']:
                    scf_energy = {
                        'units' : 'hartree',
                        'value' : float(ln_segments[5])
                    }
                elif ln_segments[:6] == ["Number", "of", "active", "occupied", "alpha", "="]:
                    n_occ_alpha = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "occupied", "beta", "="]:
                    n_occ_beta = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "virtual", "alpha", "="]:
                    n_virt_alpha = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "virtual", "beta", "="]:
                    n_virt_beta = int(ln_segments[6])
                elif ln_segments[:2] == ["n_frozen_core", "="]:
                    n_frozen_core = int(ln_segments[2])
                elif ln_segments[:6] == ['CCSD', 'total', 'energy', '/', 'hartree', '=']:
                    note = "Full CCSD energy = " + ln_segments[6]
                elif ln_segments[:3] == ['Total', 'Energy', 'Shift:']:
                    Energy_shift = float(ln_segments[3])
                elif ln_segments[:1] == ['ducc_lvl']:
                    n_orbitals = n_occ_alpha + n_virt_alpha
                    assert n_occ_alpha - frzncore > 0, "Freezing too many orbitals"
                    assert n_occ_beta - frzncore > 0, "Freezing too many orbitals"
                    one_body = np.zeros((n_orbitals,n_orbitals))
                    two_body = np.zeros((n_orbitals,n_orbitals,n_orbitals,n_orbitals))
            if reader_mode == "cartesian_geometry":
                if ln_segments[0] == '"units":':
                    reader_mode = ""
                    geometry['units'] = ln_segments[1].replace('"','')
                elif len(ln_segments) == 4:
                    assert 'atoms' in geometry
                    geometry['atoms'] += [{"name":ln_segments[0].replace('"',''),
                                           "coords":
                                           [float(ln_segments[1]), float(ln_segments[2]), float(ln_segments[3].replace('"','').replace(',',''))]}]
#            if reader_mode == "basis":
#                if ln_segments[0] == "debug":
#                    reader_mode = ""
#                elif ln_segments[:2] == ["basis", "="]:
#                    basis_set = {}
#                    basis_set['name'] = ln_segments[2]
#                    basis_set['type'] = 'gaussian'



    with open(integralpath, 'r') as f:
        for line in f:
            ln = line.strip()
            ln_segments = ln.split()
            if len(ln) == 0 or ln[0]=="#": #blank or comment line
                continue
            if reader_mode == "":
                if ln_segments[:3] == ["Begin", "IJ", "Block"]:
                    reader_mode = "Read IJ"
                elif ln_segments[:3] == ["Begin", "IA", "Block"]:
                    reader_mode = "Read IA"
                elif ln_segments[:3] == ["Begin", "AB", "Block"]:
                    reader_mode = "Read AB"
                elif ln_segments[:3] == ["Begin", "IJKL", "Block"]:
                    reader_mode = "Read IJKL"
                elif ln_segments[:3] == ["Begin", "ABCD", "Block"]:
                    reader_mode = "Read ABCD"
                elif ln_segments[:3] == ["Begin", "IJAB", "Block"]:
                    reader_mode = "Read IJAB"
                elif ln_segments[:3] == ["Begin", "AIJB", "Block"]:
                    reader_mode = "Read AIJB"
                elif ln_segments[:3] == ["Begin", "IJKA", "Block"]:
                    reader_mode = "Read IJKA"
                elif ln_segments[:3] == ["Begin", "IABC", "Block"]:
                    reader_mode = "Read IABC"
            if reader_mode == "Read IJ":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 3
                    i = int(ln_segments[0])-1
                    j = int(ln_segments[1])-1
                    one_body[i,j] = float(ln_segments[2])
                    one_body[j,i] = float(ln_segments[2])
            if reader_mode == "Read IA":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 3
                    i = int(ln_segments[0])-1
                    a = int(ln_segments[1])-1+n_occ_alpha
                    one_body[i,a] = float(ln_segments[2])
                    one_body[a,i] = float(ln_segments[2])
            if reader_mode == "Read AB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 3
                    a = int(ln_segments[0])-1+n_occ_alpha
                    b = int(ln_segments[1])-1+n_occ_alpha
                    one_body[a,b] = float(ln_segments[2])
                    one_body[b,a] = float(ln_segments[2])
            if reader_mode == "Read IJKL":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    i = int(ln_segments[0])-1
                    j = int(ln_segments[1])-1-n_occ_alpha
                    k = int(ln_segments[2])-1
                    l = int(ln_segments[3])-1-n_occ_alpha
                    two_body[i,k,j,l] = float(ln_segments[4])
            if reader_mode == "Read ABCD":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    a = int(ln_segments[0])-1+n_occ_alpha
                    b = int(ln_segments[1])-1-n_virt_alpha+n_occ_alpha
                    c = int(ln_segments[2])-1+n_occ_alpha
                    d = int(ln_segments[3])-1-n_virt_alpha+n_occ_alpha
                    two_body[a,c,b,d] = float(ln_segments[4])
            if reader_mode == "Read IJAB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    i = int(ln_segments[0])-1
                    j = int(ln_segments[1])-1-n_occ_alpha
                    a = int(ln_segments[2])-1+n_occ_alpha
                    b = int(ln_segments[3])-1-n_virt_alpha+n_occ_alpha
                    two_body[i,a,j,b] = float(ln_segments[4])
                    two_body[a,i,b,j] = float(ln_segments[4])
            if reader_mode == "Read AIJB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    if int(ln_segments[2]) > n_occ_alpha:
                        a = int(ln_segments[0])-1+n_occ_alpha
                        i = int(ln_segments[1])-1-n_occ_alpha
                        j = int(ln_segments[2])-1-n_occ_alpha
                        b = int(ln_segments[3])-1+n_occ_alpha
                        two_body[a,b,i,j] = -float(ln_segments[4])
                        two_body[i,j,a,b] = -float(ln_segments[4])
                    else:
                        a = int(ln_segments[0])-1+n_occ_alpha
                        i = int(ln_segments[1])-1-n_occ_alpha
                        j = int(ln_segments[2])-1
                        b = int(ln_segments[3])-1-n_virt_alpha+n_occ_alpha
                        two_body[a,j,i,b] = float(ln_segments[4])
                        two_body[j,a,b,i] = float(ln_segments[4])
            if reader_mode == "Read IJKA":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    i = int(ln_segments[0])-1
                    j = int(ln_segments[1])-1-n_occ_alpha
                    k = int(ln_segments[2])-1
                    a = int(ln_segments[3])-1-n_virt_alpha+n_occ_alpha
                    two_body[i,k,j,a] = float(ln_segments[4])
                    two_body[j,a,i,k] = float(ln_segments[4])
                    two_body[k,i,a,j] = float(ln_segments[4])
                    two_body[a,j,k,i] = float(ln_segments[4])
            if reader_mode == "Read IABC":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    assert len(ln_segments) == 5
                    i = int(ln_segments[0])-1
                    a = int(ln_segments[1])-1-n_virt_alpha+n_occ_alpha
                    b = int(ln_segments[2])-1+n_occ_alpha
                    c = int(ln_segments[3])-1-n_virt_alpha+n_occ_alpha
                    two_body[i,b,a,c] = float(ln_segments[4])
                    two_body[b,i,c,a] = float(ln_segments[4])
                    two_body[a,c,i,b] = float(ln_segments[4])
                    two_body[c,a,b,i] = float(ln_segments[4])

    # Compute the Hartree--Fock energy
    Energy = coulomb_repulsion['value'] + Energy_shift
    for x in range(n_occ_alpha):
        Energy += 2*one_body[x,x]
        for y in range(n_occ_alpha):
            Energy += 2*two_body[x,x,y,y] - two_body[x,y,y,x]

    # Compute the core/offset energy
    Core_Energy = 0.0
    for x in range(frzncore):
        Core_Energy += 2*one_body[x,x]
        for y in range(frzncore):
            Core_Energy += 2*two_body[x,x,y,y] - two_body[x,y,y,x]
    scf_energy_offset = {
        'units' : 'hartree',
        'value' : float(Core_Energy)
    }
    energy_offset = {
        'units' : 'hartree',
        'value' : float(Core_Energy)
    }

    # Form Fock operator
    Fock = np.zeros((n_orbitals,n_orbitals))
    for x in range(n_orbitals):
        for y in range(n_orbitals):
            Fock[x,y] += one_body[x,y]
            for z in range(n_occ_alpha):
                Fock[x,y] += 2*two_body[z,z,x,y] - two_body[z,y,x,z]

    # Printing info and new orbital energies
    infopath = open(output+"-info", 'w')
    infopath.write("**************************************************************\n")
    infopath.write("File = {}\n\n".format(output))
    infopath.write("Nuclear Repulsion E = {}\n".format(coulomb_repulsion['value']))
    infopath.write("DUCC Energy Shift = {}\n".format(Energy_shift))
    infopath.write("Total Energy Shift = {}\n".format(coulomb_repulsion['value'] + Energy_shift))
    infopath.write("# Active Orbitals = {}\n".format(n_orbitals))
    infopath.write("# Frozen Core = {}\n".format(frzncore))
    infopath.write("# Occupied Alpha Orbitals = {}\n".format(n_occ_alpha))
    infopath.write("# Occupied Beta Orbitals = {}\n".format(n_occ_beta))
    infopath.write("# Virtual Alpha Orbitals = {}\n".format(n_virt_alpha))
    infopath.write("# Virtual Alpha Orbitals = {}\n\n".format(n_virt_beta))
    infopath.write("Original SCF Energy = {}\n".format(Energy))
    infopath.write("Core Energy ={}\n\n".format(Core_Energy))
    infopath.write("New Repulsion Energy (Core+Repulsion)= {}\n\n".format(Core_Energy + coulomb_repulsion['value'] + Energy_shift))

    infopath.write("New orbital energies after downfolding\n")
    infopath.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    Orb_E,Orb_Eig = LA.eig(Fock)
    for x in sorted(Orb_E):
        if sorted(Orb_E).index(x) < frzncore:
            infopath.write("{:.6f}    FROZEN\n".format(x))
        else:
            infopath.write("{:.6f}\n".format(x))

    # Zero out frozen core part
    for x in range(n_orbitals):
        for y in range(n_orbitals):
            if x < frzncore:
                Fock[x,y] = 0.0
            if y < frzncore:
                Fock[x,y] = 0.0

    # Remove two-electron contribution
    for x in range(frzncore,n_orbitals):
        for y in range(frzncore,n_orbitals):
            for z in range(frzncore,n_occ_alpha):
                Fock[x,y] -= 2*two_body[z,z,x,y] - two_body[z,y,x,z]

    one_electron_integrals = {
        'units' : 'hartree',
        'format' : 'sparse',
        'values' : []
    }

    two_electron_integrals = {
        'units' : 'hartree',
        'format' : 'sparse',
        # 'symmetry' : {'permutation' : 'fourfold', 'spin' : 'up-down'},
        'symmetry' : {'permutation' : 'fourfold'},
        'index_convention' : 'mulliken',
        'values' : []
    }

    #CREATE FCIDUMP FILE
    fcidumppath = open(output+"-FCIDUMP", 'w')
    fcidumppath.write("&FCI NORB=%3d,NELEC=%3d,MS2=0,\n" % (n_orbitals-frzncore, 2*(n_occ_alpha-frzncore)))
    fcidumppath.write(" ORBSYM=1"+",1"*(n_orbitals-frzncore-1))
    fcidumppath.write("\n ISYM=1,\n")
    fcidumppath.write("&END\n")

    #AAAA
    for w in range(frzncore,n_orbitals):
        for x in range(frzncore,n_orbitals):
            for y in range(frzncore,n_orbitals):
                for z in range(frzncore,n_orbitals):
                    if abs(two_body[w,x,y,z]-two_body[w,z,y,x])>printthresh:
                        fcidumppath.write(" %12.10f %5d %5d %5d %5d\n" % ((two_body[w,x,y,z]-two_body[w,z,y,x]), w-frzncore+1,x-frzncore+1,y-frzncore+1,z-frzncore+1))
                        xacc_hamiltonian.append("("+str((two_body[w,x,y,z]-two_body[w,z,y,x])*0.25)+",0)"+
                                                    str(w-frzncore)+"^ "+
                                                    str(y-frzncore)+"^ "+
                                                    str(z-frzncore)+" "+
                                                    str(x-frzncore)+" +")
    fcidumppath.write("0.0 0 0 0 0\n")

    #BBBB
    for w in range(frzncore,n_orbitals):
        for x in range(frzncore,n_orbitals):
            for y in range(frzncore,n_orbitals):
                for z in range(frzncore,n_orbitals):
                    if abs(two_body[w,x,y,z]-two_body[w,z,y,x])>printthresh:
                        fcidumppath.write(" %12.10f %5d %5d %5d %5d\n" % ((two_body[w,x,y,z]-two_body[w,z,y,x]), w-frzncore+1,x-frzncore+1,y-frzncore+1,z-frzncore+1))
                        xacc_hamiltonian.append("("+str((two_body[w,x,y,z]-two_body[w,z,y,x])*0.25)+",0)"+
                                                    str(w-2*frzncore+n_orbitals)+"^ "+
                                                    str(y-2*frzncore+n_orbitals)+"^ "+
                                                    str(z-2*frzncore+n_orbitals)+" "+
                                                    str(x-2*frzncore+n_orbitals)+" +")
    fcidumppath.write("0.0 0 0 0 0\n")

    #AABB
    for w in range(frzncore,n_orbitals):
        for x in range(frzncore,n_orbitals):
            for y in range(frzncore,n_orbitals):
                for z in range(frzncore,n_orbitals):
                    if abs(two_body[w,x,y,z])>printthresh:
                        fcidumppath.write(" %12.10f %5d %5d %5d %5d\n" % (two_body[w,x,y,z], w-frzncore+1,x-frzncore+1,y-frzncore+1,z-frzncore+1))
                        two_electron_integrals['values'] += [{
                            'key' : [
                            w-frzncore+1,
                            x-frzncore+1,
                            y-frzncore+1,
                            z-frzncore+1], 'value' :
                            float(two_body[w,x,y,z])
                            }]
                        xacc_hamiltonian.append("("+str(two_body[w,x,y,z]*0.25)+",0)"+
                                                    str(w-frzncore)+"^ "+
                                                    str(y-2*frzncore+n_orbitals)+"^ "+
                                                    str(z-2*frzncore+n_orbitals)+" "+
                                                    str(x-frzncore)+" +")
                        xacc_hamiltonian.append("("+str(two_body[w,x,y,z]*0.25)+",0)"+
                                                    str(y-2*frzncore+n_orbitals)+"^ "+
                                                    str(w-frzncore)+"^ "+
                                                    str(x-frzncore)+" "+
                                                    str(z-2*frzncore+n_orbitals)+" +")
                        xacc_hamiltonian.append("("+str(two_body[w,x,y,z]*-0.25)+",0)"+
                                                    str(y-2*frzncore+n_orbitals)+"^ "+
                                                    str(w-frzncore)+"^ "+
                                                    str(z-2*frzncore+n_orbitals)+" "+
                                                    str(x-frzncore)+" +")
                        xacc_hamiltonian.append("("+str(two_body[w,x,y,z]*-0.25)+",0)"+
                                                    str(w-frzncore)+"^ "+
                                                    str(y-2*frzncore+n_orbitals)+"^ "+
                                                    str(x-frzncore)+" "+
                                                    str(z-2*frzncore+n_orbitals)+" +")
    fcidumppath.write("0.0 0 0 0 0\n")

    #AA
    for w in range(frzncore,n_orbitals):
        for x in range(frzncore,n_orbitals):
            if abs(Fock[w,x])>printthresh:
                fcidumppath.write(" %12.10f %5d %5d %5d %5d\n" % (Fock[w,x], w-frzncore+1,x-frzncore+1, 0, 0))
                one_electron_integrals['values'] += [{
                    'key' : [
                    w-frzncore+1,
                    x-frzncore+1], 'value' :
                    float(Fock[w,x])
                    }]
                xacc_hamiltonian.append("("+str(Fock[w,x])+",0)"+
                                            str(w-frzncore)+"^ "+
                                            str(x-frzncore)+" +")
    fcidumppath.write("0.0 0 0 0 0\n")

    #BB
    for w in range(frzncore,n_orbitals):
        for x in range(frzncore,n_orbitals):
            if abs(Fock[w,x])>printthresh:
                fcidumppath.write(" %12.10f %5d %5d %5d %5d\n" % (Fock[w,x], w-frzncore+1,x-frzncore+1, 0, 0))
                xacc_hamiltonian.append("("+str(Fock[w,x])+",0)"+
                                            str(w-2*frzncore+n_orbitals)+"^ "+
                                            str(x-2*frzncore+n_orbitals)+" +")
    fcidumppath.write("%12.10f 0 0 0 0\n" % ((Core_Energy + coulomb_repulsion['value'] + Energy_shift)))
    xacc_hamiltonian.append("("+str((Core_Energy + coulomb_repulsion['value'] + Energy_shift))+",0)")

    xaccpath = open(output+"-xacc", 'w')
    for term in xacc_hamiltonian:
        xaccpath.write(term+"\n")

    fci_energy = {"units" : "hartree", "value" : 0.0, "upper": 0.0, "lower": 0.0}
    if frzncore > 0:
        assert n_frozen_core == 0, "Trying to freeze core orbitals of a calculations with frozen core orbitals"

    assert n_occ_alpha == n_occ_beta, "# occupied alpha is not equal to the # of occupied beta."
    assert geometry is not None, "geometry information is missing from output. Required to extract YAML"
    assert coulomb_repulsion is not None, "coulomb_repulsion is missing from output. Required to extract YAML"
    assert scf_energy is not None, "scf_energy is missing from output. Required to extract YAML"
    assert n_orbitals is not None, "n_orbitals is missing from output. Required to extract YAML"
#    assert basis_set is not None, "basis_set is missing from output. Required to extract YAML"
    assert one_electron_integrals is not None, "one_electron_integrals is missing from output. Required to extract YAML"
    assert two_electron_integrals is not None, "two_electron_integrals is missing from output. Required to extract YAML"
    assert scf_energy_offset is not None, "scf_energy_offset was not computed. Required to extract YAML"
    assert energy_offset is not None, "energy_offset was not computed. Required to extract YAML"
    assert fci_energy is not None, "fci_energy is missing. Required to extract YAML"

    coulomb_repulsion['value'] = float(Core_Energy + coulomb_repulsion['value'] + Energy_shift)
    
    hamiltonian = {'one_electron_integrals' : one_electron_integrals,
                   'two_electron_integrals' : two_electron_integrals}
    integral_sets =  [{"metadata": { 'molecule_name' : 'unknown', 'note' : note},
                       "geometry":geometry,
                       "coulomb_repulsion" : coulomb_repulsion,
                       "scf_energy" : scf_energy,
                       "n_orbitals" : n_orbitals - frzncore, 
                       "n_electrons" : n_occ_alpha + n_occ_beta - 2*frzncore,
#                       "basis_set": basis_set,
                       "fci_energy" : fci_energy,
                       "hamiltonian" : hamiltonian,
                       "scf_energy_offset" : scf_energy_offset,
                       "energy_offset" : energy_offset
                       }]
    data['problem_description'] = integral_sets
    return data

def emitter_ruamel_func():
    yaml = ruamel.YAML(typ="safe")
    yamlpath.write(preamble)
    data = extract_fields()
    # yamlpath.write(yaml.dump(data, sys.stdout))
    yaml.dump(data, yamlpath)

def emitter_yaml_func():
    yamlpath.write(preamble)
    data = extract_fields()
    yamlpath.write(yaml.dump(data, default_flow_style=None))

def main():
    """
    ExtractDUCC: parses integral data and outputs relevant CC information.
    Usage: python ExtractDUCC.py <output> <integralpath> <frzncore>
    """

    assert emitter_yaml or emitter_ruamel, "Extraction failed: could not import YAML or RUAMEL packages."
    if emitter_yaml:
        emitter_yaml_func()
    elif emitter_ruamel:
        emitter_ruamel_func()
    else:
        assert False, "Unreachable code"

if __name__ == '__main__':
    main()
