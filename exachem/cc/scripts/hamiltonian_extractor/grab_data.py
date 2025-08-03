"""
Hamiltonian data extraction utility.
"""
import numpy as np
try:
    from .hamiltonian_data import HamiltonianData
except ImportError:
    from hamiltonian_data import HamiltonianData


def extract_hamiltonian_data(output_file, integral_path):
    """
    Extract Hamiltonian data from output files.
    
    Args:
        output_file (str): Path to the output file
        integral_path (str): Path to the integral file
        
    Returns:
        HamiltonianData: Container with extracted data
    """
    data = HamiltonianData()
    
    # Extract data from output file
    _extract_output_data(output_file, data)
    
    # Extract integral data
    _extract_integral_data(integral_path, data)
    
    # Perform analysis
    data.compute_energies()
    data.compute_fock_matrix()
    
    return data


def _extract_output_data(output_file, data):
    """Extract data from the output file."""
    reader_mode = ""
    full_scf = None
    bare_scf = None
    
    with open(output_file, 'r') as f:
        for line in f:
            ln = line.strip()
            ln_segments = ln.split()
            
            if len(ln) == 0 or ln.startswith("#"):
                continue
                
            if reader_mode == "":
                if ln_segments[0] == '"coordinates":':
                    reader_mode = "cartesian_geometry"
                    data.geometry = {'coordinate_system': 'cartesian'}
                    data.geometry['atoms'] = []
                    data.geometry['symmetry'] = "C1"
                elif ln_segments[:4] == ["Nuclear", "repulsion", "energy", "="]:
                    data.coulomb_repulsion = {
                        'units': 'hartree',
                        'value': float(ln_segments[4])
                    }
                elif ln_segments[:5] == ['**', 'Total', 'SCF', 'energy', '=']:
                    data.scf_energy = {
                        'units': 'hartree',
                        'value': float(ln_segments[5])
                    }
                elif ln_segments[:6] == ["Number", "of", "active", "occupied", "alpha", "="]:
                    data.n_occ_alpha = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "occupied", "beta", "="]:
                    data.n_occ_beta = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "virtual", "alpha", "="]:
                    data.n_virt_alpha = int(ln_segments[6])
                elif ln_segments[:6] == ["Number", "of", "active", "virtual", "beta", "="]:
                    data.n_virt_beta = int(ln_segments[6])
                elif ln_segments[:2] == ["n_frozen_core", "="]:
                    data.n_frozen_core = int(ln_segments[2])
                elif ln_segments[:6] == ['CCSD', 'total', 'energy', '/', 'hartree', '=']:
                    data.note = "Full CCSD energy = " + ln_segments[6]
                elif ln_segments[:3] == ['Total', 'Energy', 'Shift:']:
                    data.energy_shift = float(ln_segments[3])
                elif ln_segments[:3] == ['Full', 'SCF', 'Energy:']:
                    full_scf = float(ln_segments[3])
                elif ln_segments[:3] == ['Bare', 'SCF', 'Energy:']:
                    bare_scf = float(ln_segments[3])
                elif ln_segments[:1] == ['ducc_lvl']:
                    data.n_orbitals = data.n_occ_alpha + data.n_virt_alpha
                    data.one_body = np.zeros((data.n_orbitals, data.n_orbitals))
                    data.two_body = np.zeros((data.n_orbitals, data.n_orbitals, 
                                            data.n_orbitals, data.n_orbitals))
                    
            elif reader_mode == "cartesian_geometry":
                if ln_segments[0] == '"units":':
                    reader_mode = ""
                    data.geometry['units'] = ln_segments[1].replace('"', '')
                elif len(ln_segments) == 4:
                    data.geometry['atoms'].append({
                        "name": ln_segments[0].replace('"', ''),
                        "coords": [float(ln_segments[1]), float(ln_segments[2]), 
                                 float(ln_segments[3].replace('"', '').replace(',', ''))]
                    })

        # After reading file, check if we need to calculate energy shift
        if not hasattr(data, 'energy_shift') or data.energy_shift is None:
            if full_scf is not None and bare_scf is not None:
                data.energy_shift = full_scf - bare_scf


def _extract_integral_data(integral_path, data):
    """Extract integral data from the integral file."""
    reader_mode = ""
    with open(integral_path, 'r') as f:
        for line in f:
            ln = line.strip()
            ln_segments = ln.split()
            
            if len(ln) == 0 or ln.startswith("#"):
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
                    
            elif reader_mode == "Read IJ":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i, j = int(ln_segments[0]) - 1, int(ln_segments[1]) - 1
                    value = float(ln_segments[2])
                    data.one_body[i, j] = data.one_body[j, i] = value
                    
            elif reader_mode == "Read IA":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i = int(ln_segments[0]) - 1
                    a = int(ln_segments[1]) - 1 + data.n_occ_alpha
                    value = float(ln_segments[2])
                    data.one_body[i, a] = data.one_body[a, i] = value
                    
            elif reader_mode == "Read AB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    a = int(ln_segments[0]) - 1 + data.n_occ_alpha
                    b = int(ln_segments[1]) - 1 + data.n_occ_alpha
                    value = float(ln_segments[2])
                    data.one_body[a, b] = data.one_body[b, a] = value
                    
            elif reader_mode == "Read IJKL":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i = int(ln_segments[0]) - 1
                    j = int(ln_segments[1]) - 1 - data.n_occ_alpha
                    k = int(ln_segments[2]) - 1
                    l = int(ln_segments[3]) - 1 - data.n_occ_alpha
                    value = float(ln_segments[4])
                    data.two_body[i, k, j, l] = value
                    
            elif reader_mode == "Read ABCD":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    a = int(ln_segments[0]) - 1 + data.n_occ_alpha
                    b = int(ln_segments[1]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    c = int(ln_segments[2]) - 1 + data.n_occ_alpha
                    d = int(ln_segments[3]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    value = float(ln_segments[4])
                    data.two_body[a, c, b, d] = value
                    
            elif reader_mode == "Read IJAB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i = int(ln_segments[0]) - 1
                    j = int(ln_segments[1]) - 1 - data.n_occ_alpha
                    a = int(ln_segments[2]) - 1 + data.n_occ_alpha
                    b = int(ln_segments[3]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    value = float(ln_segments[4])
                    data.two_body[i, a, j, b] = value
                    data.two_body[a, i, b, j] = value
                    
            elif reader_mode == "Read AIJB":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    if int(ln_segments[2]) > data.n_occ_alpha:
                        a = int(ln_segments[0]) - 1 + data.n_occ_alpha
                        i = int(ln_segments[1]) - 1 - data.n_occ_alpha
                        j = int(ln_segments[2]) - 1 - data.n_occ_alpha
                        b = int(ln_segments[3]) - 1 + data.n_occ_alpha
                        value = -float(ln_segments[4])
                        data.two_body[a, b, i, j] = value
                        data.two_body[i, j, a, b] = value
                    else:
                        a = int(ln_segments[0]) - 1 + data.n_occ_alpha
                        i = int(ln_segments[1]) - 1 - data.n_occ_alpha
                        j = int(ln_segments[2]) - 1
                        b = int(ln_segments[3]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                        value = float(ln_segments[4])
                        data.two_body[a, j, i, b] = value
                        data.two_body[j, a, b, i] = value
                        
            elif reader_mode == "Read IJKA":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i = int(ln_segments[0]) - 1
                    j = int(ln_segments[1]) - 1 - data.n_occ_alpha
                    k = int(ln_segments[2]) - 1
                    a = int(ln_segments[3]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    value = float(ln_segments[4])
                    data.two_body[i, k, j, a] = value
                    data.two_body[j, a, i, k] = value
                    data.two_body[k, i, a, j] = value
                    data.two_body[a, j, k, i] = value
                    
            elif reader_mode == "Read IABC":
                if ln_segments[0] == "End":
                    reader_mode = ""
                elif ln_segments[0] != "Begin":
                    i = int(ln_segments[0]) - 1
                    a = int(ln_segments[1]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    b = int(ln_segments[2]) - 1 + data.n_occ_alpha
                    c = int(ln_segments[3]) - 1 - data.n_virt_alpha + data.n_occ_alpha
                    value = float(ln_segments[4])
                    data.two_body[i, b, a, c] = value
                    data.two_body[b, i, c, a] = value
                    data.two_body[a, c, i, b] = value
                    data.two_body[c, a, b, i] = value
