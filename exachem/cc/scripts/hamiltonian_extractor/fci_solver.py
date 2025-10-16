"""
Full Configuration Interaction (FCI) solver implementation.
"""
import numpy as np
from typing import Dict, Tuple, Optional

def parse_fci_vector(ci_vecmat: np.ndarray, orbs: int, electrons: int, threshold: float = 0.001) -> Dict[str, float]:
    """Parse the FCI vector into an occupation string representation.

    Args:
        ci_vecmat: The CI vector matrix
        orbs: Number of orbitals
        electrons: Number of electrons
        threshold: Amplitude threshold for printing (default: 0.001)

    Returns:
        Dictionary mapping occupation strings to amplitudes
    """
    try:
        from pyscf.fci import cistring
    except ImportError:
        raise ImportError("PySCF is required for parsing FCI vectors. Please install it using 'pip install pyscf'.")
    
    conf_bin = cistring.gen_strings4orblist(list(range(orbs)), electrons//2)

    OCCMAP = {('0', '0'): '0',
              ('1', '0'): 'a',
              ('0', '1'): 'b',
              ('1', '1'): '2'}

    fcivecorb = {}
    # Handle both single and multi-root cases
    if ci_vecmat.ndim == 1:
        for i, ca in enumerate(conf_bin):
            astring = bin(ca)[2:].zfill(orbs)
            bstring = bin(ca)[2:].zfill(orbs)
            s = ''.join(reversed([OCCMAP[a, b]
                        for a, b in zip(astring, bstring)]))
            fcivecorb[s] = ci_vecmat[i]
    else:
        for i, ca in enumerate(conf_bin):
            for j, cb in enumerate(conf_bin):
                astring = bin(ca)[2:].zfill(orbs)
                bstring = bin(cb)[2:].zfill(orbs)
                s = ''.join(reversed([OCCMAP[a, b]
                            for a, b in zip(astring, bstring)]))
                fcivecorb[s] = ci_vecmat[i, j]

    # Filter by threshold
    return {key: round(val, 5) for key, val in fcivecorb.items() 
            if abs(val) > threshold}

def solve_fci(data, n_roots: int = 1, threshold: float = 0.001, max_space: int = 450) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the FCI problem for given one and two electron integrals.

    Args:
        data: HamiltonianData object containing one_body and two_body integrals
        n_roots: Number of roots to compute (default: 1)
        threshold: Threshold for printing CI vector components (default: 0.001)
        max_space: Maximum space dimension for Davidson algorithm (default: 450)

    Returns:
        Tuple of (energies, CI vectors)
    """
    # Import pyscf only when needed
    try:
        from pyscf import fci
    except ImportError:
        raise ImportError("PySCF is required for FCI calculations. Please install it using 'pip install pyscf'.")
    
    # Set up FCI solver
    solver = fci.direct_nosym.FCISolver()
    
    # Calculate number of electrons
    n_electrons = 2 * data.n_occ_alpha  # For closed shell
    
    # Solve FCI problem
    energies, fcivecs = solver.kernel(
        data.one_body,
        data.two_body, 
        data.n_orbitals,
        n_electrons,
        nroots=n_roots,
        max_space=max_space,
        verbose=5
    )
    
    # Convert to numpy array if single root
    if n_roots == 1:
        energies = np.array([energies])
        fcivecs = np.array([fcivecs])
    
    # Process results
    results = []
    for i in range(n_roots):
        # Calculate total energy
        total_energy = (energies[i] + 
                       data.coulomb_repulsion['value'] + 
                       data.energy_shift)
        
        # Calculate spin multiplicity
        s, multiplicity = solver.spin_square(fcivecs[i], 
                                           data.n_orbitals,
                                           n_electrons)
        
        # Parse CI vector
        ci_dict = parse_fci_vector(fcivecs[i], data.n_orbitals, n_electrons, threshold)
        
        results.append({
            'energy': total_energy,
            'multiplicity': multiplicity,
            'ci_vector': ci_dict
        })
        
    return results
