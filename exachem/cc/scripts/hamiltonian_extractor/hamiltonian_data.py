"""
Data structures and utility functions for Hamiltonian extraction.
"""
import numpy as np
from numpy import linalg as LA


class HamiltonianData:
    """Container for extracted Hamiltonian data and analysis results."""
    
    def __init__(self):
        # Raw data from files
        self.geometry = None
        self.coulomb_repulsion = None
        self.scf_energy = None
        self.n_occ_alpha = None
        self.n_occ_beta = None
        self.n_virt_alpha = None
        self.n_virt_beta = None
        self.n_frozen_core = None
        self.n_orbitals = None
        self.energy_shift = None
        self.note = None
        
        # Integral matrices
        self.one_body = None
        self.two_body = None
        self.fock = None
        
        # Processed data
        self.scf_energy = None
        # self.orbital_energies = None
        
        # Parameters
        self.printthresh = 0.00000000005
        
    def compute_energies(self):
        """Compute Hartree-Fock and core energies."""
        # Compute the Hartree-Fock energy
        self.scf_energy = self.coulomb_repulsion['value'] + self.energy_shift
        for x in range(self.n_occ_alpha):
            self.scf_energy += 2 * self.one_body[x, x]
            for y in range(self.n_occ_alpha):
                self.scf_energy += 2 * self.two_body[x, x, y, y] - self.two_body[x, y, y, x]


    def compute_fock_matrix(self):
        """Form the Fock operator matrix."""
        self.fock = np.zeros((self.n_orbitals, self.n_orbitals))
        for x in range(self.n_orbitals):
            for y in range(self.n_orbitals):
                self.fock[x, y] += self.one_body[x, y]
                for z in range(self.n_occ_alpha):
                    self.fock[x, y] += 2 * self.two_body[z, z, x, y] - self.two_body[z, y, x, z]
        
        # # Compute orbital energies
        # self.orbital_energies, _ = LA.eig(self.fock)
