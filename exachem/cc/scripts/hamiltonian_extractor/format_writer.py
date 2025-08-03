"""
Base class for format writers.
"""
from abc import ABC, abstractmethod


class FormatWriter(ABC):
    """Abstract base class for Hamiltonian format writers."""
    
    def __init__(self, output_prefix):
        self.output_prefix = output_prefix
        
    @abstractmethod
    def write(self, data):
        """
        Write the Hamiltonian data to the specified format.
        
        Args:
            data (HamiltonianData): Container with extracted data
        """
        pass
    
    @abstractmethod
    def get_file_extension(self):
        """Get the file extension for this format."""
        pass
    
    @abstractmethod
    def get_description(self):
        """Get a description of this format."""
        pass


def write_info_file(data, output_prefix):
    """
    Write information file about the extracted Hamiltonian.
    
    Args:
        data (HamiltonianData): Container with extracted data
        output_prefix (str): Output file prefix
    """
    with open(f"{output_prefix}-info", 'w') as f:
        f.write("**************************************************************\n")
        f.write(f"File = {output_prefix}\n\n")
        f.write(f"Nuclear Repulsion E = {data.coulomb_repulsion['value']}\n")
        f.write(f"Energy Shift = {data.energy_shift}\n")
        f.write(f"Total Energy Shift = {data.coulomb_repulsion['value'] + data.energy_shift}\n")
        f.write(f"# Active Orbitals = {data.n_orbitals}\n")
        f.write(f"# Occupied Alpha Orbitals = {data.n_occ_alpha}\n")
        f.write(f"# Occupied Beta Orbitals = {data.n_occ_beta}\n")
        f.write(f"# Virtual Alpha Orbitals = {data.n_virt_alpha}\n")
        f.write(f"# Virtual Beta Orbitals = {data.n_virt_beta}\n\n")
        f.write(f"Original SCF Energy = {data.scf_energy}\n")
        f.write(f"New Repulsion Energy (Repulsion + Shift) = {data.coulomb_repulsion['value'] + data.energy_shift}\n\n")
        
        # f.write("New orbital energies after downfolding\n")
        # f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        # sorted_energies = sorted(data.orbital_energies)
        # for i, energy in enumerate(sorted_energies):
        #     f.write(f"{energy:.6f}\n")

        f.write("Fock Matrix Elements (i <= j)\n")
        f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        for i in range(data.n_orbitals):
            for j in range(i, data.n_orbitals):
                if abs(data.fock[i, j]) > data.printthresh:
                    f.write(f"Fock[{i+1},{j+1}] = {data.fock[i, j]:.6f}\n")
