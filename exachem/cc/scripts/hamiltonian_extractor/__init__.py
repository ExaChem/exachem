"""
Hamiltonian Extractor Package

This package provides utilities for extracting Hamiltonian data from
calculation output files and converting it to various formats for quantum
chemistry and quantum computing applications.

Main Components:
- grab_data: Core data extraction functionality
- hamiltonian_data: Data structures for Hamiltonian information
- format_writer: Base class for format writers
- fcidump_writer: FCIDUMP format writer
- yaml_writer: YAML format writer (Broombridge schema)
- xacc_writer: XACC format writer
- extract_hamiltonian: Main command-line utility

Usage:
    python extract_hamiltonian.py <output_file> <integral_path> [options]

For detailed usage information, run:
    python extract_hamiltonian.py --help
"""

# __version__ = "1.0.0"
# __author__ = "ExaChem Development Team - Nicholas Bauman"

from .grab_data import extract_hamiltonian_data
from .hamiltonian_data import HamiltonianData
from .format_writer import FormatWriter, write_info_file
from .fcidump_writer import FCIDUMPWriter
from .yaml_writer import YAMLWriter
from .xacc_writer import XACCWriter

__all__ = [
    'extract_hamiltonian_data',
    'HamiltonianData',
    'FormatWriter',
    'write_info_file',
    'FCIDUMPWriter',
    'YAMLWriter',
    'XACCWriter'
]
