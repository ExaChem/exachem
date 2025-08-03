# Hamiltonian Extractor

A modular utility for extracting Hamiltonian data from ExaChem calculation output files and converting it to various formats for quantum chemistry and quantum computing applications.

## Overview

This utility processes calculation results and generates Hamiltonian representations in multiple formats:
- **FCIDUMP**: Modified for DMRG calculations with MolMPS program
- **YAML**: Based on Broombridge version 0.3 schema
- **XACC**: For quantum computing applications with XACC software
- **Info**: Human-readable analysis file

## Requirements

- Python 3.6+
- NumPy
- PyYAML or ruamel.yaml (for YAML format output)
- PySCF (for FCI calculations)

## Installation

No installation required. Simply use the script directly from the `hamiltonian_extractor` directory.

## Usage

### Basic Usage

```bash
python extract_hamiltonian.py <output_file> <integral_path> [options]
```

### Examples

1. **Extract all formats:**
   ```bash
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt
   ```

2. **Extract only FCIDUMP format:**
   ```bash
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --format fcidump
   ```

3. **Extract multiple specific formats:**
   ```bash
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --format yaml xacc
   ```

4. **Use custom output prefix and threshold:**
   ```bash
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --output-prefix my_hamiltonian --threshold 1e-10
   ```

5. **Run FCI calculation:**
   ```bash
   # Calculate ground state
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --fci

   # Calculate multiple states
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --fci --n-roots 3

   # Adjust CI vector printing threshold
   python extract_hamiltonian.py h2o_ducc.out h2o_ducc.cc-pvdz_files/restricted/ducc/h2o_ducc.cc-pvdz.ducc.results.txt --fci --fci-threshold 0.01
   ```

### Command-line Options

- `output_file`: Path to the DUCC output file
- `integral_path`: Path to the integral file (typically ends with .ducc.results.txt)
- `--format`: Output format(s) to generate (choices: fcidump, yaml, xacc, all; default: all)
- `--output-prefix`: Prefix for output files (default: use input output_file name)
- `--threshold`: Threshold for printing integrals (default: 5e-11)
- `--verbose`: Enable verbose output
- `--fci`: Perform Full Configuration Interaction (FCI) calculation
- `--n-roots`: Number of FCI roots to compute (default: 1)
- `--fci-threshold`: Threshold for printing CI vector components (default: 0.001)

### Output Files

The utility generates the following files (using `output_prefix` as the base name):

1. **`{output_prefix}-info`**: Human-readable analysis file containing:
   - Nuclear repulsion energy
   - Energy shifts
   - Orbital counts
   - SCF and core energies
   - Orbital energies after downfolding

2. **`{output_prefix}-FCIDUMP`**: FCIDUMP format file for DMRG calculations

3. **`{output_prefix}.yaml`**: YAML format file based on Broombridge version 0.3 schema

4. **`{output_prefix}-xacc`**: XACC format file for quantum computing applications

## Architecture

The utility is designed with a modular architecture:

```
hamiltonian_extractor/
├── __init__.py                 # Package initialization
├── extract_hamiltonian.py      # Main command-line utility
├── grab_data.py                # Core data extraction
├── hamiltonian_data.py         # Data structures and analysis
├── format_writer.py            # Base class for format writers
├── fcidump_writer.py           # FCIDUMP format writer
├── yaml_writer.py              # YAML format writer
├── xacc_writer.py              # XACC format writer
├── fci_solver.py              # Full Configuration Interaction solver
└── README.md                   # This file
```

### Key Components

- **`HamiltonianData`**: Container class for extracted data and analysis results
- **`FormatWriter`**: Abstract base class for format writers
- **Format Writers**: Specific implementations for each output format
- **`FCISolver`**: Full Configuration Interaction solver with support for:
  - Ground state and excited states calculation
  - Spin multiplicity analysis
  - CI vector analysis with configurable threshold
- **`extract_hamiltonian_data()`**: Main extraction function

## Adding New Formats

To add a new output format:

1. Create a new writer class inheriting from `FormatWriter`
2. Implement the required methods:
   - `write(data)`: Write the format
   - `get_file_extension()`: Return file extension
   - `get_description()`: Return format description
3. Add the writer to the `get_format_writers()` function in `extract_hamiltonian.py`
4. Update the argument parser choices

Example:
```python
from .format_writer import FormatWriter

class MyFormatWriter(FormatWriter):
    def get_file_extension(self):
        return ".myformat"
    
    def get_description(self):
        return "My custom format"
    
    def write(self, data):
        filename = f"{self.output_prefix}{self.get_file_extension()}"
        with open(filename, 'w') as f:
            # Write your format here
            pass
```

## Data Access

The `HamiltonianData` object provides access to:

- **Raw data**: geometry, energies, orbital counts
- **Integral matrices**: one_body, two_body, fock
- **Processed data**: scf_energy
- **Helper methods**: compute_energies(), compute_fock_matrix(), validate_data()

## Limitations

- **Closed Shell Systems Only**: Currently designed for closed shell systems
- **DUCC Format Specific**: Tailored for DUCC output format
- **Python Dependencies**: Requires NumPy; YAML format needs PyYAML or ruamel.yaml
