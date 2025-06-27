# Prints chemical formula from an .xyz file, sorted by atomic number
# Requires the periodictable package: pip install periodictable
# Usage: python xyz_names.py path/to/file.xyz

from collections import defaultdict
from periodictable import elements

def read_xyz(filename):
    atom_counts = defaultdict(int)
    with open(filename) as f:
        lines = f.readlines()[2:]  # skip first two lines (atom count and comment)
        for line in lines:
            if line.strip() == "":
                continue
            parts = line.split()
            atom = parts[0]
            atom_counts[atom] += 1
    return atom_counts

def atomic_number(elem):
    try:
        return elements.symbol(elem).number
    except:
        return float('inf')  # for unknown elements, push to end

def generate_formula(atom_counts):
    sorted_elements = sorted(atom_counts.items(), key=lambda x: -atomic_number(x[0]))
    return ''.join(f"{elem}{count}" for elem, count in sorted_elements)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file", help="Path to the .xyz file")
    args = parser.parse_args()

    counts = read_xyz(args.xyz_file)
    formula = generate_formula(counts)
    print(formula)

