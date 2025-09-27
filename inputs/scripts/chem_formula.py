# Prints chemical formula from an .xyz or .json file, sorted by atomic number
# Requires the periodictable package: pip install periodictable
# Usage: python chem_formula.py path/to/input_file

import json
from collections import defaultdict
from periodictable import elements

def read_input(filename):

    xyz = []
    if filename.endswith('.json'):
        with open(filename) as f:
            jinput = json.load(f)
            xyz = jinput['geometry']['coordinates']
    else:
        with open(filename) as f:
            lines = f.readlines()[2:]  # skip first two lines (atom count and comment)
            xyz = lines

    atom_counts = defaultdict(int)

    for line in xyz:
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
    parser.add_argument("inp_file", help="Path to the .xyz or .json file")
    args = parser.parse_args()
    counts = read_input(args.inp_file)

    formula = generate_formula(counts)
    print(formula)

