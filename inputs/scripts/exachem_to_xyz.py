import os
import sys
import json

json_file = os.path.abspath(sys.argv[1])

if not os.path.exists(json_file): 
    print("ERROR: [" + json_file + "] does not exist!")
    sys.exit(1)

file_name = os.path.splitext(os.path.basename(json_file))[0]

xyz_file = file_name + ".xyz"

with open(json_file, 'r') as file:
    data = json.load(file)

# Prepare the XYZ file content
num_atoms = len(data['geometry']['coordinates'])
units = data['geometry']['units']
bohr2ang = 1.0
if "bohr" in units: bohr2ang = 0.529177210544

xyz_content = [f"{num_atoms}", ""]

for atom_line in data['geometry']['coordinates']:
    atom = atom_line.split(maxsplit=3)
    element, x, y, z = atom[0], float(atom[1])*bohr2ang, float(atom[2])*bohr2ang, float(atom[3])*bohr2ang
    xyz_content.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")

# Write to an XYZ file
with open(xyz_file, 'w') as file:
    file.write("\n".join(xyz_content))
    file.write("\n")

print(f"XYZ file generated as {xyz_file}")
