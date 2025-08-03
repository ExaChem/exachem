"""
FCIDUMP format writer for DMRG calculations.
"""
try:
    from .format_writer import FormatWriter
except ImportError:
    from format_writer import FormatWriter


class FCIDUMPWriter(FormatWriter):
    """Writer for FCIDUMP format (modified for DMRG calculations)."""
    
    def get_file_extension(self):
        return "-FCIDUMP"
    
    def get_description(self):
        return "FCIDUMP format (modified for DMRG calculations)"
    
    def write(self, data):
        """Write Hamiltonian data in FCIDUMP format."""
        filename = f"{self.output_prefix}{self.get_file_extension()}"
        
        with open(filename, 'w') as f:
            # Write header
            f.write("&FCI NORB=%3d,NELEC=%3d,MS2=0,\n" % 
                   (data.n_orbitals, 2 * data.n_occ_alpha))
            f.write(" ORBSYM=1" + ",1" * (data.n_orbitals - 1))
            f.write("\n ISYM=1,\n")
            f.write("&END\n")
            
            # Write two-electron integrals
            self._write_two_electron_integrals(f, data)
            
            # Write one-electron integrals
            self._write_one_electron_integrals(f, data)
            
            # Write nuclear repulsion energy
            core_repulsion = (data.coulomb_repulsion['value'] + 
                            data.energy_shift)
            f.write(f"{core_repulsion:12.10f} 0 0 0 0\n")
    
    def _write_two_electron_integrals(self, f, data):
        """Write two-electron integrals to file."""
        # AAAA (alpha-alpha-alpha-alpha)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = (data.two_body[w, x, y, z] - 
                                data.two_body[w, z, y, x])
                        if abs(value) > data.printthresh:
                            f.write(f" {value:12.10f} {w+1:5d} "
                                   f"{x+1:5d} {y+1:5d} "
                                   f"{z+1:5d}\n")
        f.write("0.0 0 0 0 0\n")
        
        # BBBB (beta-beta-beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = (data.two_body[w, x, y, z] - 
                                data.two_body[w, z, y, x])
                        if abs(value) > data.printthresh:
                            f.write(f" {value:12.10f} {w+1:5d} "
                                   f"{x+1:5d} {y+1:5d} "
                                   f"{z+1:5d}\n")
        f.write("0.0 0 0 0 0\n")
        
        # AABB (alpha-alpha-beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = data.two_body[w, x, y, z]
                        if abs(value) > data.printthresh:
                            f.write(f" {value:12.10f} {w+1:5d} "
                                   f"{x+1:5d} {y+1:5d} "
                                   f"{z+1:5d}\n")
        f.write("0.0 0 0 0 0\n")
    
    def _write_one_electron_integrals(self, f, data):
        """Write one-electron integrals to file."""
        # AA (alpha-alpha)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                if abs(data.one_body[w, x]) > data.printthresh:
                    f.write(f" {data.one_body[w, x]:12.10f} {w+1:5d} "
                           f"{x+1:5d}     0     0\n")
        f.write("0.0 0 0 0 0\n")
        
        # BB (beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                if abs(data.one_body[w, x]) > data.printthresh:
                    f.write(f" {data.one_body[w, x]:12.10f} {w+1:5d} "
                           f"{x+1:5d}     0     0\n")
