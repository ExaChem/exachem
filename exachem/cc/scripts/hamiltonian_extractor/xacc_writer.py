"""
XACC format writer for quantum computing applications.
"""
try:
    from .format_writer import FormatWriter
except ImportError:
    from format_writer import FormatWriter


class XACCWriter(FormatWriter):
    """Writer for XACC format needed for the XACC program."""
    
    def get_file_extension(self):
        return "-xacc"
    
    def get_description(self):
        return "XACC format for quantum computing applications"
    
    def write(self, data):
        """Write Hamiltonian data in XACC format."""
        filename = f"{self.output_prefix}{self.get_file_extension()}"
        
        xacc_hamiltonian = []
        
        # Two-electron integrals
        self._add_two_electron_terms(xacc_hamiltonian, data)
        
        # One-electron integrals
        self._add_one_electron_terms(xacc_hamiltonian, data)
        
        # Constant term (nuclear repulsion + energy shift)
        constant_term = (data.coulomb_repulsion['value'] + 
                        data.energy_shift)
        xacc_hamiltonian.append(f"({constant_term},0)")
        
        # Write to file
        with open(filename, 'w') as f:
            for term in xacc_hamiltonian:
                f.write(term + "\n")
    
    def _add_two_electron_terms(self, xacc_hamiltonian, data):
        """Add two-electron integral terms to XACC Hamiltonian."""
        # AAAA terms (alpha-alpha-alpha-alpha)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = (data.two_body[w, x, y, z] - 
                                data.two_body[w, z, y, x])
                        if abs(value) > data.printthresh:
                            coefficient = value * 0.25
                            w_idx = w
                            x_idx = x
                            y_idx = y
                            z_idx = z
                            
                            term = (f"({coefficient},0){w_idx}^ {y_idx}^ "
                                   f"{z_idx} {x_idx} +")
                            xacc_hamiltonian.append(term)
        
        # BBBB terms (beta-beta-beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = (data.two_body[w, x, y, z] - 
                                data.two_body[w, z, y, x])
                        if abs(value) > data.printthresh:
                            coefficient = value * 0.25
                            w_idx = w + data.n_orbitals
                            x_idx = x + data.n_orbitals
                            y_idx = y + data.n_orbitals
                            z_idx = z + data.n_orbitals
                            
                            term = (f"({coefficient},0){w_idx}^ {y_idx}^ "
                                   f"{z_idx} {x_idx} +")
                            xacc_hamiltonian.append(term)
        
        # AABB terms (alpha-alpha-beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        value = data.two_body[w, x, y, z]
                        if abs(value) > data.printthresh:
                            coefficient = value * 0.25
                            w_idx = w
                            x_idx = x
                            y_idx = y + data.n_orbitals
                            z_idx = z + data.n_orbitals
                            
                            # Four terms for AABB
                            terms = [
                                f"({coefficient},0){w_idx}^ {y_idx}^ {z_idx} {x_idx} +",
                                f"({coefficient},0){y_idx}^ {w_idx}^ {x_idx} {z_idx} +",
                                f"({-coefficient},0){y_idx}^ {w_idx}^ {z_idx} {x_idx} +",
                                f"({-coefficient},0){w_idx}^ {y_idx}^ {x_idx} {z_idx} +"
                            ]
                            xacc_hamiltonian.extend(terms)
    
    def _add_one_electron_terms(self, xacc_hamiltonian, data):
        """Add one-electron integral terms to XACC Hamiltonian."""
        # AA terms (alpha-alpha)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                if abs(data.one_body[w, x]) > data.printthresh:
                    w_idx = w
                    x_idx = x
                    term = f"({data.one_body[w, x]},0){w_idx}^ {x_idx} +"
                    xacc_hamiltonian.append(term)
        
        # BB terms (beta-beta)
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                if abs(data.one_body[w, x]) > data.printthresh:
                    w_idx = w + data.n_orbitals
                    x_idx = x + data.n_orbitals
                    term = f"({data.one_body[w, x]},0){w_idx}^ {x_idx} +"
                    xacc_hamiltonian.append(term)
