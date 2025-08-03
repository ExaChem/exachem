"""
YAML format writer based on Broombridge version 0.3.
"""
try:
    from .format_writer import FormatWriter
except ImportError:
    from format_writer import FormatWriter

# Try to import YAML libraries
emitter_yaml = False
emitter_ruamel = False
try:
    try:
        import ruamel.yaml as ruamel
    except ImportError:
        import ruamel_yaml as ruamel
    emitter_ruamel = True
except ImportError:
    try:
        import yaml
        emitter_yaml = True
    except ImportError:
        pass


class YAMLWriter(FormatWriter):
    """Writer for YAML format based on Broombridge version 0.3."""
    
    def __init__(self, output_prefix):
        super().__init__(output_prefix)
        self.preamble = '"$schema": https://raw.githubusercontent.com/Microsoft/Quantum/master/Chemistry/Schema/broombridge-0.3.schema.json \n'
        
        if not (emitter_yaml or emitter_ruamel):
            raise ImportError("Could not import YAML or RUAMEL packages. Please install one of them.")
    
    def get_file_extension(self):
        return ".yaml"
    
    def get_description(self):
        return "YAML format based on Broombridge version 0.3"
    
    def write(self, data):
        """Write Hamiltonian data in YAML format."""
        filename = f"{self.output_prefix}{self.get_file_extension()}"
        
        with open(filename, 'w') as f:
            f.write(self.preamble)
            yaml_data = self._build_yaml_data(data)
            
            if emitter_ruamel:
                yaml_obj = ruamel.YAML(typ="safe")
                yaml_obj.dump(yaml_data, f)
            elif emitter_yaml:
                yaml.dump(yaml_data, f, default_flow_style=None)
    
    def _get_integral_data(self, data):
        """Get processed integral data for YAML format."""
        one_electron_integrals = {
            'units': 'hartree',
            'format': 'sparse',
            'values': []
        }
        
        two_electron_integrals = {
            'units': 'hartree',
            'format': 'sparse',
            'symmetry': {'permutation': 'fourfold'},
            'index_convention': 'mulliken',
            'values': []
        }
        
        # Fill one-electron integrals
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                if abs(data.one_body[w, x]) > data.printthresh:
                    one_electron_integrals['values'].append({
                        'key': [w + 1, x + 1],
                        'value': float(data.one_body[w, x])
                    })
        
        # Fill two-electron integrals
        for w in range(data.n_orbitals):
            for x in range(data.n_orbitals):
                for y in range(data.n_orbitals):
                    for z in range(data.n_orbitals):
                        if abs(data.two_body[w, x, y, z]) > data.printthresh:
                            two_electron_integrals['values'].append({
                                'key': [w + 1, x + 1,
                                       y + 1, z + 1],
                                'value': float(data.two_body[w, x, y, z])
                            })
        
        return one_electron_integrals, two_electron_integrals

    def _build_yaml_data(self, data):
        """Build the YAML data structure."""
        # Get integral data
        one_electron_integrals, two_electron_integrals = self._get_integral_data(data)
        
        # Build main data structure
        yaml_data = {
            'format': {'version': '0.3'},
            'bibliography': [{'url': 'https://doi.org/10.48550/arXiv.2201.01257'}]
        }
        
        # Build energy offsets
        scf_energy_offset = {
            'units': 'hartree',
            'value': 0.0
        }
        
        energy_offset = {
            'units': 'hartree',
            'value': 0.0
        }
        
        # Build FCI energy placeholder
        fci_energy = {
            "units": "hartree",
            "value": 0.0,
            "upper": 0.0,
            "lower": 0.0
        }
        
        # Build Hamiltonian
        hamiltonian = {
            'one_electron_integrals': one_electron_integrals,
            'two_electron_integrals': two_electron_integrals
        }
        
        # Update coulomb repulsion with energy shift
        coulomb_repulsion = data.coulomb_repulsion.copy()
        coulomb_repulsion['value'] = float(data.coulomb_repulsion['value'] + data.energy_shift
        )
        
        # Build integral sets
        integral_sets = [{
            "metadata": {
                'molecule_name': 'unknown',
                'note': data.note
            },
            "geometry": data.geometry,
            "coulomb_repulsion": coulomb_repulsion,
            "scf_energy": data.scf_energy,
            "n_orbitals": data.n_orbitals,
            "n_electrons": data.n_occ_alpha + data.n_occ_beta,
            "fci_energy": fci_energy,
            "hamiltonian": hamiltonian,
            "scf_energy_offset": scf_energy_offset,
            "energy_offset": energy_offset
        }]
        
        yaml_data['problem_description'] = integral_sets
        return yaml_data
