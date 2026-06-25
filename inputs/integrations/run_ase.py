#!/usr/bin/env python3
"""Drive an ASE workflow using ExaChem as the force/energy backend.

This example runs NVT molecular dynamics by default; the body below also sketches
other ASE workflows (geometry optimization, NEB, vibrational analysis) - uncomment
the one you want.

ASE opens the UNIX domain socket /tmp/ase_exachem_socket and waits for ExaChem to
connect. Start this script first, then launch ExaChem against the matching input:

    python run_ase.py &
    mpirun -np <N> ExaChem ch4_ase.json

The geometry read here (ch4.xyz) must list the same atoms, in the same order, as
the ExaChem input (ch4_ase.json).
"""
from ase.io import read
from ase.md import MDLogger
from ase.calculators.socketio import SocketIOCalculator

atoms = read('ch4.xyz')  # angstrom, standard XYZ

with SocketIOCalculator(unixsocket='ase_exachem_socket') as calc:
    atoms.calc = calc

    # --- pick ONE of these depending on what you want ---

    # geometry optimization
    # from ase.optimize import BFGS
    # opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')
    # opt.run(fmax=0.01)  # hartree/bohr convergence threshold

    # NVT MD
    from ase.md.langevin import Langevin
    from ase import units
    dyn = Langevin(atoms, 0.5*units.fs, temperature_K=300, friction=0.01, fixcm=False)
    # logger writes to file
    dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False, 
                        peratom=False), interval=1)
    
    # print to stdout every step
    def print_step():
        print(f"  step={dyn.nsteps:4d}  E={atoms.get_potential_energy():.6f} eV  "
              f"T={atoms.get_temperature():.2f} K")
    
    dyn.attach(print_step, interval=1)    
    dyn.run(50)

    # NEB
    # from ase.neb import NEB
    # ... 

    # vibrational analysis
    # from ase.vibrations import Vibrations
    # vib = Vibrations(atoms)
    # vib.run()
    # vib.summary()