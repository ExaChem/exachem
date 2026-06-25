# ExaChem integration examples

Ready-to-run inputs for driving ExaChem from an external simulation engine. In
these workflows ExaChem runs as a persistent energy/force **backend**: the
external driver orchestrates the simulation (molecular dynamics, geometry
optimization, ...) and asks ExaChem for the energy and forces at each step. See
the *Integrations* page of the user guide for details.

All examples use the same CH4 molecule. The geometry the driver reads
(`ch4.xyz`) and the geometry in the ExaChem JSON must describe the **same atoms
in the same order** - only the coordinate values are exchanged over the socket
(the JSON coordinate values and units are overwritten and do not matter).

ExaChem is the socket client, so **start the driver first**, then launch ExaChem.

## Files

| File            | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `ch4.xyz`       | Shared initial geometry (read by the i-PI / ASE driver)             |
| `ipi.xml`       | i-PI server config (INET socket on `localhost:31415`)               |
| `ch4_ipi.json`  | ExaChem input, `operation: ["ipi", "analytical"]`                   |
| `run_ase.py`    | ASE NVT MD driver (UNIX socket `/tmp/ipi_ase_exachem_socket`)        |
| `ch4_ase.json`  | ExaChem input, `operation: ["ase", "analytical"]`                   |

## i-PI (NVT molecular dynamics)

```bash
# 1. start the i-PI server (creates the socket and waits)
i-pi ipi.xml &

# 2. connect ExaChem as the force provider
mpirun -np <N> ExaChem ch4_ipi.json
```

i-PI writes its results according to the `<output prefix='ch4'>` block in
`ipi.xml`: the properties (step, time, temperature, energies) go to `ch4.out` and
the trajectory to `ch4.pos_0.xyz`.

## ASE (NVT molecular dynamics)

```bash
# 1. start the ASE workflow (creates the socket and waits)
python run_ase.py &

# 2. connect ExaChem as the force provider
mpirun -np <N> ExaChem ch4_ase.json
```

`run_ase.py` runs NVT molecular dynamics (Langevin) by default; it logs the
trajectory properties to `md.log` and prints the per-step energy and temperature
to stdout. The script also sketches other ASE workflows (geometry optimization,
NEB, vibrational analysis) - uncomment the one you want.
