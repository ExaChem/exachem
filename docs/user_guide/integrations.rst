============
Integrations
============

ExaChem can act as a compute *backend* for external simulation drivers: rather
than running a calculation and exiting, it connects to an external program over a
socket and runs as a persistent server, returning the energy and forces for each
geometry the driver requests, until the driver disconnects. This lets the external
tool orchestrate a workflow it specializes in - molecular dynamics, path-integral
MD, nudged elastic band, etc. - while ExaChem provides the electronic-structure
energies and gradients.

Two drivers are supported: `i-PI <https://ipi-code.org>`_ and
`ASE <https://ase-lib.org>`_. Both are selected through the **TASK** ``operation``
option (see :ref:`TASK`); ``ipi`` and ``ase`` are top-level operations, with an
optional second element choosing the gradient method
(``analytical`` for HF/DFT, ``numerical`` otherwise).

Ready-to-run examples for both drivers (the input files shown below, for a CH4
molecule) are provided in ``inputs/integrations/``.

Installing the drivers
======================

The drivers are external Python packages, installed independently of ExaChem.
ASE is available from PyPI:

.. code-block:: bash

   pip3 install --upgrade --user ase

i-PI is also available from PyPI:

.. code-block:: bash

   pip3 install -U --user ipi

How it works
============

In both cases ExaChem is the socket **client**: the external driver opens the
socket and waits, and ExaChem connects to it. The practical consequence is an
ordering requirement - **start the external driver first**, then launch ExaChem.

The ExaChem input JSON is still required to set up the calculation: the enabled
task (``scf``, ``mp2``, ``ccsd``, ...) defines the level of theory, and the
charge, multiplicity, and basis are taken from it as usual. Its geometry block
must list the atoms in the **same count, element, and order** as the driver's
structure (e.g. ``init.xyz``), because the driver sends only a flat list of
coordinates that ExaChem maps onto those atoms in order, overwriting them at every
step. The coordinate *values* and their ``units`` in the JSON therefore do not
matter - they are replaced before the first energy/force evaluation. i-PI and ASE
never read the ExaChem input file, and ExaChem never reads the driver's geometry
file; the two only need to describe the same atoms in the same order.

All quantities exchanged over the socket are in atomic units (positions in bohr,
energy in hartree, forces in hartree/bohr).

The socket endpoints are currently fixed in ExaChem, so the driver must be
configured to match them:

================  ==================================================
Driver            Endpoint
================  ==================================================
``ipi``           INET socket, host ``localhost``, port ``31415``
``ase``           UNIX domain socket ``/tmp/ipi_ase_exachem_socket``
================  ==================================================

i-PI
====

Configure the i-PI server with a socket force field that matches the endpoint
above (an internet socket on ``localhost:31415``). i-PI reads the initial
geometry and the simulation cell from its own input - these must be consistent
with the ExaChem geometry (use a large cell for gas-phase molecules). The example
below runs a short NVT molecular-dynamics trajectory:

.. code-block:: xml

   <simulation verbosity='medium'>
     <total_steps> 100 </total_steps>

     <!-- "name" is arbitrary but must match the <force> reference below -->
     <ffsocket name='exachem' mode='inet'>
       <address> localhost </address>
       <port> 31415 </port>
     </ffsocket>

     <system>
       <initialize nbeads='1'>
         <file mode='xyz' units='angstrom'> init.xyz </file>
         <cell mode='abc' units='angstrom'> [20.0, 20.0, 20.0] </cell>
         <velocities mode='thermal' units='kelvin'> 300 </velocities>
       </initialize>
       <forces>
         <force forcefield='exachem'/>
       </forces>
       <motion mode='dynamics'>
         <dynamics mode='nvt'>
           <timestep units='femtosecond'> 0.5 </timestep>
           <thermostat mode='langevin'>
             <tau units='femtosecond'> 100 </tau>
           </thermostat>
         </dynamics>
       </motion>
     </system>
   </simulation>

Then start i-PI, and once it is listening launch ExaChem against the same input
geometry/method with ``"operation": ["ipi", ...]``:

.. code-block:: bash

   # 1. start the i-PI server (it creates the socket and waits)
   i-pi input.xml &

   # 2. connect ExaChem as the force provider
   mpirun -np <N> ExaChem input.json

ASE
===

On the ASE side, attach a ``SocketIOCalculator`` configured to listen on the same
socket file ExaChem connects to, ``/tmp/ipi_ase_exachem_socket``. ASE creates the
socket and waits for ExaChem to connect:

.. note::

   ASE's ``SocketIOCalculator`` speaks the i-PI socket protocol, and for UNIX
   domain sockets it prepends ``/tmp/ipi_`` to the ``unixsocket`` name. So
   ``unixsocket="ase_exachem_socket"`` corresponds to the socket path
   ``/tmp/ipi_ase_exachem_socket`` that ExaChem connects to.

.. code-block:: python

   from ase.io import read
   from ase.optimize import BFGS
   from ase.calculators.socketio import SocketIOCalculator

   atoms = read("init.xyz")  # same atoms/order as the ExaChem input

   # configure the calculator to listen on /tmp/ipi_ase_exachem_socket
   with SocketIOCalculator(unixsocket="ase_exachem_socket") as calc:
       atoms.calc = calc
       BFGS(atoms).run(fmax=0.05)

Then start the ASE script, and once it is listening launch ExaChem with
``"operation": ["ase", ...]``:

.. code-block:: bash

   # 1. start the ASE workflow (it creates the socket and waits)
   python run_ase.py &

   # 2. connect ExaChem as the force provider
   mpirun -np <N> ExaChem input.json
