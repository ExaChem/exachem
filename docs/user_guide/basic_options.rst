.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters

.. TODO: Match defaults with code snippets

========================
Basic input options
========================

**The input is a JSON file with the following JSON objects documented below. 
The** ``inputs`` **folder in this repository contains json files for several molecular systems.**

| :ref:`Geometry <Geometry>`
| :ref:`Common options <Common>`
| :ref:`Basis set options <Basis>`

.. _Geometry:

Geometry
~~~~~~~~

A geometry can be specified as follows:

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 2-14

:coordinates: The element *symbol* followed by the *x y z* coordinates. Additional characters can be added to the element symbol to distinguish between atoms of the same element. For example, the atom labels ``O`` and  ``O34`` will both be interpreted as oxygen atoms, but the program will keep their distinction for further use. An atom label starting with "bq" (e.g. ``bqH``) is interpreted as a ghost atom that contributes only basis functions, not electrons.

:pcharges: The *x y z* coordinates followed by the charge *q* of point charges providing an external potential.

:units: The following possible string values are recognized ``[default=angstrom]``

   * :strong:`"bohr"`: Atomic units (A.U.) 
   * :strong:`"angstrom"`: Angstroms, converted to A.U.

:ang2au: Angstrom to A.U conversion factor. ``[default=1.8897261259077822]``

:natoms_max: Geometry analysis will only be performed if the number of atoms is less than or equal to this value. ``[default=30]``

.. _Common:

Common options
~~~~~~~~~~~~~~

Some common options supported are as follows:

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 20-28

:maxiter: An integer used to specify the maximum number of iterations for all sections below.
   This value can be changed on a per section basis. ``[default: 100]``

:debug: A boolean used to turn on debugging mode. ``[default: false]``

:file_prefix: A string indicating the prefix for the name of the workspace folder where the results of a run are stored.
   It also forms the prefix for the files written to the workspace folder. The *default prefix* is the name of the input file without the *.json* extension.

:output_dir: A string indicating the path to an existing directory where the where the results of a run are stored. If not provided, the default is the current working directory.

.. _Basis:

Basis set options
~~~~~~~~~~~~~~~~~
Currently support basis sets consisting of contracted Gaussian functions up to a maximum angular momentum of six (h functions).
Spherical-harmonic (5 *d*, 7 *f*, 9 *g*, ...) angular functions are utilized by default.

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 30-40

:basisset: String specifying the basis set name. Parsing of the basis set will be handled by *Libint*, which expects to find a ``<basisset>.g94`` file with Gaussian-style format and located at ``$LIBINT_DATA_PATH/basis`` or ``<libint2_install_prefix>/share/libint/<libint2_version>/basis`` (if ``LIBINT_DATA_PATH`` is not defined).

:df_basisset: String specifying the density-fitting basis set name. Parsing of the basis set will be handled by *Libint*, which expects to find a ``<basisset>.g94`` file with Gaussian-style format and located at ``$LIBINT_DATA_PATH/basis`` or ``<libint2_install_prefix>/share/libint/<libint2_version>/basis`` (if ``LIBINT_DATA_PATH`` is not defined).

:atom_basis: Specify the basis set for individual atoms. The full strings specified in `Geometry`_  will be used to distinguish different atoms of the same element.

:atom_ecp: For ECPs only. Specify the ECP basis set for individual atoms. The specified file should follow the NWChem format from the basis set exchange website. Parsing of the ECP basis set expects to find a ``<basisset>.ecp`` file in NWChem format and located at ``$LIBINT_DATA_PATH/basis`` or ``<libint2_install_prefix>/share/libint/<libint2_version>/basis`` (if ``LIBINT_DATA_PATH`` is not defined). The corresponding basis files are expected to contain the ECP block. Only the ECP block of these basis files is parsed and everthing else is ignored.

.. note:: All basis sets from the Basis Set Exchange (BSE) are already installed by ExaChem and are available for use. If you wish to add your own custom basis set files to be used in any of the basis options documented in this section, they should be copied to the ``$LIBINT_DATA_PATH/basis`` folder mentioned above. Spaces are not allowed in the basis set filenames. Replace a space with an underscore in the filename when copying a custom basis set file to the ``$LIBINT_DATA_PATH/basis`` folder. However, the basis set name in the input json file can be specified with or without the space. In addition, for custom augmented basis set files, the filenames must start with the prefix ``ec-`` (e.g. ``ec-aug-cc-pvdz.g94`` or ``ec-aug-cc-pvdz.ecp``). The basis set name in the input json file can be specified with or without the ``ec-`` prefix.


.. _TASK:

TASK Options
~~~~~~~~~~~~

The **TASK** block of the input file specifies the method to run. Only a single task can be enabled at once. The supported task options are shown below.

.. code-block:: json

 "TASK": {
   "scf": true,
   "mp2": false,
   "cc2": false,
   "fcidump": false,
   "cd_2e": false,
   "ducc": [false, "default"],
   "ccsd": false,
   "ccsd_t": false,
   "ccsd_lambda": false,
   "eom_ccsd": false,
   "rteom_ccsd": false,
   "gfccsd": false,
   "embedding": false,
   "operation": ["energy"]
 }

A task automatically runs the tasks it depends on. For e.g. if **ccsd** is enabled, it automatically runs the tasks **scf** (hartree fock) and **cd_2e** (cholesky decomposition of the 2e integrals). 

:ducc: The *ducc* task has two options that can be specified.
   
   * :strong:`default`  : Runs the double unitary CC formalism (DUCC).
   * :strong:`qflow`  : Runs the quantum flow variant.

:operation: ``[default=energy]`` Specifies the calculation that will be performed in the enabled task. ``operation`` is a list whose first element is the calculation type; the optional remaining elements select the gradient method and, for an optimization, the optimizer.

   * :strong:`energy`  : Computes the single point energy.
   * :strong:`gradient`: Computes the gradients for the level of theory specified. An optional second list element selects the gradient method:

     * :strong:`analytical` ``[default for HF/DFT]`` : Use analytical gradients.
     * :strong:`numerical` ``[default for all other methods]`` : Use numerical (finite-difference) gradients.

     For example, ``"operation": ["gradient", "analytical"]``.

   * :strong:`optimize`: Minimize the energy by varying the molecular structure. Optional second and third list elements select the gradient method and the optimizer, respectively.

     * Second element - gradient method: :strong:`analytical` ``[default for HF/DFT]`` or :strong:`numerical` ``[default for all other methods]`` (same as for **gradient**).
     * Third element - optimizer:

       * :strong:`geometric` ``[default when Python bindings are enabled]`` : ExaChem calls the Python `geomeTRIC <https://geometric.readthedocs.io>`_ library to perform the geometry optimization.
       * :strong:`pyberny` : Use the native C++ PyBerny implementation in ExaChem.

     For example, ``"operation": ["optimize", "analytical", "geometric"]`` is the default when Python bindings are enabled, while ``"operation": ["optimize", "analytical", "pyberny"]`` switches to the native C++ PyBerny optimizer.

   * :strong:`ipi` / :strong:`ase`: Connect ExaChem to an external simulation driver. Instead of computing a single result and exiting, ExaChem runs as a persistent server that returns the energy and forces for each geometry the driver requests, until the driver disconnects. The workflow itself (e.g. molecular dynamics, NEB, path-integral MD) is configured outside ExaChem - in the i-PI XML input or the ASE Python script. An optional second list element selects the gradient method (:strong:`analytical` or :strong:`numerical`, with the same defaults as for **gradient**).

     * :strong:`ipi`  : Act as an `i-PI <https://ipi-code.org>`_ client, connecting to an i-PI server over a socket.
     * :strong:`ase`  : Act as a force/energy backend for `ASE <https://ase-lib.org>`_ via its ``SocketIOCalculator`` over a UNIX domain socket.

     For example, ``"operation": ["ipi", "analytical"]`` or ``"operation": ["ase", "numerical"]``.


.. _DPLOT:

DPLOT Options
~~~~~~~~~~~~~

This section is used to obtain the plots of various types of electron densities (or orbitals) of the molecule. 
The electron density is calculated on a specified set of grid points using the molecular orbitals from SCF or DFT calculation. 
The output file is in the Gaussian Cube format.

.. code-block:: json

 "DPLOT": {
   "cube": false,
   "density": "total",
   "orbitals": 0
 }

:cube: A boolean used to indicate whether a cube file should be written. ``[default: false]``

:density: Plot total density by default when **cube=true**. The supported string values that specify what kind of density is to be computed are ``"total"`` and ``"spin"``.

:orbitals: Specify the highest occupied orbitals for both spins to be plotted. ``[default: 0]``


.. _PDOS:

PDOS options
~~~~~~~~~~~~

This section is used to obtain the projected density of states after an SCF calculation. The output file is in text format with the first block presenting the total density of states followed by *natom* blocks presenting atom-specific l-decomposed projected density of states. For restricted calculations, the columns follow the order *s p d f ...*. For unrestricted calculations the columns follow the order *s(alpha) s(beta) p(alpha) p(beta) d(alpha) d(beta) ...*.

.. code-block:: json

 "PDOS": {
   "emin": 0.0,
   "emax": 0.0,
   "npoints": 100,
   "distribution": "lorentzian",
   "fwhm": 0.0055
 }

:emin: The minimum energy (in Hartrees) to compute the projected density of states. If *emin* is not given, then the projected density of states will be computed starting from 1 Ha below the Fermi level.

:emax: The maximum energy (in Hartrees) to compute the projected density of states. If *emax* is not given, then the projected density of states will be computed up to 1 Ha above the Fermi level.

:npoints: ``[default=100]`` The number of bins in which the range *emin*-*emax* is subdivided.

:distribution: ``[default="lorentzian"]`` A string that identifies the smearing function used to broaden the discrete energy spectrum.

    * :strong:`lorentzian` : A unit-area Lorentzian function is used.
    * :strong:`gaussian`   : A unit-area Gaussian function is used.

:fwhm: ``[default=0.0055]`` The full-width at half-maximum value in Hartree.


.. _EMBEDDING:

EMBEDDING options
~~~~~~~~~~~~~~~~~

This section is used to define projector-based quantum embedding parameters.

.. code-block:: json

 "EMBEDDING": {
   "lambda": 1.0e6,
   "use_ksref": false
   "projector": "huzinaga",
   "high_level": ["CCSD"],
   "pao_thresh1": 0.01,
   "pao_thresh2": 0.005,
   "active_atoms": [],
   "freeze_projected": false,
 }

:lambda: ``[default=1.0e6]`` The level-shift value used to move the eigenvalues of the projected orbitals.

:use_ksref: If **true**, use a Kohn-Sham reference to start post-HF calculations.

:projector: Type of projector used to keep orthogonality between active and inactive subsystems.

    * :strong:`"huzinaga"`
    * :strong:`"level"`

:high_level: A string specifying the high-level calculation. It can be a combination of exchange-correlation functionals, or a string identifying the post-HF method.

:pao_thresh1: Threshold to control the number of PAOs kept.

:pao_thresh2: Threshold to control the number of PAOS kept.

:active_atoms: A list of atom indices that defines the active region.

:freeze_projected: Wether the subsequent post-HF method freeze the projected orbitals (occupied and unoccupied) from the environment.
