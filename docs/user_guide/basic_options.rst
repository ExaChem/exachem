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
   :lines: 3-10

:coordinates: The element *symbol* followed by the *x y z* coordinates. Additional characters can be added to the element symbol to distinguish between atoms of the same element. For example, the strings ``O`` and  ``O34`` will both be interpreted as oxygen atoms, but the program will keep their distinction for further use.

:units: The following possible string values are recognized ``[default=angstrom]``

   * :strong:`"bohr"`: Atomic units (A.U.) 
   * :strong:`"angstrom"`: Angstroms, converted to A.U. using the conversion factor 1.8897259878858.

.. _Common:

Common options
~~~~~~~~~~~~~~

Some common options supported are as follows:

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 16-23

:maxiter: An integer used to specify the maximum number of iterations for all sections below.
   This value can be changed on a per section basis. ``[default: 50]``

:debug: A boolean used to turn on debugging mode. ``[default: false]``

:file_prefix: A string indicating the prefix for the name of the workspace folder where the results of a run are stored.
   It also forms the prefix for the files written to the workspace folder. The *default prefix* is the name of the input file without the *.json* extension.

:output_dir: A string indicating the path to an existing directory where the where the results of a run are stored. If not provided, the default is the current working directory.

.. _Basis:

Basis set options
~~~~~~~~~~~~~~~~~
Currently support basis sets consisting of contracted Gaussian functions up to a maximum angular momentum of six (h functions).
Spherical-harmonic (5 *d*, 7 *f*, 9 *g*, ...) angular functions are utilized by default.

.. code-block:: json

 "basis": {
   "basisset": "cc-pvdz",
   "df_basisset": "cc-pvdz-ri",
   "basisfile": "/path/to/basis_file_with_ecps",
   "atom_basis": {
      "H": "cc-pvtz",
      "O": "aug-cc-pvtz"
   }
 }

:basisset: String specifying the basis set name. Parsing of the basis set will be handled by *Libint*, which expects to find a ``<basisset>.g94`` file with Gaussian-style format and located at ``$LIBINT_DATA_PATH/basis`` or ``<libint2_install_prefix>/share/libint/<libint2_version>/basis`` (if ``LIBINT_DATA_PATH`` is not defined).

:df_basisset: String specifying the density-fitting basis set name. Parsing of the basis set will be handled by *Libint*, which expects to find a ``<basisset>.g94`` file with Gaussian-style format and located at ``$LIBINT_DATA_PATH/basis`` or ``<libint2_install_prefix>/share/libint/<libint2_version>/basis`` (if ``LIBINT_DATA_PATH`` is not defined).

:basisfile: Currently used for ECPs only. This is a basis file containing the ECP block. Only the ECP block of this file is parsed and everthing else is ignored. The specified file should follow the NWChem format from the basis set exchange website.

:atom_basis: Specify the basis set for individual atoms. The full strings specified in `Geometry`_  will be used to distinguish different atoms of the same element.

..  :df_basisset: Used to specify the auxiliary basisset for density fitting.

.. _TASK:

TASK Options
~~~~~~~~~~~~

The **TASK** block of the input file specifies the method to run. Only a single task can be specified at once. The supported task options are shown below.

.. code-block:: json

 "TASK": {
   "scf": true,
   "mp2": false,
   "cc2": false,
   "fcidump": false,
   "cd_2e": false,
   "ducc": false,
   "ccsd": false,
   "ccsd_t": false,
   "ccsd_lambda": false,
   "eom_ccsd": false,
   "rteom_ccsd": false,
   "gfccsd": false,
   "operation": ["energy"]
 }

A task automatically runs the tasks it depends on. For e.g. if **ccsd** is enabled, it automatically runs the tasks **scf** (hartree fock) and **cd_2e** (cholesky decomposition of the 2e integrals).

:operation: ``[default=energy]`` Specifies the calculation that will be performed in the task.

   * :strong:`energy`  : Computes the single point energy.
   * :strong:`gradient`: Computes numerical gradients for the level of theory specified.


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

