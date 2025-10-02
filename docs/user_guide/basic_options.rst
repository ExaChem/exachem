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
   "operation": ["energy"]
 }

A task automatically runs the tasks it depends on. For e.g. if **ccsd** is enabled, it automatically runs the tasks **scf** (hartree fock) and **cd_2e** (cholesky decomposition of the 2e integrals). 

:ducc: The *ducc* task has two options that can be specified.
   
   * :strong:`default`  : Runs the double unitary CC formalism (DUCC).
   * :strong:`qflow`  : Runs the quantum flow variant.

:operation: ``[default=energy]`` Specifies the calculation that will be performed in the enabled task.

   * :strong:`energy`  : Computes the single point energy.
   * :strong:`gradient`: Computes numerical gradients for the level of theory specified.
   * :strong:`optimize`: Minimize the energy by varying the molecular structure.


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

