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

:coordinates: The atom *symbol* followed by the *x y z* coordinates

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

:output_file_prefix: A string indicating the prefix for the name of the output folder where the results of a run are stored.
   The *default prefix* is the name of the input file without the *.json* extension.

.. _Basis:

Basis set options
~~~~~~~~~~~~~~~~~
Currently support basis sets consisting of generally contracted Cartesian Gaussian functions up to a maximum angular momentum of six (h functions)::

 "basis": {
   "basisset": "cc-pvdz",
   "gaussian_type": "spherical",
   "atom_basis": {
     "H": "cc-pvtz",
     "O": "aug-cc-pvtz"
    }
 }

:basisset: String specifying the basis set name.

:atom_basis: Specify the basis set for individual atoms.

..  :df_basisset: Used to specify the auxiliary basisset for density fitting.

:gaussian_type: The following values are recognized

   * :strong:`spherical (default)`: spherical-harmonic (5 d, 7 f, 9 g, ...) angular functions are utilized.
   * :strong:`cartesian`: Cartesian (6 d, 10 f, 15 g, ...) angular functions are utilized.

.. note:: 
   
   The correlation-consistent basis sets were designed using spherical harmonics and to use these, the spherical keyword should be specified. 

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
   "gfccsd": false
 }

A task automatically runs the tasks it depends on. For e.g. if **ccsd** is enabled, it automatically runs the tasks **scf** (hartree fock) and **cd_2e** (cholesky decomposition of the 2e integrals).

