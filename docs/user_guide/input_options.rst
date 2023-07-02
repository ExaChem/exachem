.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters

.. TODO: Match defaults with code snippets

==================
Input file options
==================

**The input is a JSON file with the following JSON objects documented below. 
The** ``inputs`` **folder in this repository contains many such json files.**

Topics
~~~~~~
| :ref:`Geometry <Geometry>`
| :ref:`Common options <Common>`
| :ref:`Basis set options <Basis>`
| :ref:`SCF options <SCF>`
| :ref:`CC options <CC>`

.. _Geometry:

Geometry
~~~~~~~~

A geometry can be specified as follows:

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 3-10

:coordinates: The atom *symbol* followed by the *x y z* coordinates

:units: The following possible string values are recognized

   * :strong:`"bohr"`: Atomic units (A.U.) ``[default]``
   * :strong:`"angstrom"`: Angstroms, converted to A.U. using the conversion factor 1.889725989.

.. _Common:

Common options
~~~~~~~~~~~~~~

Some common options supported are as follows:

.. literalinclude:: ../../inputs/example.json
   :language: json
   :lines: 16-23

:maxiter: An integer used to specify the maximum number of iterations for all sections below.
   This value can be changed on a per section basis. ``[default: 50]``

**debug**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**output_file_prefix**
   :sep:`|` :aspect:`Type:` String
   :sep:`|` :aspect:`Default:` ""
   :sep:`|`

  Text describing ...

.. _Basis:

Basis set options
~~~~~~~~~~~~~~~~~
Currently support basis sets consisting of generally contracted Cartesian Gaussian functions up to a maximum angular momentum of six (h functions)::

| "basis": {
|   "basisset": "cc-pvdz",
|   "df_basisset": "cc-pv5z-ri",
|   "gaussian_type": "spherical"
| }

**basisset**
   :sep:`|` :aspect:`Type:` String
   :sep:`|` :aspect:`Default:` "sto-3g"
   :sep:`|`

  Text describing ...

**df_basisset**
   :sep:`|` :aspect:`Type:` String
   :sep:`|` :aspect:`Default:` ''
   :sep:`|`

  Used to specify the auxiliary basisset for density fitting.

**gaussian_type**
   :sep:`|` :aspect:`Type:` String
   :sep:`|` :aspect:`Default:` "spherical"
   :sep:`|`

The code recognizes the following possible values for the string variable:

- "spherical": spherical-harmonic (5 d, 7 f, 9 g, ...) angular functions are utilized.
- "cartesian": Cartesian (6 d, 10 f, 15 g, ...) angular functions are utilized.

.. note:: 
   
   The correlation-consistent basis sets were designed using spherical harmonics and to use these, the 
   spherical keyword should be specified. The use of spherical functions also helps eliminate problems 
   with linear dependence.

.. _SCF:

SCF options
~~~~~~~~~~~

The following SCF options are supported. The values listed below are the defaults::

| "SCF": {
|   "charge": 0,
|   "multiplicity": 1,
|   "lshift": 0,
|   "tol_int": 1e-12,
|   "tol_lindep": 1e-5,
|   "conve": 1e-8,
|   "convd": 1e-7,
|   "diis_hist": 10,
|   "force_tilesize": false,
|   "tilesize": 30,
|   "alpha": 0.7,
|   "nnodes": 1,
|   "writem": 10,
|   "restart": true,
|   "noscf": true,
|   "scf_type": "restricted",
|   "debug": false
| }

Note: We do not support any symmetry options to specify point groups. The default is no symmetry (i.e., C1 point group).

**charge**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 0
   :sep:`|`

  Only integer charges are supported. A positive value, :code:`"charge": n`, indicates that *n* electrons are removed from the chemical system. A negative value, :code:`"charge": -n`, indicates that *n* electrons are added to the chemical system.


**multiplicity**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 1
   :sep:`|`

  Specifies the number of singly occupied orbitals for a particular calculation. A value, :code:`"multiplicity": n`, indicates the calculation has *n-1* singly occupied orbitals. The value *n=1* corresponds to a closed-shell singlet, *n=2* corresponds to a doublet, and so on.

**lshift**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 0.0
   :sep:`|`

  Text describing ...

**tol_int**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-12
   :sep:`|`

  Used to determine the integral screening threshold for the evaluation of the energy
  and related Fock-like matrices. The Schwarz inequality is used to screen the product of integrals and density
  matrices in a manner that results in an accuracy in the energy and Fock matrices that approximates the value specified for tol_int.

**tol_lindep**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-5
   :sep:`|`

  Tolerance for detecting the linear dependence of basis set

**conve**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-8
   :sep:`|`

  Specifies the energy convergence threshold.

**convd**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-6
   :sep:`|`

  Specifies the density convergence threshold.

**diis_hist**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 10
   :sep:`|`

  Text describing ...

**force_tilesize**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**tilesize**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 30
   :sep:`|`

  Text describing ...

**alpha**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 0.7
   :sep:`|`

  Text describing ...

**nnodes**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 1
   :sep:`|`

  Text describing ...

**writem**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` :code:`diis_hist`
   :sep:`|`

  Text describing ...

**restart**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**noscf**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**ediis**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**ediis_off**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-5
   :sep:`|`

  Text describing ...

**sad**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**scf_type**
   :sep:`|` :aspect:`Type:` String
   :sep:`|` :aspect:`Default:` "restricted"
   :sep:`|`

  The code recognizes the following possible values for the string variable :
   * "restricted":
   * "unrestricted":

**debug**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...


.. _CD:

CD options
~~~~~~~~~~

Options used in the Cholesky decomposition of atomic-orbital based two-electron integral tensor.

| "CC": {
|   "threshold": 1e-5
| }

**diagonal**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-5
   :sep:`|`

   A diagonal threshold used to terminate the decomposition procedure and truncate the Cholesky vectors.


.. _CC:

CC options
~~~~~~~~~~

The following CC options are supported::

| "CC": {
|   "threshold": 1e-6,
|   "force_tilesize": false,
|   "tilesize": 50,
|   "lshift": 0,
|   "ndiis": 5,
|   "ccsd_maxiter": 50,
|   
|   "readt": false,
|   "writet": false,
|   "writev": false,
|   "writet_iter": 5,
|   
|   "debug": false,
|   "profile_ccsd": false,
|   
|   "CCSD(T)": {
|     "ccsdt_tilesize": 32
|   }
| }

**threshold**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 1e-6
   :sep:`|`

  Specifies the convergence threshold of iterative solutions of amplitude equations. The threshold refers to the norm of residual, namely, the deviation from the amplitude equations.

**force_tilesize**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**tilesize**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 50
   :sep:`|`

  Text describing ...

**lshift**
   :sep:`|` :aspect:`Type:` Double
   :sep:`|` :aspect:`Default:` 0.0
   :sep:`|`

  The level shift option that increases small orbital energy differences used in calculating the updates for cluster amplitudes. When calculating ground states with multi-configurational character or if convergence is slow, one may need to specify a typical values for lshift between 0.3 and 0.5.

**ndiis**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 5
   :sep:`|`

  The number iterations in which a DIIS extrapolation is performed to accelerate the convergence of excitation amplitudes. The default value is 5, which means in every five iteration, one DIIS extrapolation is performed (and in the rest of the iterations, Jacobi rotation is used). When zero or negative value is specified, the DIIS is turned off. It is not recommended to perform DIIS every iteration, whereas setting a large value for this parameter necessitates a large memory space to keep the excitation amplitudes of previous iterations.

**ccsd_maxiter**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default: 50`
   :sep:`|`

  The maximum number of iterations performed during the iterative solutions of amplitude equations.

**readt**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**writet**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**writev**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:`
   :sep:`|`

  Text describing ...

**writet_iter**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` :code:`ndiis`
   :sep:`|`

  Text describing ...

**debug**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:`
   :sep:`|`

  Text describing ...

**profile_ccsd**
   :sep:`|` :aspect:`Type:` Bool
   :sep:`|` :aspect:`Default:` False
   :sep:`|`

  Text describing ...

**ccsdt_tilesize**
   :sep:`|` :aspect:`Type:` Integer
   :sep:`|` :aspect:`Default:` 28
   :sep:`|`

  Text describing ...
