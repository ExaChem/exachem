.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters


SCF
===

| :ref:`SCF options <SCF>`

.. | :ref:`DFT options <DFT>`

.. _SCF:

The ExaChem self-consistent field (SCF) module computes closed-shell restricted Hartree-Fock (RHF) wavefunctions and spin-unrestricted Hartree-Fock (UHF) wavefunctions. 
It also computes closed shell and open shell densities and Kohn-Sham orbitals.
The values listed below are the defaults where few options are automatically adjusted based on the system size as explained later in this section.

.. code-block:: json

 "SCF": {
   "charge": 0,
   "multiplicity": 1,
   "lshift": 0,
   "tol_int": 1e-22,
   "tol_sch": 1e-10,
   "tol_lindep": 1e-5,
   "conve": 1e-8,
   "convd": 1e-7,
   "diis_hist": 10,
   "force_tilesize": false,
   "tilesize": 30,
   "damp": 100,
   "nnodes": 1,
   "writem": 1,
   "restart": false,
   "noscf": false,
   "scf_type": "restricted",
   "xc_type": [],
   "xc_grid_type": "UltraFine",
   "debug": false,
    "guess": {
      "atom_options":{
        "Fe1": [1,2],
        "H": [1,2]
      }
    },   
   "PRINT": {
     "mos_txt" : false,
     "mulliken": false,
     "mo_vectors" : [false,0.15]
   }
 }

.. note:: We do not support any symmetry options to specify point groups. The default is no symmetry (i.e., C1 point group).


:charge: An integer ``[default=0]``

   * :strong:`charge:  n` indicates that **n** electrons are removed from the chemical system. 
   * :strong:`charge: -n` (a negative value) indicates that **n** electrons are added to the chemical system.


:multiplicity: An integer ``[default=1]``. Specifies the number of singly occupied orbitals for a particular calculation. A value, :code:`"multiplicity": n`, indicates the calculation has *n-1* singly occupied orbitals. The value *n=1* corresponds to a closed-shell singlet, *n=2* corresponds to a doublet, and so on.

:lshift: ``[default=0]`` level shift factor (a `real` number) denoting the amount of shift applied to the diagonal elements of the unoccupied block of the Fock matrix. 

:conve: ``[default=1e-8]``  Specifies the energy convergence threshold.

:convd: ``[default=1e-6]``  Specifies the density convergence threshold.

:tol_int: ``[default=1e-22]`` Used to determine the integral primitive screening threshold for the evaluation of the energy and related Fock-like matrices.

:tol_sch: ``[default=min(1e-10, 1e-2 * conve)]``
  The Schwarz inequality is used to screen the product of integrals and density
  matrices in a manner that results in an accuracy in the energy and Fock matrices that approximates the value specified for **tol_sch**.

:tol_lindep: ``[default=1e-5]``  Tolerance for detecting the linear dependence of basis set.

:diis_hist: ``[default=10]`` Specifies the number of DIIS history entries to store for the fock and error matrices.

:force_tilesize: ``[default=false]``

:tilesize: The tilesize for the AO dimension. An integer value that is automatically set to ``ceil(Nbf * 0.05)``. If **force_tilesize=true**, 
   the value specified by the user is respected. It is recommended to let the SCF module automatically determine this value.

:damp: damping (mixing) factor for the density matrix where :math:`0 \leq \alpha \leq 100`.  Specifies the percentage of the current iterations density mixed with the previous iterations density. ``default=100`` indicates no damping.

:writem: ``[default=1]`` An integer specifying the frequency (as number of iterations) after which the movecs and density matrices are written to disk for restarting the calculation.

:restart: ``[default=false]`` indicates the calculation be restarted.

:noscf: ``[default=false]`` typically used together with `restart` option. Computes only the SCF energy upon restart.

:scf_type: ``[default=restricted]``  The following values are supported

   * :strong:`restricted`: for closed-shell restricted Hartree-Fock (RHF) calculation
   * :strong:`unrestricted`: for spin-unrestricted Hartree-Fock (UHF) calculation

:xc_type: ``[default=[]]`` A list of strings specifying the exchange and correlation functionals for DFT calculations using `GauXC <https://github.com/wavefunction91/GauXC>`_.
   The list of available functionals using the `builtin` backend can be found at the `ExchCXX <https://github.com/wavefunction91/ExchCXX>`_ repository.
   The `Libxc` backend is restricted to the list of LDA and GGA functionals without range separation available at `Libxc <https://tddft.org/programs/libxc/functionals/libxc-6.2.2/>`_.

:xc_grid_type: ``[default=UltraFine]`` Specifies the quality of the numerical integration grid. The following values are supported

   * :strong:`Fine`: 75 radial shells with 302 angular points per shell.
   * :strong:`UltraFine`: 99 radial shells with 590 angular points per shell.
   * :strong:`SuperFine`: 175 radial shells with 974 angular points per shell for first row elements and 250 radial shells with 974 Lebedev points per shell for the rest.

   All **xc_grid_type** options use a `Mura-Knowles` radial quadrature, a `Lebedev-Laikov` angular quadrature, a `Laqua-Kussmann-Ochsenfeld` partitioning scheme, and a `Robust` pruning method.

:debug: ``[default=false]`` enable verbose printing for debugging a calculation.

:nnodes: On a distributed machine, the number of processors for an SCF run is chosen by default depending on the problem size (i.e. number of basis functions **Nbf**).
   If a larger number of processors than required are used, the SCF module automatically chooses a smaller subset of processors for the calculation. 
   The SCF module automatically chooses the number of processors to be ``50% * Nbf``. This option allows to override this behavior and choose a larger set of processors by specifying 
   the percentage (as an integer value) of the total number of processors to use.  

:guess: This block allows specifying options for individual atoms for the initial guess specified as atom symbol with charge and multiplicity values.

:PRINT: This block allows specifying a couple of printing options. When enabled, they provide the following

   * :strong:`mos_txt`: Writes the coeffcient matrix (lcao), transformed core Hamilotonian, Fock, and 2e integral tensors in molecular spin-orbital (MSO) basis to disk as text files.
   * :strong:`mulliken`: Mulliken population analysis will be carried out on both the input and output densities, providing explicit population analysis of the basis functions.
   * :strong:`mo_vectors`: Enables molecular orbital analysis. Prints all orbitals with energies :math:`\geq` the specified threshold.

 
