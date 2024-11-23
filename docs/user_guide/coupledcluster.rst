.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters


Coupled Cluster Calculations
============================

| :ref:`CC options <CC>`

.. _CC:

CCSD 
~~~~

The following CCSD options are supported. The remaining CC methods (CC2, Lambda, EOMCC, etc) inherit these options in addition to having their own options which are elaborated later in this document.

.. code-block:: json

 "CC": {
   "threshold": 1e-6,
   "tilesize": 50,
   "lshift": 0,
   "ndiis": 5,
   "ccsd_maxiter": 50,
 
   "readt": false,
   "writet": false,
   "writet_iter": 5,
 
   "debug": false,
   "profile_ccsd": false,

   "freeze": {
      "atomic": false,
      "core": 0,
      "virtual": 0
   },   

   "PRINT": {
      "tamplitudes": [false,0.05],
      "ccsd_diagnostics" : false,
      "rdm": [1,2]
   }   
  }


:threshold: ``[default=1e-6]`` Specifies the convergence threshold of iterative solutions of amplitude equations. The threshold refers to the norm of residual, namely, the deviation from the amplitude equations.

:tilesize: ``[default=40]`` The tilesize for the MO or MSO space. An integer value that is automatically set unless specified explicitly by the user. It is recommended to let the CC module determine this value automatically.

   .. warning::

      The automatically determined tilesize may vary depending on node count and the number of processes per node. When restarting a calculation with different node counts, the user is expected to explicitly set the tilesize to match the value used in the initial run.

:lshift: ``[default=0]`` The level shift option that increases small orbital energy differences used in calculating the updates for cluster amplitudes. When calculating ground states with multi-configurational character or if convergence is slow, one may need to specify a typical values for lshift between 0.3 and 0.5.

:ndiis: ``[default=5]`` The number iterations in which a DIIS extrapolation is performed to accelerate the convergence of excitation amplitudes. The default value is 5, which means in every five iteration, one DIIS extrapolation is performed (and in the rest of the iterations, Jacobi rotation is used). When zero or negative value is specified, the DIIS is turned off. It is not recommended to perform DIIS every iteration, whereas setting a large value for this parameter necessitates a large memory space to keep the excitation amplitudes of previous iterations.

:ccsd_maxiter: ``[default=50]`` The maximum number of iterations performed during the iterative solutions of amplitude equations.

:writet: ``[default=false]`` Writes the T1,T2 amplitude tensors and the 2e integral tensor to disk to be used later for restarting a CC calculation. Currently, the cholesky decomposition module uses this option as well to write the 2e integral tensor to disk. Enabling this option implies restart. 

:readt: ``[default=false]`` Reads the T1,T2 amplitude tensors and the 2e integral tensor from disk for restarting a CC calculation. Enabling this option implies restart. Does not need to be explicitly specified if ``writet=true``.

:writet_iter: ``[default=ndiis]`` This option requires ``writet=true``. An integer that determines the frequency of writing tensors to disk for restart purposes. The tensors are written to disk after every *writet_iter* iterations. 

:debug: ``[default=false]`` enable verbose printing for debugging a CC calculation.

:profile_ccsd: ``[default=false]`` When enabled, writes a csv file containing the performance data for every tensor contraction. Useful for profiling contractions in a single iteration by setting ``ccsd_maxiter=1``.

:freeze: This block allows specifying freezing options. Some of the lowest-lying core orbitals and/or some of the highest-lying virtual orbitals may be excluded using this block. No orbitals are frozen by default.

   * :strong:`atomic`:  Enable to exclude the atom-like core regions altogether. (H-He: 0, Li-Ne: 1, Na-Ar: 5, K-Kr: 9, Rb-Xe: 18, Cs-Rn: 27, Fr-Og: 43).
   * :strong:`core`: The specified number of lowest-lying occupied orbitals are excluded.
   * :strong:`virtual`: The specified number of highest-lying virtual orbitals are excluded.

:PRINT: This block allows specifying a couple of printing options. When enabled, they provide the following

   * :strong:`ccsd_diagnostics`: Print CCSD T1, D1, D2 diagnostics.
   * :strong:`tamplitudes`: Write T1,T2 amplitude tensor values above a certain threshold to text files.
   * :strong:`rdm`: Write 1- and 2-RDM (reduced density matrix) tensors to disk as plain text files. Specifying 1 and/or 2 to write the desired RDM tensor. Specifying 1 also computes the CCSD natural orbitals and writes them to the SCF files directory. This option only applies when CCSD Lambda is run.

.. note::

   The following capabilities inherit some of the above described options and need to be
   specified within the "CC": { ... } block of the json input file as shown in example.json
   in the inputs folder of the exachem repository.

CCSD perturbative triples (T)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The (T) implementation and additional optimizations on various GPU architectures are described in the following papers.

- Jinsung Kim, Ajay Panyala, Bo Peng, Karol Kowalski, P Sadayappan and Sriram Krishnamoorthy. **Scalable Heterogeneous Execution of a Coupled-Cluster Model with Perturbative Triples.** *International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*, Nov 2020. https://doi.org/10.1109/SC41405.2020.00083

- Abhishek Bagusetty, Ajay Panyala, Gordon Brown, Jack Kirk. **Towards Cross-Platform Portability of Coupled-Cluster Methods with Perturbative Triples using SYCL.** *IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC)*, Nov 2022. https://doi.org/10.1109/P3HPC56579.2022.00013

.. code-block:: json

 "CCSD(T)": {
    "cache_size": 8,
    "skip_ccsd": false,
    "ccsdt_tilesize": 40
 }

:cache_size: ``[default=8]`` Each process (MPI rank) caches the specified number of blocks of the T2 and 2e integral tensors. This increases the overall memory consumption, but reduces the communication time for large calculations. The value should be set to 0 if minimal memory overhead is desired.

:ccsdt_tilesize: ``[default=40]`` tilesize for the MSO dimension of the T1,T2 amplitude and 2e integral tensors. The tensors are re-tiled post CCSD just before the (T) calculation begins.

:skip_ccsd: ``[default=false]`` Mostly used for performance benchmarking for the (T) calculation. When enabled, the cholesky decomposition and CCSD iterations are skipped.


DUCC
~~~~

The double unitary CC formalism (DUCC) is described in the following paper. 

- Nicholas P. Bauman, Eric J. Bylaska, Sriram Krishnamoorthy, Guang Hao Low, Nathan Wiebe, Christopher E. Granade, Martin Roetteler, Matthias Troyer, Karol Kowalski. **Downfolding of many-body Hamiltonians using active-space models: Extension of the sub-system embedding sub-algebras approach to unitary coupled cluster formalisms.** *The Journal of Chemical Physics (JCP)*, July 2019. https://doi.org/10.1063/1.5094643

.. code-block:: json

 "CC": {
   "nactive"  : 0,
   "ducc_lvl" : 2
 }

:nactive: ``[default=0]`` An integer that specifies the size of the active space (or) equivalently the number of active virtual orbitals included in the Hamiltonian H.

:ducc_lvl: ``[default=2]`` An integer that specifies the level of DUCC theory.

   * :strong:`0`: Only computes the bare hamiltonian.
   * :strong:`1`: Computes level 0 plus the Single Commutator and Double Commutator of F.
   * :strong:`2`: Computes level 1 plus the Double Commutator and Triple Commutator of F.

EOMCCSD
~~~~~~~

.. code-block:: json

 "EOMCCSD": {
   "eom_nroots": 0,
   "eom_type": "right",
   "eom_threshold": 1e-6,
   "eom_microiter": 50
 }

:eom_nroots: Specify the number of excited state roots to be determined ``[default=1]``.

:eom_type: Specifies the type of eigenvectors to be computed in the EOMCCSD calculation.

   * :strong:`right (default)`: Compute the right eigenvectors.
   * :strong:`left:` Compute the left eigenvectors.

:eom_threshold: ``[default=threshold]`` Specifies the convergence threshold for the iterative solution of the EOMCCSD equations.

:eom_microiter: ``[default=ccsd_maxiter]`` Number of iterations until the iterative subspace is collapsed into new initial guess vectors. 

.. eom_maxiter option is not provided since it uses the value of ccsd_maxiter


RT-EOMCCSD
~~~~~~~~~~

The RT-EOMCCSD procedure is described in the following paper. 

- Himadri Pathak, Ajay Panyala, Bo Peng, Nicholas P. Bauman, Erdal Mutlu, John J. Rehr, Fernando D. Vila, Karol Kowalski. **Real-Time Equation-of-Motion Coupled-Cluster Cumulant Green’s Function Method: Heterogeneous Parallel Implementation Based on the Tensor Algebra for Many-Body Methods Infrastructure.** *Journal of Chemical Theory and Computation (JCTC)*, April 2023. https://doi.org/10.1021/acs.jctc.3c00045

.. code-block:: json

 "RT-EOMCC": {
   "pcore"  : 0,
   "ntimesteps": 10,
   "rt_microiter": 20,
   "rt_threshold": 1e-6,
   "rt_step_size": 0.025,
   "rt_multiplier": 0.5
 }

:pcore: ``[default=0]`` The occupied orbital with its corresponding index needs to be moved to a virtual orbital while maintaining a hole in the occupied subspace. The SCF eigenvector analysis assists in selecting the appropriate index for this orbital. Note that the value for `pcore` orbitals should be provided starting from 1, rather than 0. The *RT-EOMCCSD* calculation currently requires the **exachem** executable to be run twice. For the first run, task ``cd_2e`` needs to be enabled and ``scf_type`` set to ``restricted`` in the SCF block. In this run, a *Hartree-Fock* calculation is performed, the coefficient matrix and the fock matrix (in MSO basis) are written to disk. The subsequent run skips *Hartree-Fock*, reads these matrices and performs the actual *RT-EOMCCSD* calculation. This run requires ``scf_type`` set to ``unrestricted`` with the appropriate `charge` and `multiplicity` values in the SCF block and task ``rteom_ccsd`` enabled.

:rt_threshold: ``[default=1e-6]`` Specifies the convergence threshold for the time-dependent EOMCCSD calculation.

:rt_microiter: ``[default=20]`` Specifies the number of microiterations performed within each macroiteration.

:ntimesteps: ``[default=10]`` Specifies the number of timesteps used in the time propagation of the wavefunction.

:rt_step_size: ``[default=0.025]`` Specifies the step size used in the time propagation of the wavefunction.

:rt_multiplier: ``[default=0.5]`` Specifies a multiplier factor that scales the step size in the time propagation of the wavefunction.

.. note::

   The same options described here can be used to run an RT-EOM-CC2 calculation using task ``rteom_cc2`` in the input file.
