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
   "ccsd_maxiter": 100,
 
   "readt": false,
   "writet": false,
   "writet_iter": 5,
 
   "debug": false,
   "profile": false,
   "store_v4": false,

   "freeze": {
      "atomic": false,
      "core": 0,
      "virtual": 0
   },   

   "PRINT": {
      "noprint" : false,
      "tamplitudes": [false,0.05],
      "ccsd_diagnostics" : false,
      "rdm": [1,2]
   },

   "solvers": {
      "solver_type": "diis",
      "newton_krylov": {
         "preconditioner": "jacobi",
         "krylov_dims": 5,
         "eta": 0.1,
         "gamma": 0.9,
         "alpha": 1.5,
         "adaptive_forcing": true,
         "newton_tol": 1e-6,
         "gmres_tol": 1e-6
      }
   }
  }

:threshold: ``[default=1e-6]`` Specifies the convergence threshold of iterative solutions of amplitude equations. The threshold refers to the norm of residual, namely, the deviation from the amplitude equations.

:tilesize: ``[default=40]`` The tilesize for the MO or MSO space. An integer value that is automatically set unless specified explicitly by the user. It is recommended to let the CC module determine this value automatically.

   .. warning::

      The automatically determined tilesize may vary depending on node count and the number of processes per node. When restarting a calculation with different node counts, the user is expected to explicitly set the tilesize to match the value used in the initial run.

:lshift: ``[default=0]`` The level shift option that increases small orbital energy differences used in calculating the updates for cluster amplitudes. When calculating ground states with multi-configurational character or if convergence is slow, one may need to specify a typical values for lshift between 0.3 and 0.5.

:ndiis: ``[default=5]`` The number iterations in which a DIIS extrapolation is performed to accelerate the convergence of excitation amplitudes. The default value is 5, which means in every five iteration, one DIIS extrapolation is performed (and in the rest of the iterations, Jacobi rotation is used). When zero or negative value is specified, the DIIS is turned off. It is not recommended to perform DIIS every iteration, whereas setting a large value for this parameter necessitates a large memory space to keep the excitation amplitudes of previous iterations.

:ccsd_maxiter: ``[default=100]`` The maximum number of iterations performed during the iterative solutions of amplitude equations.

:writet: ``[default=false]`` Writes the T1,T2 amplitude tensors and the 2e integral tensor to disk to be used later for restarting a CC calculation. Currently, the cholesky decomposition module uses this option as well to write the 2e integral tensor to disk. Enabling this option implies restart. 

:readt: ``[default=false]`` Reads the T1,T2 amplitude tensors and the 2e integral tensor from disk for restarting a CC calculation. Enabling this option implies restart. Does not need to be explicitly specified if ``writet=true``.

:writet_iter: ``[default=ndiis]`` This option requires ``writet=true``. An integer that determines the frequency of writing tensors to disk for restart purposes. The tensors are written to disk after every *writet_iter* iterations. 

:debug: ``[default=false]`` enable diagnostic printing for CC calculations.

:profile: ``[default=false]`` When enabled, writes a csv file containing the performance data for every tensor contraction. Useful for profiling contractions. Also applies to the sub-sections (*EOMCCSD*, *GFCCSD*, etc).

:store_v4: ``[default=false]`` By default the :math:`V^4` (four-virtual-index) intermediate is not stored: its blocks are recomputed from the Cholesky vectors each time the doubles residual reads them, which removes the largest tensor from the memory requirement at the cost of extra computation. Enable to compute the intermediate once per iteration and keep it in memory instead, which is faster when it fits. Its size is then included in the reported memory requirement. This option currently applies only to Cholesky-based CCSD; it may be extended in the future to other methods where a similar :math:`V^4` intermediate occurs.

:freeze: This block allows specifying freezing options. Some of the lowest-lying core orbitals and/or some of the highest-lying virtual orbitals may be excluded using this block. No orbitals are frozen by default.

   * :strong:`atomic`:  Enable to exclude the atom-like core regions altogether. (H-He: 0, Li-Ne: 1, Na-Ar: 5, K-Kr: 9, Rb-Xe: 18, Cs-Rn: 27, Fr-Og: 43).
   * :strong:`core`: The specified number of lowest-lying occupied orbitals are excluded.
   * :strong:`virtual`: The specified number of highest-lying virtual orbitals are excluded.

:PRINT: This block allows specifying a couple of printing options. When enabled, they provide the following

   * :strong:`noprint`: Suppress all printing output (Note: not implemented for all CC modules yet).
   * :strong:`ccsd_diagnostics`: Print CCSD T1, D1, D2 diagnostics.
   * :strong:`tamplitudes`: Write T1,T2 amplitude tensor values above a certain threshold to text files.
   * :strong:`rdm`: Write 1- and 2-RDM (reduced density matrix) tensors to disk as plain text files. Specifying 1 and/or 2 to write the desired RDM tensor. Specifying 1 also computes the CCSD natural orbitals and writes them to the SCF files directory. This option only applies when CCSD Lambda is run.

:solvers: This block selects and configures the iterative solver used to converge the CC amplitude equations.

   * :strong:`solver_type`: ``[default=diis]`` Selects the amplitude solver. One of:

     * :strong:`diis`: The standard DIIS-accelerated Jacobi iteration. This is the default and long-standing CC solver.
     * :strong:`newton_krylov`: A Jacobian-free Newton-Krylov solver that solves the nonlinear residual equations :math:`F(x)=0` with an optional inner GMRES (Arnoldi) linear solve at each Newton step. Configured by the ``newton_krylov`` block below.

       Supported only for standalone ``TASK.ccsd`` and ``TASK.ccsd_canonical``.

:newton_krylov: Options for the Newton-Krylov solver. These apply only when ``solver_type`` is set to ``newton_krylov``.

   * :strong:`preconditioner`: ``[default=jacobi]`` Applies the diagonal Fock preconditioner through DIIS's existing ``jacobi`` or ``jacobi_cs`` routines. ``identity`` applies no preconditioning.
   * :strong:`krylov_dims`: ``[default=5]`` The maximum number of Arnoldi steps for the inner Newton GMRES solve. Set to ``0`` to apply the preconditioner correction directly, :math:`x_{k+1}=x_k-M^{-1}F(x_k)`, without Jacobian-vector products. With the default preconditioner, this gives Jacobi iteration.
   * :strong:`eta`: ``[default=0.1]`` The initial Eisenstat-Walker forcing term, which controls how tightly the inner GMRES solve is converged in the early Newton iterations to avoid over-solving.
   * :strong:`gamma`: ``[default=0.9]`` Coefficient of the Eisenstat-Walker forcing term update :math:`\eta = \gamma\,(\lVert F_k\rVert / \lVert F_{k-1}\rVert)^{\alpha}`.
   * :strong:`alpha`: ``[default=1.5]`` Exponent of the Eisenstat-Walker forcing term update (see ``gamma``).
   * :strong:`adaptive_forcing`: ``[default=true]`` When enabled, uses the Eisenstat-Walker adaptive forcing term to stop the inner GMRES solve early (when the residual falls below ``eta`` times the right-hand-side norm), avoiding over-solving the linearized system in the early Newton iterations.
   * :strong:`newton_tol`: ``[default=threshold]`` Convergence threshold for the outer Newton iterations; the solver stops when the residual norm :math:`\lVert F(x)\rVert` falls below this value. If unspecified, it follows the CCSD ``threshold``.
   * :strong:`gmres_tol`: ``[default=threshold]`` Convergence threshold for the inner GMRES solve. If unspecified, it follows the CCSD ``threshold``.

Newton-Krylov Solvers
---------------------

The Newton-Krylov solvers are useful for challenging coupled-cluster calculations, including Hubbard models and strongly correlated systems, where the standard DIIS-accelerated  
Jacobi iteration may converge slowly or fail to converge. Jacobi is the default Newton-Krylov preconditioner for the currently supported canonical gauge. Noncanonical-gauge selection is not implemented; the full Fock-block Sylvester inverse is retained internally for future support.

.. note::

   These solvers are currently available for CCSD only.

:strong:`Choosing the preconditioner and Krylov dimension`

   * Use ``preconditioner: jacobi`` for diagonal Fock preconditioning, or ``preconditioner: identity`` to evaluate the effect of removing preconditioning.
   * Positive ``krylov_dims`` controls the maximum Krylov subspace dimension. Increase it if GMRES stagnates or frequently reaches the dimension limit; decrease it to reduce memory usage and orthogonalization costs.
   * Set ``krylov_dims: 0`` for direct preconditioner updates. The inner tolerance and forcing parameters have no effect in this mode.

:strong:`Tuning the forcing parameters`

   The parameters ``eta``, ``gamma``, ``alpha``, and ``adaptive_forcing`` control the accuracy of the inner GMRES solve.

   * A larger ``eta`` or ``gamma`` generally allows looser inner solves, reducing computational cost but potentially slowing Newton convergence. If inner solves are unnecessarily expensive, increase ``eta`` or ``gamma``; if they are too loose, decrease ``eta`` or ``gamma``.
   * A larger ``alpha`` makes the forcing term decrease more rapidly as the nonlinear residual decreases, resulting in tighter inner solves.
   * With ``adaptive_forcing=false``, the inner GMRES solve uses ``gmres_tol`` directly.
   * Typical range: :math:`1 < \alpha \le 2` and :math:`0 < \gamma < 1`.

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

:ccsdt_tilesize: ``[default=40]`` Applies only to the (T) calculation. This specifies the tilesize for the MSO dimension of the T1,T2 amplitude and 2e integral tensors. The tensors are re-tiled post CCSD just before the (T) calculation begins.

:skip_ccsd: ``[default=false]`` Mostly used for performance benchmarking for the (T) calculation. When enabled, the cholesky decomposition and CCSD iterations are skipped.


DUCC
~~~~

The double unitary coupled-cluster formalism (DUCC) is described in the following paper. 

- Nicholas P. Bauman, Eric J. Bylaska, Sriram Krishnamoorthy, Guang Hao Low, Nathan Wiebe, Christopher E. Granade, Martin Roetteler, Matthias Troyer, Karol Kowalski. **Downfolding of many-body Hamiltonians using active-space models: Extension of the sub-system embedding sub-algebras approach to unitary coupled cluster formalisms.** *The Journal of Chemical Physics (JCP)*, July 2019. https://doi.org/10.1063/1.5094643

.. code-block:: json

 "CC": {
   "nactive_oa"  : 0,
   "nactive_ob"  : 0,
   "nactive_va"  : 0,
   "nactive_vb"  : 0,
   "ducc_lvl" : 2,
   "qflow_cycles" : 100,
   "qflow_threshold" : 1e-3,
   "qflow_nproc_pc" : 0
 }

:nactive_oa: ``[default=0]`` An integer that specifies the number of active occupied alpha orbitals included in the Hamiltonian H.
:nactive_ob: ``[default=0]`` An integer that specifies the number of active occupied beta orbitals included in the Hamiltonian H.
:nactive_va: ``[default=0]`` An integer that specifies the number of active virtual alpha orbitals included in the Hamiltonian H.
:nactive_vb: ``[default=0]`` An integer that specifies the number of active virtual beta orbitals included in the Hamiltonian H.

:ducc_lvl: ``[default=2]`` An integer that specifies the level of DUCC theory.

   * :strong:`0`: Only computes the bare Hamiltonian.
   * :strong:`1`: Computes level 0 plus the single commutator of F and V and double commutator of F.
   * :strong:`2`: Computes level 1 plus the double commutator of V and triple commutator of F.

:qflow_cycles: ``[default=100]`` An integer that specifies the number of QFlow cycles. A single QFlow cycle computes all of the effective Hamiltonians of a given size given by nactive_oa, nactive_ob, nactive_va, and nactive_vb and then call solvers to update the global pool of cluster amplitudes.  
:qflow_threshold: ``[default=1e-3]`` Specifies the convergence threshold for the energy of the leading active space of the QFlow cycles. The QFlow cycles are converged when the difference between the current and previous energy is less than qflow_threshold.
:qflow_nproc_pc: ``[default=0]`` Number of mpi processes to use to process a single combination.

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

- Himadri Pathak, Ajay Panyala, Bo Peng, Nicholas P. Bauman, Erdal Mutlu, John J. Rehr, Fernando D. Vila, Karol Kowalski. **Real-Time Equation-of-Motion Coupled-Cluster Cumulant Green's Function Method: Heterogeneous Parallel Implementation Based on the Tensor Algebra for Many-Body Methods Infrastructure.** *Journal of Chemical Theory and Computation (JCTC)*, April 2023. https://doi.org/10.1021/acs.jctc.3c00045

.. code-block:: json

 "RT-EOMCC": {
   "pcore"  : 0,
   "ntimesteps": 10,
   "rt_microiter": 20,
   "rt_threshold": 1e-6,
   "rt_step_size": 0.025,
   "rt_secant_x": 0.1,
   "rt_step_dec": 0.9,
   "rt_step_inc": 1.1,
   "rt_step_max": 0.05
 }

:pcore: ``[default=0]`` The occupied orbital with its corresponding index needs to be moved to a virtual orbital while maintaining a hole in the occupied subspace. The SCF eigenvector analysis assists in selecting the appropriate index for this orbital. Note that the value for `pcore` orbitals should be provided starting from 1, rather than 0. A value greater than 0 is required for an *RT-EOMCCSD* calculation.

The input file for an *RT-EOMCCSD* calculation describes the closed-shell **reference** system: the SCF block must have ``scf_type`` set to ``restricted`` with the charge and multiplicity of the neutral system, and task ``rteom_ccsd`` enabled. The *RT-EOMCCSD* calculation itself is performed on the **core-hole** state, obtained by removing one beta electron from the orbital selected by `pcore` while keeping the reference orbitals. When task ``rteom_ccsd`` is specified, **exachem** runs both steps automatically: a *Hartree-Fock* calculation on the reference system, after which the coefficient matrix and the fock matrix (in MSO basis) are written to disk; then the SCF options are rewritten for the core-hole system (``charge`` + 1, ``multiplicity`` 2, ``scf_type`` ``unrestricted``, ``noscf``) and the reference matrices are read back to set up and run the *RT-EOMCCSD* calculation.

:rt_threshold: ``[default=1e-6]`` Specifies the convergence threshold for the time-dependent EOMCCSD calculation.

:rt_microiter: ``[default=20]`` Specifies the maximum number of micro-iterations performed within each time step.

:ntimesteps: ``[default=10]`` Specifies the number of timesteps used in the time propagation of the wavefunction.

:rt_step_size: ``[default=0.025]`` Specifies the step size used in the time propagation of the wavefunction. For *RT-EOMCCSD* this is the initial step size, which is adjusted after every time step (see ``rt_step_dec``, ``rt_step_inc`` and ``rt_step_max``).

*RT-EOMCCSD* propagates the amplitudes with the trapezoidal rule, solving the implicit equation at each time step by secant micro-iterations. The step size is adapted between time steps based on how many micro-iterations the previous step needed. The following options control this scheme.

:rt_secant_x: ``[default=0.1]`` Scaling parameter for the two initial secant guesses, which are built from the previous amplitudes advanced by ``(1 - rt_secant_x)`` and ``(1 + rt_secant_x)`` times the current step size.

:rt_step_dec: ``[default=0.9]`` Factor by which the step size is reduced for the next time step when the current time step needed 7 or more micro-iterations (or did not converge within ``rt_microiter``).

:rt_step_inc: ``[default=1.1]`` Factor by which the step size is increased for the next time step when the current time step converged in 4 or fewer micro-iterations.

:rt_step_max: ``[default=0.05]`` Maximum step size. The adapted step size is capped at this value.

Analysis scripts
----------------

An *RT-EOMCCSD* calculation writes the time derivative of the cumulant, :math:`dC/dt`, at every time step to ``<prefix>.dcdt`` in the ``rteom`` subdirectory of the run's files directory. Each line holds the time step number, the time (au) and the real and imaginary parts of :math:`dC/dt`. The scripts in ``exachem/cc/rteom/scripts`` post-process this file. They need ``numpy``, ``plotly`` and ``kaleido`` (``pip3 install numpy plotly kaleido``).

``plot_dcdt.py`` plots the real and imaginary parts of :math:`dC/dt` against time and writes ``<prefix>.dcdt.re_im_vs_t.png``:

.. code-block:: bash

   python3 plot_dcdt.py <prefix>.dcdt

``rteom_analysis.py`` builds the cumulant and the retarded Green's function from :math:`dC/dt`, Fourier transforms it to get the spectral function, and fits the quasiparticle (QP) peak with a Lorentzian to report the QP binding energy (eV) and strength. With ``use_pade`` enabled, it also computes the spectral function from a linear-prediction Padé fit of the Green's function, which extrapolates the time signal to infinite time.

.. code-block:: bash

   python3 rteom_analysis.py <prefix>.dcdt params.json

Output files, all named after the dcdt file and written to the current directory:

- ``.cumulant_gf.txt``: time, cumulant (real, imag) and Green's function (real, imag).
- ``.ft_greens_function.txt`` and ``.ft_gf.png``: energy (eV) and spectral function from the Fourier transform.
- ``.pade_greens_function.txt``, ``.pade_gf.png`` and ``.pade_uniform_gf_t.txt``: the same from the Padé fit, and the Green's function on the uniform time grid used for the fit (only with ``use_pade``).
- ``.ft_beta.txt`` and ``.dc2dt2.txt``: :math:`\beta(\omega)` and :math:`-i\,d^2C/dt^2`.

If a restarted calculation appended time steps that were already in the file, the last entry for each time step is kept and the cleaned file is written to ``<prefix>.dcdt.new``.

``params.json`` (a template is provided next to the scripts) holds the analysis parameters:

.. code-block:: json

 {
   "delta_real": 0.01,
   "E_Ext_l": 5.0,
   "E_Ext_h": 5.0,
   "dE_local": 0.001,
   "E_c": 0.0,
   "E_Corr_0": 0.0,

   "use_pade": true,
   "pade_order": 800,
   "pade_reg": 1.0e-8,
   "pade_eta": 0.0,
   "pade_nfit": 0,
   "pade_dt": 0.05,
   "pade_root_radius": 1.000001
 }

:delta_real: Broadening parameter (au), added as an imaginary part to the energy grid.

:E_Ext_l, E_Ext_h: Energy extensions (au) below and above the reference energy ``E_c + E_Corr_0``. The grid runs from ``-E_Ext_l`` to ``E_Ext_h`` relative to it.

:dE_local: Energy-grid spacing (au).

:E_c: Core orbital energy (au), i.e. the orbital energy of the ``pcore`` orbital in the closed-shell reference SCF.

:E_Corr_0: *N*-electron ground-state correlation energy (au), from a separate CCSD calculation on the closed-shell reference.

:use_pade: ``[default=false]`` Also compute the spectral function from a linear-prediction Padé fit.

:pade_order: ``[default=200]`` Number of linear-prediction coefficients (poles) in the fit. It is reduced automatically if it is too large for the number of fitted points.

:pade_reg: ``[default=1e-8]`` Tikhonov regularization of the linear-prediction least-squares fit.

:pade_eta: ``[default=0.0]`` Optional extra exponential damping (au) applied to the signal before fitting.

:pade_nfit: ``[default=0]`` Number of points of the uniform grid used in the fit. 0 uses all of them.

:pade_dt: ``[default=0.05]`` Time step (au) of the uniform grid the Green's function is resampled on for the fit. Set it to the ``rt_step_max`` used in the calculation.

:pade_root_radius: ``[default=1.000001]`` Only poles with modulus up to this value (non-growing terms) are kept.

:trim_last_points: ``[default=0]`` Number of time steps dropped from the end of the dcdt file before the analysis.
