
Running the various methods
============================

* SCF: Run SCF using ``HartreeFock`` executable

* Cholesky Decomposition: Set the SCF options ``restart,noscf`` to ``true``, and ``writet`` to ``true`` and run Cholesky Decomposition using ``CholeskyDecomp`` executable.

* CCSD: Run CCSD_CS until convergence (or) repeat Step 3 until CCSD converges.

* Generate data for CCSD(T): Set ``writev:true``. Also add the CCSD(T) section and run ``CCSD_CS`` executable. 
  This will compute and dump the full T1,T2,V2 tensors first using CCSD tilesize, read
  them back, re-tile them accordingly for (T) and dump these `re-tiled` tensors.

* CCSD(T): Run the ``CCSD_T_CS`` executable. This will read the `re-tiled` full T1,T2,V2 and proceed to the triples calculation.
