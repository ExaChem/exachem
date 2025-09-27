
CCSD
====

This section presents a range of CCSD performance benchmarks performed with ExaChem across three DOE supercomputing platforms: `Frontier (OLCF) <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html>`_, `Aurora (ALCF) <https://docs.alcf.anl.gov/aurora/>`_, and `Perlmutter (NERSC) <https://docs.nersc.gov/systems/perlmutter/architecture/>`_. 

The benchmarks show the performance using the CCSD per-iteration timing on all three supercomputing platforms for a variety of molecular configurations. All benchmarks shown are closed-shell calculations and use an explicitly correlated all-electron treatment with no frozen core. The benchmarks are arranged in increasing order of basis functions. On each machine, the number of MPI processes per node is set to the number of GPU tiles per node - 4 for Perlmutter, 8 for Frontier, and 12 for Aurora. All molecular systems benchmarked demonstrate strong scaling with increasing node counts, though the optimal node count varies with the system and molecular configuration. For certain molecular systems, we also include data points beyond the strong-scaling limit, even where performance degrades, to highlight the scalability limits of the benchmarked systems. Please note that these timings were obtained in Summer 2025 and may evolve in the future.

+--------------------------------------------------------------------------------+-------------+----------+----------------+
|System                                                                          |basis        |#electrons|#basis functions|
+================================================================================+=============+==========+================+
| :ref:`ubiquitin-dgrtl [O8N8C22H41] <fig-combined_ccsd_Oa146_Va278>`            | 6-31g       | 292      | 424            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Co1N6C30H22] <fig-combined_ccsd_Oa134_Va523>`    | cc-pvdz     | 268      | 657            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Mn1O2N4C22H28] <fig-combined_ccsd_Oa116_Va575>`  | cc-pvdz     | 228      | 691            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`nitrobenzene [Br1O2N1C6H5] <fig-combined_ccsd_Oa49_Va663>`               | cc-pvqz     | 98       | 712            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`ubiquitin-dgrtl [O8N8C22H41] <fig-combined_ccsd_Oa146_Va591>`            | cc-pvdz     | 292      | 737            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`radon3 [Fe1N6C12H30] <fig-combined_ccsd_Oa85_Va698>`                     | def2-tzvp   | 168      | 783            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Fe1O2N4C16H28] <fig-combined_ccsd_Oa99_Va718>`   | aug-cc-pvdz | 194      | 817            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`porphyrin [N4C20H14] <fig-combined_ccsd_Oa81_Va835>`                     | cc-pvtz     | 162      | 916            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Mn1O2N4C22H28] <fig-combined_ccsd_Oa116_Va840>`  | aug-cc-pvdz | 228      | 956            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`mobius [C60H30] <fig-combined_ccsd_Oa196_Va764>`                         | 6-31g*      | 390      | 960            |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Co1N6C30H22] <fig-combined_ccsd_Oa134_Va951>`    | aug-cc-pvdz | 268      | 1085           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Fe1O2N4C16H28] <fig-combined_ccsd_Oa99_Va1021>`  | cc-pvtz     | 194      | 1120           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`siosi5 [Si26O37H30] <fig-combined_ccsd_Oa345_Va791>`                     | cc-pvdz     | 690      | 1136           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`guanine-cytosine-3bp [O6N24C27H30] <fig-combined_ccsd_Oa204_Va969>`      | 6-31+g**    | 408      | 1173           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`water53 [O53H106] <fig-combined_ccsd_Oa235_Va1007>`                      | cc-pvdz     | 470      | 1242           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`ubiquitin-dgrtl [O8N8C22H41] <fig-combined_ccsd_Oa146_Va1096>`           | aug-cc-pvdz | 292      | 1242           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`C60 [C60] <fig-combined_ccsd_Oa180_Va1070>`                              | aug-cc-pvdz | 360      | 1250           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Mn1O2N4C22H28] <fig-combined_ccsd_Oa116_Va1184>` | cc-pvtz     | 228      | 1300           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`spin-crossover complex [Co1N6C30H22] <fig-combined_ccsd_Oa134_Va1200>`   | cc-pvtz     | 268      | 1334           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+
| :ref:`ubiquitin-dgrtl [O8N8C22H41] <fig-combined_ccsd_Oa146_Va1568>`           | cc-pvtz     | 292      | 1714           |
+--------------------------------------------------------------------------------+-------------+----------+----------------+

    .. _fig-combined_ccsd_Oa146_Va278:

    .. figure:: ccsd/combined_ccsd_Oa146_Va278.png
       :width: 100%
       :align: center

       **ubiquitin-dgrtl [O8N8C22H41] (6-31g, 292 electrons, 424 basis functions)**
   


    .. _fig-combined_ccsd_Oa134_Va523:

    .. figure:: ccsd/combined_ccsd_Oa134_Va523.png
       :width: 100%
       :align: center

       **spin-crossover complex [Co1N6C30H22] (cc-pvdz, 268 electrons, 657 basis functions)**
   


    .. _fig-combined_ccsd_Oa116_Va575:

    .. figure:: ccsd/combined_ccsd_Oa116_Va575.png
       :width: 100%
       :align: center

       **spin-crossover complex [Mn1O2N4C22H28] (cc-pvdz, 228 electrons, 691 basis functions)**
   


    .. _fig-combined_ccsd_Oa49_Va663:

    .. figure:: ccsd/combined_ccsd_Oa49_Va663.png
       :width: 100%
       :align: center

       **nitrobenzene [Br1O2N1C6H5] (cc-pvqz, 98 electrons, 712 basis functions)**
   


    .. _fig-combined_ccsd_Oa146_Va591:

    .. figure:: ccsd/combined_ccsd_Oa146_Va591.png
       :width: 100%
       :align: center

       **ubiquitin-dgrtl [O8N8C22H41] (cc-pvdz, 292 electrons, 737 basis functions)**
   


    .. _fig-combined_ccsd_Oa85_Va698:

    .. figure:: ccsd/combined_ccsd_Oa85_Va698.png
       :width: 100%
       :align: center

       **radon3 [Fe1N6C12H30] (def2-tzvp, 168 electrons, 783 basis functions)**
   


    .. _fig-combined_ccsd_Oa99_Va718:

    .. figure:: ccsd/combined_ccsd_Oa99_Va718.png
       :width: 100%
       :align: center

       **spin-crossover complex [Fe1O2N4C16H28] (aug-cc-pvdz, 194 electrons, 817 basis functions)**
   


    .. _fig-combined_ccsd_Oa81_Va835:

    .. figure:: ccsd/combined_ccsd_Oa81_Va835.png
       :width: 100%
       :align: center

       **porphyrin [N4C20H14] (cc-pvtz, 162 electrons, 916 basis functions)**
   


    .. _fig-combined_ccsd_Oa116_Va840:

    .. figure:: ccsd/combined_ccsd_Oa116_Va840.png
       :width: 100%
       :align: center

       **spin-crossover complex [Mn1O2N4C22H28] (aug-cc-pvdz, 228 electrons, 956 basis functions)**
   


    .. _fig-combined_ccsd_Oa196_Va764:

    .. figure:: ccsd/combined_ccsd_Oa196_Va764.png
       :width: 100%
       :align: center

       **mobius [C60H30] (6-31g\*, 390 electrons, 960 basis functions)**
   


    .. _fig-combined_ccsd_Oa134_Va951:

    .. figure:: ccsd/combined_ccsd_Oa134_Va951.png
       :width: 100%
       :align: center

       **spin-crossover complex [Co1N6C30H22] (aug-cc-pvdz, 268 electrons, 1085 basis functions)**
   


    .. _fig-combined_ccsd_Oa99_Va1021:

    .. figure:: ccsd/combined_ccsd_Oa99_Va1021.png
       :width: 100%
       :align: center

       **spin-crossover complex [Fe1O2N4C16H28] (cc-pvtz, 194 electrons, 1120 basis functions)**
   


    .. _fig-combined_ccsd_Oa345_Va791:

    .. figure:: ccsd/combined_ccsd_Oa345_Va791.png
       :width: 100%
       :align: center

       **siosi5 [Si26O37H30] (cc-pvdz, 690 electrons, 1136 basis functions)**
   


    .. _fig-combined_ccsd_Oa204_Va969:

    .. figure:: ccsd/combined_ccsd_Oa204_Va969.png
       :width: 100%
       :align: center

       **guanine-cytosine-3bp [O6N24C27H30] (6-31+g\*\*, 408 electrons, 1173 basis functions)**
   


    .. _fig-combined_ccsd_Oa235_Va1007:

    .. figure:: ccsd/combined_ccsd_Oa235_Va1007.png
       :width: 100%
       :align: center

       **water53 [O53H106] (cc-pvdz, 470 electrons, 1242 basis functions)**
   


    .. _fig-combined_ccsd_Oa146_Va1096:

    .. figure:: ccsd/combined_ccsd_Oa146_Va1096.png
       :width: 100%
       :align: center

       **ubiquitin-dgrtl [O8N8C22H41] (aug-cc-pvdz, 292 electrons, 1242 basis functions)**
   


    .. _fig-combined_ccsd_Oa180_Va1070:

    .. figure:: ccsd/combined_ccsd_Oa180_Va1070.png
       :width: 100%
       :align: center

       **C60 [C60] (aug-cc-pvdz, 360 electrons, 1250 basis functions)**
   


    .. _fig-combined_ccsd_Oa116_Va1184:

    .. figure:: ccsd/combined_ccsd_Oa116_Va1184.png
       :width: 100%
       :align: center

       **spin-crossover complex [Mn1O2N4C22H28] (cc-pvtz, 228 electrons, 1300 basis functions)**
   


    .. _fig-combined_ccsd_Oa134_Va1200:

    .. figure:: ccsd/combined_ccsd_Oa134_Va1200.png
       :width: 100%
       :align: center

       **spin-crossover complex [Co1N6C30H22] (cc-pvtz, 268 electrons, 1334 basis functions)**
   


    .. _fig-combined_ccsd_Oa146_Va1568:

    .. figure:: ccsd/combined_ccsd_Oa146_Va1568.png
       :width: 100%
       :align: center

       **ubiquitin-dgrtl [O8N8C22H41] (cc-pvtz, 292 electrons, 1714 basis functions)**
   

