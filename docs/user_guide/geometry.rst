==================================
Geometry Analysis and Optimization
==================================


Geometry Analysis
=================

The geometry analysis module first prints bond lengths, where a bond is defined as two atoms within 1.3 times the sum of their covalent radii. If an atom is not bonded to any other atom, the threshold for a bond is increased until a bond is found. These extended bonds are also printed. This process will not necessarily connect fragments. Bond angles are calculated similarly, where for an atomic configuration ijk, bonds ij and jk must exist. For dihedral angles, for atom configuration ijkl, bonds ij, jk, and kl must exist. The dihedral angle is not printed if it is equal to exactly 0.0 or 180.0 degrees. The center of mass, moments of inertia, and the z-matrix representation of the geometry are also printed.

Geometry Optimization
=====================

We developed a C++ implementation of `PyBerny <https://github.com/jhrmnn/pyberny>`_, which is effectively identical to the original Python implementation. A slight difference is that, due to a discrepancy in how division by zero is handled, certain internal variables are set to zero if their absolute value exceeds 1e25. ExaChem does not support crystal geometries, and therefore that portion of the PyBerny code is not implemented.

ExaChem also provides an interface to `geomeTRIC <https://geometric.readthedocs.io>`_. This interface calls the geomeTRIC Python code directly from within the C++ code, with ExaChem supplying the energies and gradients at each step. It is the default optimizer when Python bindings are enabled. The native C++ PyBerny implementation can also be selected instead via the **TASK** ``operation`` option (see :ref:`TASK`).

Both optimizers run for a maximum of 300 iterations by default.
