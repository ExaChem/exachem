.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters


Quantum Electrodynamics (QED)
=============================

| :ref:`QED options <QED>`

.. _QED:

Setting the QED options in the SCF block of an input file with ExaChem enables calculations that incorporate quantum electrodynamics (QED) effects into the method. This includes mean-field methods such as QED Hartree-Fock (QED-HF) and QED Density Functional Theory (QED-DFT), as well as post-Hartree-Fock methods like QED Coupled Cluster Singles and Doubles (QED-CCSD). The QED-CCSD method can be used for both closed-shell and open-shell systems. The QED options are detailed below:

.. code-block:: json
  
  "SCF": {    
    "qed_omegas" : [0.10],
    "qed_lambdas": [0.05],
    "qed_volumes": [],
    "qed_polvecs": [[0.0, 0.0, 1.0]]
  }  

.. note:: Currently only a single-cavity mode is supported, with up to two photon excitations in QED-CCSD.


:qed_omegas: A list ``[default: []]``. Specifies the cavity mode frequencies (in atomic units) for each cavity mode. If not provided, the default is an empty list, indicating no cavity modes.

:qed_lambdas: A list ``[default: []]``. Specifies the coupling strengths (in atomic units) for each cavity mode. List dimension must match that of ``qed_omegas``.

:qed_volumes: A list ``[default: []]``. Instead of specifying the coupling strengths directly, one can provide the effective mode volumes (in atomic units) for each cavity mode. The coupling strengths will be computed internally based on the provided mode volumes and ``qed_omegas``. List dimension must match that of ``qed_omegas``.

:qed_polvecs: A list of lists ``[default: []]``. Specifies the polarization vectors for each cavity mode. Each polarization vector should be a list of three components (x, y, z). List dimension must match that of ``qed_omegas``.

Citing this work
~~~~~~~~~~~~~~~~
If you are referencing the QED-CCSD implementation in a publication, please cite the following papers:

-  Nicholas P Bauman, Himadri Pathak, Marcus D Liebenthal, Ajay Panyala, Daniel Mejia-Rodriguez, Niranjan Govind, Karol Kowalski,
   **Quantum electrodynamics coupled-cluster at scale: High-performance implementation for complex systems**, *Journal of Chemical Theory and Computation* (Dec 2025)
   `DOI:10.1021/acs.jctc.5c01599 <https://doi.org/10.1021/acs.jctc.5c01599>`__.
