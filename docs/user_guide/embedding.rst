.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters

.. TODO: Match defaults with code snippets

.. _EMBEDDING:

EMBEDDING options
~~~~~~~~~~~~~~~~~

This section is used to define projector-based quantum embedding parameters.

.. code-block:: json

 "EMBEDDING": {
   "partition": "SPADE",
   "lambda": 1.0e6,
   "use_ksref": false
   "projector": "huzinaga",
   "high_level": ["CCSD"],
   "pao_thresh1": 0.01,
   "pao_thresh2": 0.005,
   "active_atoms": [],
   "active_orbitals": [],
   "nactive_occ_orbitals": "",
   "nactive_vir_orbitals": "",
   "avas_occ_threshold": 0.1,
   "avas_vir_threshold": 0.1,
   "freeze_projected": false,
 }

:lambda: ``[default=1.0e6]`` The level-shift value used to move the eigenvalues of the projected orbitals.

:use_ksref: If **true**, use a Kohn-Sham reference to start post-HF calculations. This includes FCIDUMP: with a DFT low level, **false** writes the file in Hartree-Fock orbitals and **true** writes it in Kohn-Sham orbitals.

:partition: Type of partitioning scheme for the active and inactive subsystems.

    * :strong:`"SPADE"`
    * :strong:`"AVAS"`

:projector: Type of projector used to keep orthogonality between active and inactive subsystems.

    * :strong:`"huzinaga"`
    * :strong:`"level"`

:high_level: A string specifying the high-level calculation. It can be a combination of exchange-correlation functionals, a string identifying the post-HF method, or FCIDUMP for active space integral writeout. FCIDUMP does not compute a high-level energy, so the run prints the ``Embedding energy correction`` instead: add it to an energy computed from the file to get the embedded total energy.

    * :strong:`"HF"`
    * :strong:`"PBE0"`
    * :strong:`"MP2"`
    * :strong:`"CCSD"`
    * :strong:`"CCSD_CANONICAL"`
    * :strong:`"FCIDUMP"`

:pao_thresh1: Norm-based threshold to control the number of PAOs kept.

:pao_thresh2: Linear dependence-based threshold to control the number of PAOs kept.

:active_atoms: A list of atom indices that defines the active region.

:active_orbitals: A dictionary of atom indices and lists of respective shells that define the active region (e.g., {"0": ["s", "p", "d", "f"], "1": ["p"]})

:nactive_occ_orbitals: A list of numbers that define the size of the active occupied space (e.g., [10] for spin-restricted or [10, 5] for spin-unrestricted)

:nactive_vir_orbitals: A list of numbers that define the size of the active virtual space

:avas_occ_threshold: Threshold to control the number of SPADE or AVAS occupied orbitals kept.

:avas_vir_threshold: Threshold to control the number of AVAS virtual orbitals kept.

:freeze_projected: Whether the subsequent post-HF method freeze the projected orbitals (occupied and unoccupied) from the environment.
