
<p align="center"><img src="./docs/logos/exachem_logo.png" width="70%"></p>

<p align="center">
<img alt="pnnl logo" src="./docs/logos/pnnl_logo.png" width="200pt" height="100pt"/> &emsp;
<img alt="pnnl logo" src="./docs/logos/doe_logo.png" width="200pt" height="60pt"/>
</p>

<br /><br />

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Documentation Status](https://readthedocs.org/projects/exachem/badge/?version=latest)](https://exachem.readthedocs.io/en/latest/?badge=latest)

## Overview

**ExaChem** is a suite of scalable electronic structure methods to perform ground and excited-state calculations on molecular systems. It is currently being developed synergistically with the [NWChemEx](https://nwchemex.github.io/NWChemEx), [SPEC](https://spec.labworks.org/home), QIS, and [MAPOL](https://github.com/mapol-chem) projects (please see acknowledgements). The methodologies in ExaChem are implemented using the **T**ensor **A**lgebra for **M**any-body **M**ethods ([TAMM](https://github.com/NWChemEx/TAMM)) library. TAMM is a parallel tensor algebra library for performance-portable development of scalable electronic structure methods that can be run on modern exascale computing platforms. ExaChem  currently includes native implementations of: Hartree-Fock (HF), MP2, CC2, CCSD, CCSD(T), CCSD-Lambda, EOM-CCSD, RT-EOM-CCSD, GFCCSD, and double unitary coupled-cluster (DUCC). ExaChem has also been interfaced with the [GauXC library](https://github.com/wavefunction91/GauXC) to enable DFT calculations.

ExaChem and TAMM are actively being developed and maintained at the Pacific Northwest National Laboratory ([PNNL](https://pnnl.gov)) and distributed as open-source under the terms of the Apache License version 2.0.


## Build 

Build instructions are available [here](https://exachem.readthedocs.io/en/latest/install.html)

## ExaChem Citation
#### Please cite the following reference when publishing results obtained with ExaChem. 

Panyala, Ajay; Bauman, Nicholas; Mejia Rodriguez, Daniel; Pathak, Himadri; Peng, Bo; Mutlu, Erdal; Murphy, David; Nandipati, Giridhar; Krishnamoorthy, Sriram; Xantheas, Sotiris; Govind, Niranjan; Kowalski, Karol. **ExaChem: Open Source Exascale Computational Chemistry Software.** https://github.com/ExaChem/exachem [DOI:10.11578/dc.20230628.1](https://doi.org/10.11578/dc.20230628.1)

#### Please cite the following reference in addition if using the ground-state closed-shell CCSD and CCSD(T) capabilities.

Kowalski, Karol, Bair, Raymond, Bauman, Nicholas P., Boschen, Jeffery S., Bylaska, Eric J., Daily, Jeff, de Jong, Wibe A., Dunning, Thom Jr., Govind, Niranjan, Harrison, Robert J., Keçeli, Murat, Keipert, Kristopher, Krishnamoorthy, Sriram, Kumar, Suraj, Mutlu, Erdal, Palmer, Bruce, Panyala, Ajay, Peng, Bo, Richard, Ryan M., Straatsma, T. P., Sushko, Peter, Valeev, Edward F., Valiev, Marat, van Dam, Hubertus J. J., Waldrop, Jonathan M., Williams-Young, David B., Yang, Chao, Zalewski, Marcin and Windus, Theresa L. **From NWChem to NWChemEx: Evolving with the Computational Chemistry Landscape.** Chemical Reviews (2021) [DOI:10.1021/acs.chemrev.0c00998](doi.org/10.1021/acs.chemrev.0c00998).

## Acknowledgements

The TAMM library (core infrastructure and current optimizations), the ground-state formulations of the closed-shell CCSD, and CCSD(T) methods have been supported by the [NWChemEx](https://nwchemex.github.io/NWChemEx) project, funded through the [Exascale Computing Project ECP](https://www.exascaleproject.org) (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. 

The development of additional TAMM infrastructure extensions, optimizations and methodologies (HF, MP2, CC2, CCSD-Lambda, EOM-CCSD, RT-EOM-CCSD, and GFCCSD) are supported by the Center for **S**calable **P**redictive Methods for **E**xcitations and **C**orrelated Phenomena  [(SPEC)](https://spec.labworks.org/home) under FWP 70942.

The double unitary coupled-cluster (DUCC) development is supported under FWP 72689 (Embedding QC into Many-body Frameworks for Strongly Correlated Molecular and Materials Systems) funded by the DOE BES "Materials and Chemical Sciences Research for Quantum Information Science" program.

Ongoing development of many-body methodologies for molecular polaritonic systems is being funded by FWP 79715, Center for **Ma**ny-Body Methods, Spectroscopies, and Dynamics for Molecular **Pol**aritonic Systems (MAPOL).

The SPEC and MAPOL projects are funded as part of the Computational Chemical Sciences (CCS) program by the U.S. Department of Energy (DOE), Office of Science, Office of Basic Energy Sciences (BES), Division of Chemical Sciences, Geosciences and Biosciences at PNNL. PNNL is a multi-program national laboratory operated by Battelle Memorial Institute for the United States Department of Energy under DOE contract number **DE-AC05-76RL01830**.

Acknowledgements for Computing Resources can be found [here](docs/resource_ack.md).
