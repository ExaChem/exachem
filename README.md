
<p align="center"><img src="./docs/logos/exachem_logo.png" width="70%"></p>

<p align="center">
<img alt="pnnl logo" src="./docs/logos/pnnl_logo.png" width="200pt" height="100pt"/> &emsp;
<img alt="pnnl logo" src="./docs/logos/doe_logo.png" width="200pt" height="60pt"/>
</p>

<br /><br />

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Documentation Status](https://readthedocs.org/projects/exachem/badge/?version=latest)](https://exachem.readthedocs.io/en/latest/?badge=latest)
[![CLA assistant](https://cla-assistant.io/readme/badge/ExaChem/exachem)](https://cla-assistant.io/ExaChem/exachem)

## Overview

**ExaChem** is a suite of scalable electronic structure methods to perform ground and excited-state calculations on molecular systems. These methodologies are implemented using the **T**ensor **A**lgebra for **M**any-body **M**ethods ([TAMM](https://github.com/NWChemEx-Project/TAMM)) library. TAMM is a parallel tensor algebra library for performance-portable development of scalable electronic structure methods that can be run on modern exascale computing platforms. The ExaChem computational chemistry suite currently includes implementations of: Hartree-Fock (HF), MP2, CCSD, CCSD(T), CCSD-Lambda, EOM-CCSD, RT-EOM-CCSD, and GFCCSD. Additional capabilities are available via interfaces to other open-source libraries. The ExaChem suite and TAMM library are actively being developed and maintained at the Pacific Northwest National Laboratory ([PNNL](https://pnnl.gov)) and distributed as open-source under the terms of the Apache License version 2.0.


## Build 

Build instructions are available [here](docs/install.md)

## ExaChem Citation
#### Please cite the following reference when publishing results obtained with ExaChem. 

Panyala, Ajay; Govind, Niranjan; Kowalski, Karol; Bauman, Nicholas; Peng, Bo; Pathak, Himadri; Mutlu, Erda; Mejia Rodriguez, Daniel; Xantheas, Sotiris. **ExaChem: Open Source Exascale Computational Chemistry Software.** https://github.com/ExaChem/exachem  (June 2023) [DOI:10.11578/dc.20230628.1](https://doi.org/10.11578/dc.20230628.1)

#### Please cite the following reference in addition if using the ground-state closed-shell CCSD and CCSD(T) capabilities.

Kowalski, Karol, Bair, Raymond, Bauman, Nicholas P., Boschen, Jeffery S., Bylaska, Eric J., Daily, Jeff, de Jong, Wibe A., Dunning, Thom Jr., Govind, Niranjan, Harrison, Robert J., Keçeli, Murat, Keipert, Kristopher, Krishnamoorthy, Sriram, Kumar, Suraj, Mutlu, Erdal, Palmer, Bruce, Panyala, Ajay, Peng, Bo, Richard, Ryan M., Straatsma, T. P., Sushko, Peter, Valeev, Edward F., Valiev, Marat, van Dam, Hubertus J. J., Waldrop, Jonathan M., Williams-Young, David B., Yang, Chao, Zalewski, Marcin and Windus, Theresa L. **From NWChem to NWChemEx: Evolving with the Computational Chemistry Landscape.** Chemical Reviews (2021) [DOI:10.1021/acs.chemrev.0c00998](doi.org/10.1021/acs.chemrev.0c00998).

## Acknowledgements

The TAMM library (core infrastructure and current optimizations), the ground-state formulations of the closed-shell CCSD, and CCSD(T) methods was supported by the NWChemEx project, funded through the Exascale Computing Project (ECP) (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. 

The development of additional TAMM capabilities and methodologies (CCSD-Lambda, EOM-CCSD, RT-EOM-CCSD, and GFCCSD) was supported by the Center for Scalable Predictive Methods for Excitations and Correlated Phenomena  (SPEC) under FWP 70942, which is funded as part of the Computational Chemical Sciences (CCS) program by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences and Biosciences at PNNL. PNNL is a multi-program national laboratory operated by Battelle Memorial Institute for the United States Department of Energy under DOE contract number DE-AC05-76RL01830.
