
The installation instructions for this repository are the same as those for the [TAMM Library](https://github.com/NWChemEx/TAMM).

Installation
=============
- [Software Requirements](https://tamm.readthedocs.io/en/latest/prerequisites.html)

- [Build Instructions](https://tamm.readthedocs.io/en/latest/install.html)

Build instructions for a quick start
=====================================
## Step 1
```
git clone https://github.com/NWChemEx/TAMM.git
cd TAMM && mkdir build && cd build
```
- ### A detailed list of the cmake build options available are listed [here](https://tamm.readthedocs.io/en/latest/install.html)
```
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=<exachem-install-path> -DMODULES="CC" ..
make -j4 install
```

## Step 2
```
git clone https://github.com/ExaChem/exachem.git
cd exachem && mkdir build && cd build
CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=<exachem-install-path> -DMODULES="CC" ..
make -j4
```

## `NOTE:` The cmake configure line in Steps 1 and 2 should be the same.

Running the code
=================

```
export OMP_NUM_THREADS=1
export INPUT_FILE=$REPO_ROOT_PATH/inputs/ozone.json

mpirun -n 3 $REPO_INSTALL_PATH/bin/ExaChem $INPUT_FILE
```

