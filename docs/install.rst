
The installation instructions for this repository are the same as those
for the `TAMM Library <https://github.com/NWChemEx/TAMM>`__.

Installation
============

-  `Software
   Requirements <https://tamm.readthedocs.io/en/latest/prerequisites.html>`__

-  `Build
   Instructions <https://tamm.readthedocs.io/en/latest/install.html>`__

Dependencies
------------

In addition to the TAMM `dependencies <https://tamm.readthedocs.io/en/latest/install.html>`__, the following ExaChem dependencies are also automatically built by TAMM.

* Libint
* Libecpint

Build instructions for a quick start
------------------------------------

.. note:: 
   When using a specific git tag for TAMM and ExaChem, add -DCMSB_TAG=<tag-name> to the CMake configure lines listed in the steps below.

Step 1

::

   git clone https://github.com/NWChemEx/TAMM.git
   cd TAMM && mkdir build && cd build

-  .. rubric:: A detailed list of the cmake build options available are
      listed
      `here <https://tamm.readthedocs.io/en/latest/install.html>`__
      :name: a-detailed-list-of-the-cmake-build-options-available-are-listed-here

::

   CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=<exachem-install-path> -DMODULES="CC" ..
   make -j4 install

Step 2

::

   git clone https://github.com/ExaChem/exachem.git
   cd exachem && mkdir build && cd build
   CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=<exachem-install-path> -DMODULES="CC" ..
   make -j4 install

``NOTE:`` The cmake configure line in Steps 1 and 2 should be the exact same.


Running the code
----------------

::

   export OMP_NUM_THREADS=1
   export INPUT_FILE=$REPO_ROOT_PATH/inputs/ozone.json

   mpirun -n 3 $REPO_INSTALL_PATH/bin/ExaChem $INPUT_FILE
