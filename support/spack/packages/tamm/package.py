# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Tamm(CMakePackage,CudaPackage):
    """Tensor Algebra for Many-body Methods (TAMM)
    
    See https://doi.org/10.1063/5.0142433 for further information.
    """

    homepage = "https://github.com/NWChemEx-Project/TAMM"
    # url      = "https://github.com/NWChemEx-Project/TAMM"
    git      = "https://github.com/NWChemEx-Project/TAMM.git"

    tags = ['ecp', 'ecp-apps']

    version('main', branch='main')

    depends_on('mpi')
    depends_on('intel-oneapi-mkl +cluster')
    depends_on('cmake@3.22:')
    depends_on('cuda@11.5:', when='+cuda')
    depends_on('hdf5 +mpi')
    # Still need to update libint recipe for 2.7.x
    #depends_on('libint@2.7:')

    def cmake_args(self):
        args = [
            # This was not able to detect presence of libint in first test
            #'-DLibInt2_ROOT=%s' % self.spec['libint'].prefix,
            '-DMODULES=CC',
            '-DHDF5_ROOT=%s' % self.spec['hdf5'].prefix,
            '-DLINALG_VENDOR=IntelMKL',
            '-DLINALG_PREFIX=%s' % join_path(self.spec['intel-oneapi-mkl'].prefix, 'mkl', 'latest'),
        ]
        if '+cuda' in self.spec:
            args.extend([ '-DUSE_CUDA=ON',
             ])

        return args