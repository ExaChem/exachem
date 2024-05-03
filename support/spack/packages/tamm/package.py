# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Tamm(CMakePackage,CudaPackage):
    """Tensor Algebra for Many-body Methods (TAMM)
    
    See https://doi.org/10.1063/5.0142433 for further information.
    """

    homepage = "https://github.com/NWChemEx/TAMM"
    # url      = "https://github.com/NWChemEx/TAMM"
    git      = "https://github.com/NWChemEx/TAMM.git"

    tags = ['ecp', 'ecp-apps']

    version('main', branch='main')

    depends_on('mpi')
    depends_on('intel-oneapi-mkl +cluster')
    depends_on('cmake@3.22:')
    depends_on('cuda@11.8:', when='+cuda')
    depends_on('hdf5 +mpi')
    # Still need to update libint recipe for 2.9.x
    #depends_on('libint@2.9:')
    conflicts("+cuda", when="cuda_arch=none")

    def cmake_args(self):
        args = [
            # This was not able to detect presence of libint in first test
            #'-DLibInt2_ROOT=%s' % self.spec['libint'].prefix,
            '-DMODULES=CC;DFT',
            '-DHDF5_ROOT=%s' % self.spec['hdf5'].prefix,
            '-DLINALG_VENDOR=IntelMKL',
            '-DLINALG_PREFIX=%s' % join_path(self.spec['intel-oneapi-mkl'].prefix, 'mkl', 'latest'),
        ]
        if '+cuda' in self.spec:
            args.append( "-DTAMM_ENABLE_CUDA=ON" )
            args.append("-DGPU_ARCH=" + self.spec.variants["cuda_arch"].value)

        return args