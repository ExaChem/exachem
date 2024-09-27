# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class Exachem(CMakePackage,CudaPackage):
    """ExaChem: Open Source Exascale Computational Chemistry Software"""

    homepage = "https://github.com/ExaChem/exachem"
    git      = "https://github.com/ExaChem/exachem.git"

    tags = []

    version('main', branch='main')

    depends_on('tamm')
    depends_on('tamm +cuda', when='+cuda')
    depends_on('mpi')
    depends_on('intel-oneapi-mkl +cluster')
    depends_on('cmake@3.22:')
    depends_on('cuda@11.8:', when='+cuda')
    depends_on('hdf5 +mpi')
    # Still need to update libint recipe for 2.9.x
    #depends_on('libint@2.9:')
    conflicts("+cuda", when="cuda_arch=none")
    variant(
        "cuda_arch",
        description="CUDA architecture",
        values=("none",) + CudaPackage.cuda_arch_values,
        default="none",
        multi=False,
        sticky=True,
        when="+cuda",
    )

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
            cuda_arch = self.spec.variants["cuda_arch"].value
            args.append( "-DTAMM_ENABLE_CUDA=ON" )
            args.append("-DGPU_ARCH=" + cuda_arch)

        return args
