.. role:: aspect (emphasis)
.. role:: sep (strong)
.. rst-class:: dl-parameters


Cholesky Decomposition
========================


The cholesky decomposition procedure is described in the following papers

- Peng, Bo, and Karol Kowalski. "Highly Efficient and Scalable Compound Decomposition of Two-Electron Integral Tensor and Its Application in Coupled Cluster Calculations." *J. Chem. Theory Comput.* https://doi.org/10.1021/acs.jctc.7b00605.

- Peng, Bo, and Karol Kowalski. "Low-rank factorization of electron integral tensors and its application in electronic structure theory." *Chemical Physics Letters.* https://doi.org/10.1016/j.cplett.2017.01.056.

| :ref:`CD options <CD>`

.. _CD:

Options used in the Cholesky decomposition of atomic-orbital based two-electron integral tensor.

.. code-block:: json

  "CD": {
    "diagonal": 1e-5,
    "itilesize": 1000,
    "write_cv": [false,5000]
  }

:diagonal: ``[default=1e-5]`` The diagonal threshold used to terminate the decomposition procedure and truncate the Cholesky vectors.

:itilesize: ``[default=1000]`` The tilesize for the cholesky dimension representing the number of cholesky vectors. It is recommended to leave this at the default value.

The following options are applicable only for calculations involving :math:`\geq` 1000 basis functions. They are used for restarting the cholesky decomposition procedure.

:write_cv: ``[default=[false,5000]]`` When enabled, it performs parallel IO to write the tensor containing the AO cholesky vectors to disk. Enabling this option implies restart. The integer represents a count, indicating that the Cholesky vectors should be written to disk after every *count* vectors are computed.
