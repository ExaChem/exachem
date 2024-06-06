/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "scf_matrix.hpp"

template<typename T>
Matrix exachem::scf::SCFMatrix::read_scf_mat(std::string matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  auto mfile_id = H5Fopen(matfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int64_t> rdims(2);
  auto                 attr_dataset = H5Dopen(mfile_id, "rdims", H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());

  Matrix mat         = Matrix::Zero(rdims[0], rdims[1]);
  auto   mdataset_id = H5Dopen(mfile_id, mname.c_str(), H5P_DEFAULT);

  /* Read the datasets. */
  H5Dread(mdataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mat;
}

template<typename T>
void exachem::scf::SCFMatrix::write_scf_mat(Matrix& C, std::string matfile) {
  std::string mname = fs::path(matfile).extension();
  mname.erase(0, 1); // remove "."

  const auto  N      = C.rows();
  const auto  Northo = C.cols();
  TensorType* buf    = C.data();

  /* Create a file. */
  hid_t file_id = H5Fcreate(matfile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t tsize        = N * Northo;
  hid_t   dataspace_id = H5Screate_simple(1, &tsize, NULL);

  /* Create dataset. */
  hid_t dataset_id = H5Dcreate(file_id, mname.c_str(), get_hdf5_dt<T>(), dataspace_id, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  /* Write the dataset. */
  /* herr_t status = */ H5Dwrite(dataset_id, get_hdf5_dt<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

  /* Create and write attribute information - dims */
  std::vector<int64_t> rdims{N, Northo};
  hsize_t              attr_size      = rdims.size();
  auto                 attr_dataspace = H5Screate_simple(1, &attr_size, NULL);
  auto attr_dataset = H5Dcreate(file_id, "rdims", H5T_NATIVE_INT64, attr_dataspace, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(attr_dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdims.data());
  H5Dclose(attr_dataset);
  H5Sclose(attr_dataspace);

  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template Matrix exachem::scf::SCFMatrix::read_scf_mat<double>(std::string matfile);
template void   exachem::scf::SCFMatrix::write_scf_mat<double>(Matrix& C, std::string matfile);
