#include <stdio.h>

int main(int argc, char **argv){
    cudaDeviceProp dP;
    float min_cc = 5.0;

    int rc = cudaGetDeviceProperties(&dP, 0);
    if(rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s", cudaGetErrorString(error));
        return rc; /* Failure */
    }
    if((dP.major+(dP.minor/10.0)) < min_cc) {
        printf("Minimum CUDA Compute Capability of %2.1f required:  %d.%d found\n", min_cc, dP.major, dP.minor);
        return 1; /* Failure */
    } else {
        printf("%d%d", dP.major, dP.minor);
        return 0; /* Success */
    }
}

