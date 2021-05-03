#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#define MAX_THREAD      1024
#define MAX_BLOCK_SIZE  32

#define DEBUG_INITIALIZATION
// #define DEBUG_PREDICTION
// #define DEBUG_SET_MEASUREMENT
// #define DEBUG_CORRECTION
// #define DEBUG_SCORE
// #define DEBUG_RESAMPLE
// #define DEBUG_MAP_UPDATE

#define CUDA_CALL(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
        ROS_ERROR("GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
   }
}

#endif