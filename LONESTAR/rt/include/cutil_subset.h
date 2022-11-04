/*
   cutil_subset.h

   Implements a subset of the CUDA utilities. Part of the GGC source code. 

   TODO: actual owner copyright (NVIDIA) and license.
*/

#pragma once
#include "cub/cub.cuh"

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    hipError_t err = call;                                                    \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);                                            \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                         \
    hipError_t err = CUT_DEVICE_SYNCHRONIZE();                                 \
    if ( hipSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString( err) );              \
    } }

// used for calling s_getreg
extern "C" unsigned int __builtin_amdgcn_s_getreg(int);
//extern "C" __device__ inline unsigned __smid(void);

/*
   HW_ID Register bit structure
   WAVE_ID     3:0     Wave buffer slot number. 0-9.
   SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
   PIPE_ID     7:6     Pipeline from which the wave was dispatched.
   CU_ID       11:8    Compute Unit the wave is assigned to.
   SH_ID       12      Shader Array (within an SE) the wave is assigned to.
   SE_ID       14:13   Shader Engine the wave is assigned to.
   TG_ID       19:16   Thread-group ID
   VM_ID       23:20   Virtual Memory ID
   QUEUE_ID    26:24   Queue from which this wave was dispatched.
   STATE_ID    29:27   State ID (graphics only, not compute).
   ME_ID       31:30   Micro-engine ID.
*/
#define HW_ID               4

#define HW_ID_WAVE_ID_SIZE   4
#define HW_ID_WAVE_ID_OFFSET 0

#define HW_ID_CU_ID_SIZE    4
#define HW_ID_CU_ID_OFFSET  8

#define HW_ID_SE_ID_SIZE    2
#define HW_ID_SE_ID_OFFSET  13

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
*/
#define GETREG_IMMED(SZ,OFF,REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in
  bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.

  NOTE: Based on https://github.com/ROCm-Developer-Tools/HIP/blob/roc-3.1.0/include/hip/hcc_detail/device_functions.h#L1008
*/
static __device__ uint get_smid(void) {
  uint cu_id = __builtin_amdgcn_s_getreg(
					 GETREG_IMMED(HW_ID_CU_ID_SIZE-1,
						      HW_ID_CU_ID_OFFSET,
						      HW_ID));
  uint se_id = __builtin_amdgcn_s_getreg(
					 GETREG_IMMED(HW_ID_SE_ID_SIZE-1,
						      HW_ID_SE_ID_OFFSET,
						      HW_ID));

  /* Each shader engine has 16 CU normally, so shift cu_id accordingly */
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}

// this is based on the above math, but AMD does not have an example for it so could be wrong
static __device__ uint get_warpid(void) {
    uint wave_id = __builtin_amdgcn_s_getreg(
					     GETREG_IMMED(HW_ID_WAVE_ID_SIZE-1,
							  HW_ID_WAVE_ID_OFFSET,
							  HW_ID));
    return wave_id;
}

// since cub::WarpScan doesn't work very well with disabled threads in the warp
__device__ __forceinline__ void warp_active_count(int &first, int& offset, int& total) {
  unsigned int active = __ballot(1);
  total = __popc(active);
  offset = __popc(active & cub::LaneMaskLt());
  first = __ffs(active) - 1; // we know active != 0
}

// since cub::WarpScan doesn't work very well with disabled threads in the warp
__device__ __forceinline__ void warp_active_count_zero_active(int &first, int& offset, int& total) {
  unsigned int active = __ballot(1);
  total = __popc(active);
  offset = __popc(active & cub::LaneMaskLt());
  first = 0;
}
