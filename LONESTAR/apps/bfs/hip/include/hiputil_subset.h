#pragma once

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

#define HW_ID               4

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
{
  unsigned cu_id = __builtin_amdgcn_s_getreg(
                                             GETREG_IMMED(HW_ID_CU_ID_SIZE-1,
                                                          HW_ID_CU_ID_OFFSET,
                                                          HW_ID));
  unsigned se_id = __builtin_amdgcn_s_getreg(
                                             GETREG_IMMED(HW_ID_SE_ID_SIZE-1,
                                                          HW_ID_SE_ID_OFFSET,
                                                          HW_ID));

  /* Each shader engine has 16 CU normally, so shift cu_id accordingly */
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}
