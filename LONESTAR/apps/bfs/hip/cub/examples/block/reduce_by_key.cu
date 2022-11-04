#include "hip/hip_runtime.h"


#include <hipcub/hipcub.hpp>


template <
    int         BLOCK_THREADS,          ///< Number of CTA threads
    typename    KeyT,                   ///< Key type
    typename    ValueT>                 ///< Value type
__global__ void Kernel()
{
    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef hipcub::KeyValuePair<int, ValueT> OffsetValuePairT;

    // Reduce-value-by-segment scan operator
    typedef hipcub::ReduceBySegmentOp<hipcub::Sum> ReduceBySegmentOpT;

    // Parameterized BlockDiscontinuity type for setting head flags
    typedef hipcub::BlockDiscontinuity<
            KeyT,
            BLOCK_THREADS>
        BlockDiscontinuityKeysT;

    // Parameterized BlockScan type
    typedef hipcub::BlockScan<
            OffsetValuePairT,
            BLOCK_THREADS,
            hipcub::BLOCK_SCAN_WARP_SCANS>
        BlockScanT;

    // Shared memory
    __shared__ union
    {
        typename BlockScanT::TempStorage                scan;           // Scan storage
        typename BlockDiscontinuityKeysT::TempStorage   discontinuity;  // Discontinuity storage
    } temp_storage;


    // Read data (each thread gets 3 items each, every 9 items is a segment)
    KeyT    my_keys[3]      = {threadIdx.x / 3, threadIdx.x / 3, threadIdx.x / 3};
    ValueT  my_values[3]    = {1, 1, 1};

    // Set head segment head flags
    int     my_flags[3];
    BlockDiscontinuityKeysT(temp_storage.discontinuity).FlagHeads(
        my_flags,
        my_keys,
        hipcub::Inequality());

    __syncthreads();






}
