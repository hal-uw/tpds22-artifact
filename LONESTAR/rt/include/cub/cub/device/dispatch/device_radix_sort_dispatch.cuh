#include "hip/hip_runtime.h"

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * hipcub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "../../block_range/block_range_radix_sort_upsweep.cuh"
#include "../../block_range/block_range_radix_sort_downsweep.cuh"
#include "../../block_range/block_range_scan.cuh"
#include "../../grid/grid_even_share.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Upsweep pass kernel entry point (multi-block).  Computes privatized digit histograms, one per block.
 */
template <
    typename                BlockRangeRadixSortUpsweepPolicy,   ///< Parameterized BlockRangeRadixSortUpsweepPolicy tuning policy type
    bool                    DESCENDING,                         ///< Whether or not the sorted-order is high-to-low
    typename                Key,                                ///< Key type
    typename                Offset>                             ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockRangeRadixSortUpsweepPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortUpsweepKernel(
    Key                     *d_keys,                            ///< [in] Input keys buffer
    Offset                  *d_spine,                           ///< [out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    Offset                  num_items,                          ///< [in] Total number of input data items
    int                     current_bit,                        ///< [in] Bit position of current radix digit
    int                     num_bits,                           ///< [in] Number of bits of current radix digit
    bool                    first_pass,                         ///< [in] Whether this is the first digit pass
    GridEvenShare<Offset>   even_share)                         ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
{
    // Parameterize BlockRangeRadixSortUpsweep type for the current configuration
    typedef BlockRangeRadixSortUpsweep<BlockRangeRadixSortUpsweepPolicy, Key, Offset> BlockRangeRadixSortUpsweepT;          // Primary

    // Shared memory storage
    __shared__ typename BlockRangeRadixSortUpsweepT::TempStorage temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    Offset bin_count;
    BlockRangeRadixSortUpsweepT(temp_storage, d_keys, current_bit, num_bits).ProcessRegion(
        even_share.block_offset,
        even_share.block_end,
        bin_count);

    // Write out digit counts (striped)
    if (threadIdx.x < BlockRangeRadixSortUpsweepT::RADIX_DIGITS)
    {
        int bin_idx = (DESCENDING) ?
            BlockRangeRadixSortUpsweepT::RADIX_DIGITS - threadIdx.x - 1 :
            threadIdx.x;

        d_spine[(gridDim.x * bin_idx) + blockIdx.x] = bin_count;
    }
}


/**
 * Spine scan kernel entry point (single-block).  Computes an exclusive prefix sum over the privatized digit histograms
 */
template <
    typename    BlockRangeScanPolicy,       ///< Parameterizable tuning policy type for hipcub::BlockRangeScan abstraction
    typename    Offset>                     ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockRangeScanPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortScanKernel(
    Offset      *d_spine,                   ///< [in,out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    int         num_counts)                 ///< [in] Total number of bin-counts
{
    // Parameterize the BlockRangeScan type for the current configuration
    typedef BlockRangeScan<BlockRangeScanPolicy, Offset*, Offset*, hipcub::Sum, Offset, Offset> BlockRangeScanT;

    // Shared memory storage
    __shared__ typename BlockRangeScanT::TempStorage temp_storage;

    if (blockIdx.x > 0) return;

    // Block scan instance
    BlockRangeScanT block_scan(temp_storage, d_spine, d_spine, hipcub::Sum(), Offset(0)) ;

    // Process full input tiles
    int block_offset = 0;
    BlockScanRunningPrefixOp<Offset, Sum> prefix_op(0, Sum());
    while (block_offset + BlockRangeScanT::TILE_ITEMS <= num_counts)
    {
        block_scan.ConsumeTile<true, false>(block_offset, prefix_op);
        block_offset += BlockRangeScanT::TILE_ITEMS;
    }
}


/**
 * Downsweep pass kernel entry point (multi-block).  Scatters keys (and values) into corresponding bins for the current digit place.
 */
template <
    typename                BlockRangeRadixSortDownsweepPolicy,     ///< Parameterizable tuning policy type for hipcub::BlockRangeRadixSortUpsweep abstraction
    bool                    DESCENDING,                             ///< Whether or not the sorted-order is high-to-low
    typename                Key,                                    ///< Key type
    typename                Value,                                  ///< Value type
    typename                Offset>                                 ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockRangeRadixSortDownsweepPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortDownsweepKernel(
    Key                     *d_keys_in,                             ///< [in] Input keys ping buffer
    Key                     *d_keys_out,                            ///< [in] Output keys pong buffer
    Value                   *d_values_in,                           ///< [in] Input values ping buffer
    Value                   *d_values_out,                          ///< [in] Output values pong buffer
    Offset                  *d_spine,                               ///< [in] Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    Offset                  num_items,                              ///< [in] Total number of input data items
    int                     current_bit,                            ///< [in] Bit position of current radix digit
    int                     num_bits,                               ///< [in] Number of bits of current radix digit
    bool                    first_pass,                             ///< [in] Whether this is the first digit pass
    bool                    last_pass,                              ///< [in] Whether this is the last digit pass
    GridEvenShare<Offset>   even_share)                             ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
{
    // Parameterize BlockRangeRadixSortDownsweep type for the current configuration
    typedef BlockRangeRadixSortDownsweep<BlockRangeRadixSortDownsweepPolicy, DESCENDING, Key, Value, Offset> BlockRangeRadixSortDownsweepT;

    // Shared memory storage
    __shared__  typename BlockRangeRadixSortDownsweepT::TempStorage temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.BlockInit();

    // Process input tiles
    BlockRangeRadixSortDownsweepT(temp_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, num_bits).ProcessRegion(
        even_share.block_offset,
        even_share.block_end);
}



/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRadixSort
 */
template <
    bool     DESCENDING,        ///< Whether or not the sorted-order is high-to-low
    typename Key,            ///< Key type
    typename Value,          ///< Value type
    typename Offset>         ///< Signed integer type for global offsets
struct DeviceRadixSortDispatch
{
    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // Primary UpsweepPolicy
        typedef BlockRangeRadixSortUpsweepPolicy <64,     CUB_MAX(1, 18 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128,    CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortUpsweepPolicy <64,     CUB_MAX(1, 22 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS - 1> AltUpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128,    CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_LDG, RADIX_BITS - 1> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef BlockRangeScanPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_WARP_SCANS> ScanPolicy;

        // Primary DownsweepPolicy
        typedef BlockRangeRadixSortDownsweepPolicy <64,   CUB_MAX(1, 18 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128,  CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortDownsweepPolicy <128,  CUB_MAX(1, 11 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS - 1> AltDownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128,  CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_DIRECT, LOAD_LDG, false, true, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS - 1> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;
    };


    /// SM30
    struct Policy300
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRangeRadixSortUpsweepPolicy <256, CUB_MAX(1, 7 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <256, CUB_MAX(1, 5 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortUpsweepPolicy <256, CUB_MAX(1, 7 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <256, CUB_MAX(1, 5 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef BlockRangeScanPolicy <1024, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 14 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 10 / SCALE_FACTOR), BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 14 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS - 1> AltDownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 10 / SCALE_FACTOR), BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeEightByte, RADIX_BITS - 1> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;
    };


    /// SM20
    struct Policy200
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRangeRadixSortUpsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortUpsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef BlockRangeScanPolicy <512, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRangeRadixSortDownsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortDownsweepPolicy <64, CUB_MAX(1, 18 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS - 1> AltDownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 13 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS - 1> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;
    };


    /// SM13
    struct Policy130
    {
        enum {
            KEYS_ONLY       = (Equals<Value, NullType>::VALUE),
            SCALE_FACTOR    = (CUB_MAX(sizeof(Key), sizeof(Value)) + 3) / 4,
            RADIX_BITS      = 5,
        };

        // UpsweepPolicy
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 19 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 19 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS> UpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, UpsweepPolicyKeys, UpsweepPolicyPairs>::Type UpsweepPolicy;

        // Alternate UpsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyKeys;
        typedef BlockRangeRadixSortUpsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltUpsweepPolicyKeys, AltUpsweepPolicyPairs>::Type AltUpsweepPolicy;

        // ScanPolicy
        typedef BlockRangeScanPolicy <256, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_WARP_SCANS> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRangeRadixSortDownsweepPolicy <64, CUB_MAX(1, 19 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <64, CUB_MAX(1, 19 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>::Type DownsweepPolicy;

        // Alternate DownsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS - 1> AltDownsweepPolicyKeys;
        typedef BlockRangeRadixSortDownsweepPolicy <128, CUB_MAX(1, 15 / SCALE_FACTOR), BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS - 1> AltDownsweepPolicyPairs;
        typedef typename If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>::Type AltDownsweepPolicy;
    };


    /// SM10
    struct Policy100
    {
        enum {
            RADIX_BITS = 4,
        };

        // UpsweepPolicy
        typedef BlockRangeRadixSortUpsweepPolicy <64, 9, LOAD_DEFAULT, RADIX_BITS> UpsweepPolicy;

        // Alternate UpsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortUpsweepPolicy <64, 9, LOAD_DEFAULT, RADIX_BITS - 1> AltUpsweepPolicy;

        // ScanPolicy
        typedef BlockRangeScanPolicy <256, 4, BLOCK_LOAD_VECTORIZE, false, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, false, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // DownsweepPolicy
        typedef BlockRangeRadixSortDownsweepPolicy <64, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS> DownsweepPolicy;

        // Alternate DownsweepPolicy for (RADIX_BITS-1)-bit passes
        typedef BlockRangeRadixSortDownsweepPolicy <64, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false, false, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_SCATTER_TWO_PHASE, hipSharedMemBankSizeFourByte, RADIX_BITS - 1> AltDownsweepPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxUpsweepPolicy         : PtxPolicy::UpsweepPolicy {};
    struct PtxAltUpsweepPolicy      : PtxPolicy::AltUpsweepPolicy {};
    struct PtxScanPolicy            : PtxPolicy::ScanPolicy {};
    struct PtxDownsweepPolicy       : PtxPolicy::DownsweepPolicy {};
    struct PtxAltDownsweepPolicy    : PtxPolicy::AltDownsweepPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename Policy,
        typename KernelConfig,
        typename UpsweepKernelPtr,          ///< Function type of hipcub::RadixSortUpsweepKernel
        typename ScanKernelPtr,            ///< Function type of hipcub::SpineScanKernel
        typename DownsweepKernelPtr>        ///< Function type of hipcub::RadixSortUpsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t InitConfigs(
        int                     sm_version,
        int                     sm_count,
        KernelConfig            &upsweep_config,
        KernelConfig            &alt_upsweep_config,
        KernelConfig            &scan_config,
        KernelConfig            &downsweep_config,
        KernelConfig            &alt_downsweep_config,
        UpsweepKernelPtr        upsweep_kernel,
        UpsweepKernelPtr        alt_upsweep_kernel,
        ScanKernelPtr           scan_kernel,
        DownsweepKernelPtr      downsweep_kernel,
        DownsweepKernelPtr      alt_downsweep_kernel)
    {
        hipError_t error;
        do {
            if (CubDebug(error = upsweep_config.template         InitUpsweepPolicy<typename Policy::UpsweepPolicy>(         sm_version, sm_count, upsweep_kernel))) break;
            if (CubDebug(error = alt_upsweep_config.template     InitUpsweepPolicy<typename Policy::AltUpsweepPolicy>(      sm_version, sm_count, alt_upsweep_kernel))) break;
            if (CubDebug(error = scan_config.template            InitScanPolicy<typename Policy::ScanPolicy>(               sm_version, sm_count, scan_kernel))) break;
            if (CubDebug(error = downsweep_config.template       InitDownsweepPolicy<typename Policy::DownsweepPolicy>(     sm_version, sm_count, downsweep_kernel))) break;
            if (CubDebug(error = alt_downsweep_config.template   InitDownsweepPolicy<typename Policy::AltDownsweepPolicy>(  sm_version, sm_count, alt_downsweep_kernel))) break;

        } while (0);

        return error;
    }


    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <
        typename KernelConfig,
        typename UpsweepKernelPtr,          ///< Function type of hipcub::RadixSortUpsweepKernel
        typename ScanKernelPtr,            ///< Function type of hipcub::SpineScanKernel
        typename DownsweepKernelPtr>        ///< Function type of hipcub::RadixSortUpsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t InitConfigs(
        int                     ptx_version,
        int                     sm_version,
        int                     sm_count,
        KernelConfig            &upsweep_config,
        KernelConfig            &alt_upsweep_config,
        KernelConfig            &scan_config,
        KernelConfig            &downsweep_config,
        KernelConfig            &alt_downsweep_config,
        UpsweepKernelPtr        upsweep_kernel,
        UpsweepKernelPtr        alt_upsweep_kernel,
        ScanKernelPtr          scan_kernel,
        DownsweepKernelPtr      downsweep_kernel,
        DownsweepKernelPtr      alt_downsweep_kernel)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        hipError_t error;
        do {

            if (CubDebug(error = upsweep_config.template InitUpsweepPolicy<PtxUpsweepPolicy>(               sm_version, sm_count, upsweep_kernel))) break;
            if (CubDebug(error = alt_upsweep_config.template InitUpsweepPolicy<PtxAltUpsweepPolicy>(        sm_version, sm_count, alt_upsweep_kernel))) break;
            if (CubDebug(error = scan_config.template InitScanPolicy<PtxScanPolicy>(                        sm_version, sm_count, scan_kernel))) break;
            if (CubDebug(error = downsweep_config.template InitDownsweepPolicy<PtxDownsweepPolicy>(         sm_version, sm_count, downsweep_kernel))) break;
            if (CubDebug(error = alt_downsweep_config.template InitDownsweepPolicy<PtxAltDownsweepPolicy>(  sm_version, sm_count, alt_downsweep_kernel))) break;

        } while (0);

        return error;

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        hipError_t error;
        if (ptx_version >= 350)
        {
            error = InitConfigs<Policy350>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel);
        }
        else if (ptx_version >= 300)
        {
            error = InitConfigs<Policy300>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel);
        }
        else if (ptx_version >= 200)
        {
            error = InitConfigs<Policy200>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel);
        }
        else if (ptx_version >= 130)
        {
            error = InitConfigs<Policy130>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel);
        }
        else
        {
            error = InitConfigs<Policy100>(sm_version, sm_count, upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config, upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel);
        }

        return error;

    #endif
    }



    /**
     * Kernel kernel dispatch configurations
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     tile_size;
        hipSharedMemConfig     smem_config;
        int                     radix_bits;
        int                     sm_occupancy;       // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
        int                     max_grid_size;
        int                     subscription_factor;

        template <typename UpsweepPolicy, typename UpsweepKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ hipError_t InitUpsweepPolicy(
            int sm_version, int sm_count, UpsweepKernelPtr upsweep_kernel)
        {
            block_threads               = UpsweepPolicy::BLOCK_THREADS;
            items_per_thread            = UpsweepPolicy::ITEMS_PER_THREAD;
            radix_bits                  = UpsweepPolicy::RADIX_BITS;
            smem_config                 = hipSharedMemBankSizeFourByte;
            tile_size                   = block_threads * items_per_thread;
            hipError_t retval          = MaxSmOccupancy(sm_occupancy, sm_version, upsweep_kernel, block_threads);
            subscription_factor         = CUB_SUBSCRIPTION_FACTOR(sm_version);
            max_grid_size               = (sm_occupancy * sm_count) * subscription_factor;

            return retval;
        }

        template <typename ScanPolicy, typename ScanKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ hipError_t InitScanPolicy(
            int sm_version, int sm_count, ScanKernelPtr scan_kernel)
        {
            block_threads               = ScanPolicy::BLOCK_THREADS;
            items_per_thread            = ScanPolicy::ITEMS_PER_THREAD;
            radix_bits                  = 0;
            smem_config                 = hipSharedMemBankSizeFourByte;
            tile_size                   = block_threads * items_per_thread;
            sm_occupancy                = 1;
            subscription_factor         = 1;
            max_grid_size               = 1;

            return hipSuccess;
        }

        template <typename DownsweepPolicy, typename DownsweepKernelPtr>
        CUB_RUNTIME_FUNCTION __forceinline__ hipError_t InitDownsweepPolicy(
            int sm_version, int sm_count, DownsweepKernelPtr downsweep_kernel)
        {
            block_threads               = DownsweepPolicy::BLOCK_THREADS;
            items_per_thread            = DownsweepPolicy::ITEMS_PER_THREAD;
            radix_bits                  = DownsweepPolicy::RADIX_BITS;
            smem_config                 = DownsweepPolicy::SMEM_CONFIG;
            tile_size                   = block_threads * items_per_thread;
            hipError_t retval          = MaxSmOccupancy(sm_occupancy, sm_version, downsweep_kernel, block_threads);
            subscription_factor         = CUB_SUBSCRIPTION_FACTOR(sm_version);
            max_grid_size               = (sm_occupancy * sm_count) * subscription_factor;

            return retval;
        }
    };


    /******************************************************************************
     * Allocation of device temporaries
     ******************************************************************************/

    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t AllocateTemporaries(
        void                    *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        Offset*                 &d_spine,                       ///< [out] Digit count histograms per thread block
        KernelConfig            &scan_config,                   ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
        KernelConfig            &downsweep_config)              ///< [in] Dispatch parameters that match the policy that \p downsweep_kernel was compiled for
    {
        hipError_t error = hipSuccess;
        do
        {
            // Get spine size (conservative)
            int spine_size = (downsweep_config.max_grid_size * (1 << downsweep_config.radix_bits)) + scan_config.tile_size;

            // Temporary storage allocation requirements
            void* allocations[1];
            size_t allocation_sizes[1] =
            {
                spine_size * sizeof(Offset),    // bytes needed for privatized block digit histograms
            };

            // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return hipSuccess;

            // Alias the allocation for the privatized per-block digit histograms
            d_spine = (Offset*) allocations[0];

        } while(0);

        return error;
    }


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide radix sort using the
     * specified kernel functions.
     */
    template <
        typename                UpsweepKernelPtr,               ///< Function type of hipcub::RadixSortUpsweepKernel
        typename                ScanKernelPtr,                  ///< Function type of hipcub::SpineScanKernel
        typename                DownsweepKernelPtr>             ///< Function type of hipcub::RadixSortUpsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Dispatch(
        DoubleBuffer<Key>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        Offset                  *d_spine,                       ///< [in] Digit count histograms per thread block
        int                     spine_size,                     ///< [in] Number of histogram counters
        Offset                  num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        KernelConfig            &upsweep_config,                ///< [in] Dispatch parameters that match the policy that \p upsweep_kernel was compiled for
        KernelConfig            &scan_config,                   ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
        KernelConfig            &downsweep_config,              ///< [in] Dispatch parameters that match the policy that \p downsweep_kernel was compiled for
        UpsweepKernelPtr        upsweep_kernel,                 ///< [in] Kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
        ScanKernelPtr           scan_kernel,                    ///< [in] Kernel function pointer to parameterization of hipcub::SpineScanKernel
        DownsweepKernelPtr      downsweep_kernel)               ///< [in] Kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(hipErrorNotSupported );

#else

        hipError_t error = hipSuccess;
        do
        {
            // Get even-share work distribution descriptor
            GridEvenShare<Offset> even_share(num_items, downsweep_config.max_grid_size, CUB_MAX(downsweep_config.tile_size, upsweep_config.tile_size));

#if (CUB_PTX_ARCH == 0)
            // Get current smem bank configuration
            hipSharedMemConfig original_smem_config;
            if (CubDebug(error = hipDeviceGetSharedMemConfig(&original_smem_config))) break;
            hipSharedMemConfig current_smem_config = original_smem_config;
#endif
            // Iterate over digit places
            int current_bit = begin_bit;
            while (current_bit < end_bit)
            {
                int num_bits = CUB_MIN(end_bit - current_bit, downsweep_config.radix_bits);

#if (CUB_PTX_ARCH == 0)
                // Update smem config if necessary
                if (current_smem_config != upsweep_config.smem_config)
                {
                    if (CubDebug(error = hipDeviceSetSharedMemConfig(upsweep_config.smem_config))) break;
                    current_smem_config = upsweep_config.smem_config;
                }
#endif

                // Log upsweep_kernel configuration
                if (debug_synchronous)
                    CubLog("Invoking hipLaunchKernelGGL(upsweep_kernel, dim3(%d), dim3(%d), 0, %lld), %d smem config, %d items per thread, %d SM occupancy, selector %d, current bit %d, bit_grain %d\n",
                    even_share.grid_size, upsweep_config.block_threads, (long long) stream, upsweep_config.smem_config, upsweep_config.items_per_thread, upsweep_config.sm_occupancy, d_keys.selector, current_bit, downsweep_config.radix_bits);

                // Invoke upsweep_kernel with same grid size as downsweep_kernel
                hipLaunchKernelGGL(upsweep_kernel, dim3(even_share.grid_size), dim3(upsweep_config.block_threads), 0, stream, 
                    d_keys.d_buffers[d_keys.selector],
                    d_spine,
                    num_items,
                    current_bit,
                    num_bits,
                    (current_bit == begin_bit),
                    even_share);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log scan_kernel configuration
                if (debug_synchronous) CubLog("Invoking hipLaunchKernelGGL(scan_kernel, dim3(%d), dim3(%d), 0, %lld), %d items per thread\n",
                    1, scan_config.block_threads, (long long) stream, scan_config.items_per_thread);

                // Invoke scan_kernel
                hipLaunchKernelGGL(scan_kernel, dim3(1), dim3(scan_config.block_threads), 0, stream, 
                    d_spine,
                    spine_size);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;


#if (CUB_PTX_ARCH == 0)
                // Update smem config if necessary
                if (current_smem_config != downsweep_config.smem_config)
                {
                    if (CubDebug(error = hipDeviceSetSharedMemConfig(downsweep_config.smem_config))) break;
                    current_smem_config = downsweep_config.smem_config;
                }
#endif
                // Log downsweep_kernel configuration
                if (debug_synchronous) CubLog("Invoking hipLaunchKernelGGL(downsweep_kernel, dim3(%d), dim3(%d), 0, %lld), %d smem config, %d items per thread, %d SM occupancy\n",
                    even_share.grid_size, downsweep_config.block_threads, (long long) stream, downsweep_config.smem_config, downsweep_config.items_per_thread, downsweep_config.sm_occupancy);

                // Invoke downsweep_kernel
                hipLaunchKernelGGL(downsweep_kernel, dim3(even_share.grid_size), dim3(downsweep_config.block_threads), 0, stream, 
                    d_keys.d_buffers[d_keys.selector],
                    d_keys.d_buffers[d_keys.selector ^ 1],
                    d_values.d_buffers[d_values.selector],
                    d_values.d_buffers[d_values.selector ^ 1],
                    d_spine,
                    num_items,
                    current_bit,
                    num_bits,
                    (current_bit == begin_bit),
                    (current_bit + downsweep_config.radix_bits >= end_bit),
                    even_share);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Invert selectors
                d_keys.selector ^= 1;
                d_values.selector ^= 1;

                // Update current bit position
                current_bit += downsweep_config.radix_bits;
            }

#if (CUB_PTX_ARCH == 0)
            // Reset smem config if necessary
            if (current_smem_config != original_smem_config)
            {
                if (CubDebug(error = hipDeviceSetSharedMemConfig(original_smem_config))) break;
            }
#endif

        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */
    template <
        typename UpsweepKernelPtr,          ///< Function type of hipcub::RadixSortUpsweepKernel
        typename ScanKernelPtr,             ///< Function type of hipcub::SpineScanKernel
        typename DownsweepKernelPtr>        ///< Function type of hipcub::RadixSortUpsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Dispatch(
        void                    *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        Offset                  num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        UpsweepKernelPtr        upsweep_kernel,                 ///< [in] Kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
        UpsweepKernelPtr        alt_upsweep_kernel,             ///< [in] Alternate kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
        ScanKernelPtr           scan_kernel,                    ///< [in] Kernel function pointer to parameterization of hipcub::SpineScanKernel
        DownsweepKernelPtr      downsweep_kernel,               ///< [in] Kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
        DownsweepKernelPtr      alt_downsweep_kernel)           ///< [in] Alternate kernel function pointer to parameterization of hipcub::RadixSortUpsweepKernel
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(hipErrorNotSupported );

#else

        hipError_t error = hipSuccess;

        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = hipGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = hipDeviceGetAttribute (&sm_count, hipDeviceAttributeMultiprocessorCount, device_ordinal))) break;

            // Get kernel kernel dispatch configurations
            KernelConfig upsweep_config;
            KernelConfig alt_upsweep_config;
            KernelConfig scan_config;
            KernelConfig downsweep_config;
            KernelConfig alt_downsweep_config;

            if (CubDebug(error = InitConfigs(ptx_version, sm_version, sm_count,
                upsweep_config, alt_upsweep_config, scan_config, downsweep_config, alt_downsweep_config,
                upsweep_kernel, alt_upsweep_kernel, scan_kernel, downsweep_kernel, alt_downsweep_kernel))) break;

            // Get spine sizes (conservative)
            int spine_size      = (downsweep_config.max_grid_size * (1 << downsweep_config.radix_bits)) + scan_config.tile_size;
            int alt_spine_size  = (alt_downsweep_config.max_grid_size * (1 << alt_downsweep_config.radix_bits)) + scan_config.tile_size;

            // Allocate temporaries
            Offset *d_spine;
            if (spine_size > alt_spine_size)
            {
                if (CubDebug(error = AllocateTemporaries(d_temp_storage, temp_storage_bytes, d_spine, scan_config, downsweep_config))) break;
            }
            else
            {
                if (CubDebug(error = AllocateTemporaries(d_temp_storage, temp_storage_bytes, d_spine, scan_config, alt_downsweep_config))) break;
            }

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return hipSuccess;

            // Run radix sorting passes
            int num_bits = end_bit - begin_bit;
            int remaining_bits = num_bits % downsweep_config.radix_bits;

            if (remaining_bits != 0)
            {
                // Run passes of alternate configuration
                int max_alt_passes  = downsweep_config.radix_bits - remaining_bits;
                int alt_end_bit     = CUB_MIN(end_bit, begin_bit + (max_alt_passes * alt_downsweep_config.radix_bits));

                if (CubDebug(error = Dispatch(
                    d_keys,
                    d_values,
                    d_spine,
                    alt_spine_size,
                    num_items,
                    begin_bit,
                    alt_end_bit,
                    stream,
                    debug_synchronous,
                    alt_upsweep_config,
                    scan_config,
                    alt_downsweep_config,
                    alt_upsweep_kernel,
                    scan_kernel,
                    alt_downsweep_kernel))) break;

                begin_bit = alt_end_bit;
            }

            // Run passes of primary configuration
            if (CubDebug(error = Dispatch(
                d_keys,
                d_values,
                d_spine,
                spine_size,
                num_items,
                begin_bit,
                end_bit,
                stream,
                debug_synchronous,
                upsweep_config,
                scan_config,
                downsweep_config,
                upsweep_kernel,
                scan_kernel,
                downsweep_kernel))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */

    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Dispatch(
        void                    *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>       &d_keys,                        ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value>     &d_values,                      ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        Offset                  num_items,                      ///< [in] Number of items to reduce
        int                     begin_bit,                      ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t            stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous)              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        return Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous,
            RadixSortUpsweepKernel<PtxUpsweepPolicy, DESCENDING, Key, Offset>,
            RadixSortUpsweepKernel<PtxAltUpsweepPolicy, DESCENDING, Key, Offset>,
            RadixSortScanKernel<PtxScanPolicy, Offset>,
            RadixSortDownsweepKernel<PtxDownsweepPolicy, DESCENDING, Key, Value, Offset>,
            RadixSortDownsweepKernel<PtxAltDownsweepPolicy, DESCENDING, Key, Value, Offset>);
    }

};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


