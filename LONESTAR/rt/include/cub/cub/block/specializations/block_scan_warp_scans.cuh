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
 * hipcub::BlockScanWarpscans provides warpscan-based variants of parallel prefix scan across a CUDA threadblock.
 */

#pragma once

#include "../../util_arch.cuh"
#include "../../util_ptx.cuh"
#include "../../warp/warp_scan.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \brief BlockScanWarpScans provides warpscan-based variants of parallel prefix scan across a CUDA threadblock.
 */
template <
    typename    T,
    int         BLOCK_DIM_X,    ///< The thread block length in threads along the X dimension
    int         BLOCK_DIM_Y,    ///< The thread block length in threads along the Y dimension
    int         BLOCK_DIM_Z,    ///< The thread block length in threads along the Z dimension
    int         PTX_ARCH>       ///< The PTX compute capability for which to to specialize this collective
struct BlockScanWarpScans
{
    /// Constants
    enum
    {
        /// Number of warp threads
        WARP_THREADS = CUB_WARP_THREADS(PTX_ARCH),

        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        /// Number of active warps
        WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,
    };

    ///  WarpScan utility type
    typedef WarpScan<T, WARP_THREADS, PTX_ARCH> WarpScan;

    /// Shared memory storage layout type
    struct _TempStorage
    {
        typename WarpScan::TempStorage      warp_scan[WARPS];           ///< Buffer for warp-synchronous scans
        T                                   warp_aggregates[WARPS];     ///< Shared totals from each warp-synchronous scan
        T                                   block_prefix;               ///< Shared prefix for the entire threadblock
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    // Thread fields
    _TempStorage &temp_storage;
    int linear_tid;
    int warp_id;
    int lane_id;


    /// Constructor
    __device__ __forceinline__ BlockScanWarpScans(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)),
        warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS),
        lane_id(LaneId())
    {}

    template <typename ScanOp, int WARP>
    __device__ __forceinline__ void ApplyWarpAggregates(
        T               &partial,           ///< [out] The calling thread's partial reduction
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
        bool            lane_valid,         ///< [in] Whether or not the partial belonging to the current thread is valid
        Int2Type<WARP>  addend_warp)
    {
            T inclusive = scan_op(block_aggregate, partial);
            if (warp_id == WARP)
            {
                partial = (lane_valid) ?
                    inclusive :
                    block_aggregate;
            }

            T addend = temp_storage.warp_aggregates[WARP];
            block_aggregate = scan_op(block_aggregate, addend);

            ApplyWarpAggregates(partial, scan_op, block_aggregate, lane_valid, Int2Type<WARP + 1>());
    }

    template <typename ScanOp>
    __device__ __forceinline__ void ApplyWarpAggregates(
        T               &partial,           ///< [out] The calling thread's partial reduction
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
        bool            lane_valid,         ///< [in] Whether or not the partial belonging to the current thread is valid
        Int2Type<WARPS> addend_warp)
    {}


    /// Update the calling thread's partial reduction with the warp-wide aggregates from preceding warps.  Also returns block-wide aggregate in <em>thread</em><sub>0</sub>.
    template <typename ScanOp>
    __device__ __forceinline__ void ApplyWarpAggregates(
        T               &partial,           ///< [out] The calling thread's partial reduction
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>WARP_THREADS - 1</sub> only]</b> Warp-wide aggregate reduction of input items
        T               &block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
        bool            lane_valid = true)  ///< [in] Whether or not the partial belonging to the current thread is valid
    {
        // Last lane in each warp shares its warp-aggregate
        if (lane_id == WARP_THREADS - 1)
            temp_storage.warp_aggregates[warp_id] = warp_aggregate;

        __syncthreads();

        block_aggregate = temp_storage.warp_aggregates[0];

#if __CUDA_ARCH__ <= 130

        // Use template unrolling for SM1x (since the PTX backend can't handle it)
        ApplyWarpAggregates(partial, scan_op, block_aggregate, lane_valid, Int2Type<1>());

#else

        // Use the pragma unrolling (since it uses less registers)
        #pragma unroll
        for (int WARP = 1; WARP < WARPS; WARP++)
        {
            T inclusive = scan_op(block_aggregate, partial);
            if (warp_id == WARP)
            {
                partial = (lane_valid) ?
                    inclusive :
                    block_aggregate;
            }

            T addend = temp_storage.warp_aggregates[WARP];
            block_aggregate = scan_op(block_aggregate, addend);
        }

#endif
    }


    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,              ///< [in] Calling thread's input items
        T               &output,            ///< [out] Calling thread's output items (may be aliased to \p input)
        const T         &identity,          ///< [in] Identity value
        ScanOp          scan_op,            ///< [in] Binary scan operator
        T               &block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
    {
        T inclusive_output;
        WarpScan(temp_storage.warp_scan[warp_id]).Scan(input, inclusive_output, output, identity, scan_op);

        // Update outputs and block_aggregate with warp-wide aggregates
        ApplyWarpAggregates(output, scan_op, inclusive_output, block_aggregate);
    }


    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
        typename ScanOp,
        typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void ExclusiveScan(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T                       identity,                       ///< [in] Identity value
        ScanOp                  scan_op,                        ///< [in] Binary scan operator
        T                       &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_callback_op value)
        BlockPrefixCallbackOp   &block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        ExclusiveScan(input, output, identity, scan_op, block_aggregate);

        // Use the first warp to determine the threadblock prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate threadblock prefix into outputs
        T block_prefix = temp_storage.block_prefix;
        output = scan_op(block_prefix, output);
    }


    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no identity value, the output computed for <em>thread</em><sub>0</sub> is undefined.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        T inclusive_output;
        WarpScan(temp_storage.warp_scan[warp_id]).Scan(input, inclusive_output, output, scan_op);

        // Update outputs and block_aggregate with warp-wide aggregates
        ApplyWarpAggregates(output, scan_op, inclusive_output, block_aggregate, (lane_id > 0));
    }


    /// Computes an exclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
        typename ScanOp,
        typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void ExclusiveScan(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp                  scan_op,                        ///< [in] Binary scan operator
        T                       &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_callback_op value)
        BlockPrefixCallbackOp   &block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        ExclusiveScan(input, output, scan_op, block_aggregate);

        // Use the first warp to determine the threadblock prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate threadblock prefix into outputs
        T block_prefix = temp_storage.block_prefix;
        output = (linear_tid == 0) ?
            block_prefix :
            scan_op(block_prefix, output);
    }


    /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    __device__ __forceinline__ void ExclusiveSum(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T                       &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        Sum     scan_op;
        T       inclusive_output;

        WarpScan(temp_storage.warp_scan[warp_id]).Sum(input, inclusive_output, output);

        // Update outputs and block_aggregate with warp-wide aggregates from lane WARP_THREADS-1
        ApplyWarpAggregates(output, scan_op, inclusive_output, block_aggregate);
    }


    /// Computes an exclusive threadblock-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void ExclusiveSum(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T                       &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_callback_op value)
        BlockPrefixCallbackOp   &block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        ExclusiveSum(input, output, block_aggregate);

        // Use the first warp to determine the threadblock prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate threadblock prefix into outputs
        Sum scan_op;
        T block_prefix = temp_storage.block_prefix;
        output = scan_op(block_prefix, output);
    }


    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        WarpScan(temp_storage.warp_scan[warp_id]).InclusiveScan(input, output, scan_op);

        // Update outputs and block_aggregate with warp-wide aggregates from lane WARP_THREADS-1
        ApplyWarpAggregates(output, scan_op, output, block_aggregate);

    }


    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
        typename ScanOp,
        typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void InclusiveScan(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        ScanOp                  scan_op,                        ///< [in] Binary scan operator
        T                       &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_callback_op value)
        BlockPrefixCallbackOp   &block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        InclusiveScan(input, output, scan_op, block_aggregate);

        // Use the first warp to determine the threadblock prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate threadblock prefix into outputs
        T block_prefix = temp_storage.block_prefix;
        output = scan_op(block_prefix, output);
    }


    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    __device__ __forceinline__ void InclusiveSum(
        T               input,                          ///< [in] Calling thread's input item
        T               &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T               &block_aggregate)               ///< [out] Threadblock-wide aggregate reduction of input items
    {
        WarpScan(temp_storage.warp_scan[warp_id]).InclusiveSum(input, output);

        // Update outputs and block_aggregate with warp-wide aggregates from lane WARP_THREADS-1
        ApplyWarpAggregates(output, Sum(), output, block_aggregate);
    }


    /// Computes an inclusive threadblock-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Instead of using 0 as the threadblock-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the threadblock's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void InclusiveSum(
        T                       input,                          ///< [in] Calling thread's input item
        T                       &output,                        ///< [out] Calling thread's output item (may be aliased to \p input)
        T                       &block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items (exclusive of the \p block_prefix_callback_op value)
        BlockPrefixCallbackOp   &block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a threadblock-wide prefix to be applied to all inputs.
    {
        InclusiveSum(input, output, block_aggregate);

        // Use the first warp to determine the threadblock prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate threadblock prefix into outputs
        Sum scan_op;
        T block_prefix = temp_storage.block_prefix;
        output = scan_op(block_prefix, output);
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

