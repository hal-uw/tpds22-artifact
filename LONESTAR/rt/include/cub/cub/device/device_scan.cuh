
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
 * hipcub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch/device_scan_dispatch.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within global memory. ![](device_scan.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * Given a sequence of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output sequence where each element is computed to be the reduction
 * of the elements occurring earlier in the input sequence.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceScan}
 *
 * \par Performance
 * \linear_performance{prefix scan}
 *
 * \par
 * The following chart illustrates DeviceScan::ExclusiveSum
 * performance across different CUDA architectures for \p int32 keys.
 * \plots_below
 *
 * \image html scan_int32.png
 *
 */
struct DeviceScan
{
    /******************************************************************//**
     * \name Exclusive scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide exclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * The following charts illustrate saturated exclusive sum performance across different
     * CUDA architectures for \p int32 and \p int64 items, respectively.
     *
     * \image html scan_int32.png
     * \image html scan_int64.png
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix sum
     * hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access input iterator type for reading scan input data \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access output iterator type for writing scan output data \iterator
     */
    template <
        typename        InputIterator,
        typename        OutputIterator>
    CUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveSum(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIterator  d_out,                              ///< [out] Pointer to the output sequence of data items
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        hipStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Scan data type
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        return DeviceScanDispatch<InputIterator, OutputIterator, Sum, T, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            T(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceScan::ExclusiveSum.
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix min-scan of an \p int device vector
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op
     * ...
     *
     * // Determine temporary device storage requirements for exclusive prefix scan
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // Allocate temporary storage for exclusive prefix scan
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix min-scan
     * hipcub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
     *
     * \endcode
     *
     * \tparam InputIterator    <b>[inferred]</b> Random-access input iterator type for reading scan input data \iterator
     * \tparam OutputIterator   <b>[inferred]</b> Random-access output iterator type for writing scan output data \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam Identity         <b>[inferred]</b> Type of the \p identity value used Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIterator,
        typename        OutputIterator,
        typename        ScanOp,
        typename        Identity>
    CUB_RUNTIME_FUNCTION
    static hipError_t ExclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIterator  d_out,                              ///< [out] Pointer to the output sequence of data items
        ScanOp          scan_op,                            ///< [in] Binary scan functor (e.g., an instance of hipcub::Sum, hipcub::Min, hipcub::Max, etc.)
        Identity        identity,                           ///< [in] Identity element
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        hipStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, ScanOp, Identity, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            identity,
            num_items,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceScan::ExclusiveSum.
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix sum
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out <-- [8, 14, 21, 26, 29, 29, 38]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access input iterator type for reading scan input data \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access output iterator type for writing scan output data \iterator
     */
    template <
        typename            InputIterator,
        typename            OutputIterator>
    CUB_RUNTIME_FUNCTION
    static hipError_t InclusiveSum(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator       d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIterator      d_out,                              ///< [out] Pointer to the output sequence of data items
        int                 num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        hipStream_t        stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, Sum, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide inclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceScan::ExclusiveSum.
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix min-scan of an \p int device vector.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix scan
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * hipcub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // Allocate temporary storage for inclusive prefix scan
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix min-scan
     * hipcub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // d_out <-- [8, 6, 6, 5, 3, 0, 0]
     *
     * \endcode
     *
     * \tparam InputIterator    <b>[inferred]</b> Random-access input iterator type for reading scan input data \iterator
     * \tparam OutputIterator   <b>[inferred]</b> Random-access output iterator type for writing scan output data \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIterator,
        typename        OutputIterator,
        typename        ScanOp>
    CUB_RUNTIME_FUNCTION
    static hipError_t InclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIterator  d_out,                              ///< [out] Pointer to the output sequence of data items
        ScanOp          scan_op,                            ///< [in] Binary scan functor (e.g., an instance of hipcub::Sum, hipcub::Min, hipcub::Max, etc.)
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        hipStream_t    stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, ScanOp, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_scan.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


