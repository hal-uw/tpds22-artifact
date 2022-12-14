
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

#include "dispatch/device_radix_sort_dispatch.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across a sequence of data items residing within global memory. ![](sorting_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * The [<em>radix sorting method</em>](http://en.wikipedia.org/wiki/Radix_sort) arranges
 * items into ascending order.  It relies upon a positional representation for
 * keys, i.e., each key is comprised of an ordered sequence of symbols (e.g., digits,
 * characters, etc.) specified from least-significant to most-significant.  For a
 * given input sequence of keys and a set of rules specifying a total ordering
 * of the symbolic alphabet, the radix sorting method produces a lexicographic
 * ordering of those keys.
 *
 * \par
 * DeviceRadixSort can sort all of the built-in C++ numeric primitive types, e.g.:
 * <tt>unsigned char</tt>, \p int, \p double, etc.  Although the direct radix sorting
 * method can only be applied to unsigned integral types, BlockRadixSort
 * is able to sort signed and floating-point types via simple bit-wise transformations
 * that ensure lexicographic key ordering.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceRadixSort}
 *
 * \par Performance
 * \linear_performance{radix sort} The following chart illustrates DeviceRadixSort::SortKeys
 * performance across different CUDA architectures for uniform-random \p uint32 keys.
 * \plots_below
 *
 * \image html lsb_radix_sort_int32_keys.png
 *
 */
struct DeviceRadixSort
{
    /**
     * \brief Sorts key-value pairs into ascending order.
     *
     * \par
     * - The sorting operation requires a pair of key buffers and a pair of value
     *   buffers.  Each pair is wrapped in a DoubleBuffer structure whose member
     *   DoubleBuffer::Current() references the active buffer.  The currently-active
     *   buffer may be changed by the sorting operation.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * The following charts illustrate saturated sorting performance across different
     * CUDA architectures for uniform-random <tt>uint32,uint32</tt> and
     * <tt>uint64,uint64</tt> pairs, respectively.
     *
     * \image html lsb_radix_sort_int32_pairs.png
     * \image html lsb_radix_sort_int64_pairs.png
     *
     * \par Snippet
     * The code snippet below illustrates the sorting of a device vector of \p int keys
     * with associated vector of \p int values.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for sorting data
     * int  num_items;          // e.g., 7
     * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_key_alt_buf;     // e.g., [        ...        ]
     * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
     * int  *d_value_alt_buf;   // e.g., [        ...        ]
     * ...
     *
     * // Create a set of DoubleBuffers to wrap pairs of device pointers
     * hipcub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     * hipcub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
     *
     * // Allocate temporary storage
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
     *
     * // d_keys.Current()      <-- [0, 3, 5, 6, 7, 8, 9]
     * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     * \tparam Value    <b>[inferred]</b> Value type
     */
    template <
        typename            Key,
        typename            Value>
    CUB_RUNTIME_FUNCTION
    static hipError_t SortPairs(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Reference to the double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value> &d_values,                              ///< [in,out] Double-buffer of values whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceRadixSortDispatch<false, Key, Value, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Sorts key-value pairs into descending order.
     *
     * \par
     * - The sorting operation requires a pair of key buffers and a pair of value
     *   buffers.  Each pair is wrapped in a DoubleBuffer structure whose member
     *   DoubleBuffer::Current() references the active buffer.  The currently-active
     *   buffer may be changed by the sorting operation.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is similar to DeviceRadixSort::SortPairs.
     *
     * \par Snippet
     * The code snippet below illustrates the sorting of a device vector of \p int keys
     * with associated vector of \p int values.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for sorting data
     * int  num_items;          // e.g., 7
     * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_key_alt_buf;     // e.g., [        ...        ]
     * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
     * int  *d_value_alt_buf;   // e.g., [        ...        ]
     * ...
     *
     * // Create a set of DoubleBuffers to wrap pairs of device pointers
     * hipcub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     * hipcub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
     *
     * // Allocate temporary storage
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
     *
     * // d_keys.Current()      <-- [9, 8, 7, 6, 5, 3, 0]
     * // d_values.Current()    <-- [6, 0, 2, 1, 3, 4, 5]
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     * \tparam Value    <b>[inferred]</b> Value type
     */
    template <
        typename            Key,
        typename            Value>
    CUB_RUNTIME_FUNCTION
    static hipError_t SortPairsDescending(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Reference to the double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<Value> &d_values,                              ///< [in,out] Double-buffer of values whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceRadixSortDispatch<true, Key, Value, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Sorts keys into ascending order
     *
     * \par
     * - The sorting operation requires a pair of key buffers.  The pair is
     *   wrapped in a DoubleBuffer structure whose member DoubleBuffer::Current()
     *   references the active buffer.  The currently-active buffer may be changed
     *   by the sorting operation.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * The following charts illustrate saturated sorting performance across different
     * CUDA architectures for uniform-random \p uint32 and \p uint64 keys, respectively.
     *
     * \image html lsb_radix_sort_int32_keys.png
     * \image html lsb_radix_sort_int64_keys.png
     *
     * \par Snippet
     * The code snippet below illustrates the sorting of a device vector of \p int keys.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for sorting data
     * int  num_items;          // e.g., 7
     * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_key_alt_buf;     // e.g., [        ...        ]
     * ...
     *
     * // Create a DoubleBuffer to wrap the pair of device pointers
     * hipcub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Allocate temporary storage
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * hipcub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // d_keys.Current()      <-- [0, 3, 5, 6, 7, 8, 9]
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     */
    template <typename Key>
    CUB_RUNTIME_FUNCTION
    static hipError_t SortKeys(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Reference to the double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Null value type
        DoubleBuffer<NullType> d_values;

        return DeviceRadixSortDispatch<false, Key, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Sorts keys into ascending order
     *
     * \par
     * - The sorting operation requires a pair of key buffers.  The pair is
     *   wrapped in a DoubleBuffer structure whose member DoubleBuffer::Current()
     *   references the active buffer.  The currently-active buffer may be changed
     *   by the sorting operation.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is similar to DeviceRadixSort::SortKeys.
     *
     * \par Snippet
     * The code snippet below illustrates the sorting of a device vector of \p int keys.
     * \par
     * \code
     * #include <hipcub/hipcub.hpp>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for sorting data
     * int  num_items;          // e.g., 7
     * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_key_alt_buf;     // e.g., [        ...        ]
     * ...
     *
     * // Create a DoubleBuffer to wrap the pair of device pointers
     * hipcub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // Allocate temporary storage
     * hipMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sorting operation
     * hipcub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys, num_items);
     *
     * // d_keys.Current()      <-- [9, 8, 7, 6, 5, 3, 0]
     *
     * \endcode
     *
     * \tparam Key      <b>[inferred]</b> Key type
     */
    template <typename Key>
    CUB_RUNTIME_FUNCTION
    static hipError_t SortKeysDescending(
        void                *d_temp_storage,                        ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<Key>   &d_keys,                                ///< [in,out] Reference to the double-buffer of keys whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        int                 num_items,                              ///< [in] Number of items to reduce
        int                 begin_bit           = 0,                ///< [in] <b>[optional]</b> The first (least-significant) bit index needed for key comparison
        int                 end_bit             = sizeof(Key) * 8,  ///< [in] <b>[optional]</b> The past-the-end (most-significant) bit index needed for key comparison
        hipStream_t        stream              = 0,                ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous   = false)            ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Null value type
        DoubleBuffer<NullType> d_values;

        return DeviceRadixSortDispatch<true, Key, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_values,
            num_items,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous);
    }

};

/**
 * \example example_device_radix_sort.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


