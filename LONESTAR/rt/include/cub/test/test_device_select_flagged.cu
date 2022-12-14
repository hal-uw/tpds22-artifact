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

/******************************************************************************
 * Test of DeviceSelect::Flagged and DevicePartition::Flagged utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/iterator/reverse_iterator.h>

#include "test_util.h"

using namespace hipcub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
CachingDeviceAllocator  g_allocator(true);



//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to select flagged entrypoint
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
CUB_RUNTIME_FUNCTION __forceinline__
hipError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    Int2Type<false>             partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    hipStream_t                stream,
    bool                        debug_synchronous)
{
    hipError_t error = hipSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to partition flagged entrypoint
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
CUB_RUNTIME_FUNCTION __forceinline__
hipError_t Dispatch(
    Int2Type<CUB>               dispatch_to,
    Int2Type<true>              partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    hipStream_t                stream,
    bool                        debug_synchronous)
{
    hipError_t error = hipSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, stream, debug_synchronous);
    }
    return error;
}



//---------------------------------------------------------------------
// Dispatch to different Thrust entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to select flagged entrypoint
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __forceinline__
hipError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    Int2Type<false>             partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    hipStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type    T;
    typedef typename std::iterator_traits<FlagIterator>::value_type     Flag;


    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::device_ptr<T>       d_out_wrapper_end;

        thrust::device_ptr<T>       d_in_wrapper(d_in);
        thrust::device_ptr<T>       d_out_wrapper(d_out);
        thrust::device_ptr<Flag>    d_flags_wrapper(d_flags);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_wrapper_end = thrust::copy_if(d_in_wrapper, d_in_wrapper + num_items, d_flags_wrapper, d_out_wrapper, Cast<bool>());
        }

        Offset num_selected = d_out_wrapper_end - d_out_wrapper;
        CubDebugExit(hipMemcpy(d_num_selected, &num_selected, sizeof(Offset), hipMemcpyHostToDevice));
    }

    return hipSuccess;
}


/**
 * Dispatch to partition entrypoint
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset>
__host__ __forceinline__
hipError_t Dispatch(
    Int2Type<THRUST>            dispatch_to,
    Int2Type<true>              partition,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    hipStream_t                stream,
    bool                        debug_synchronous)
{
    typedef typename std::iterator_traits<InputIterator>::value_type    T;
    typedef typename std::iterator_traits<FlagIterator>::value_type     Flag;

    typedef thrust::reverse_iterator<thrust::device_ptr<T> > ReverseOutputIterator;

    if (d_temp_storage == 0)
    {
        temp_storage_bytes = 1;
    }
    else
    {
        thrust::pair<thrust::device_ptr<T>, ReverseOutputIterator> d_out_wrapper_end;

        thrust::device_ptr<T>       d_in_wrapper(d_in);
        thrust::device_ptr<T>       d_out_wrapper(d_out);
        thrust::device_ptr<Flag>    d_flags_wrapper(d_flags);

        ReverseOutputIterator d_out_unselected(d_out_wrapper + num_items);

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            d_out_wrapper_end = thrust::partition_copy(
                d_in_wrapper,
                d_in_wrapper + num_items,
                d_flags_wrapper,
                d_out_wrapper,
                d_out_unselected,
                Cast<bool>());
        }

        Offset num_selected = d_out_wrapper_end.first - d_out_wrapper;
        CubDebugExit(hipMemcpy(d_num_selected, &num_selected, sizeof(Offset), hipMemcpyHostToDevice));
    }

    return hipSuccess;
}



//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceSelect
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset, typename PartitionTag>
__global__ void CnpDispatchKernel(
    PartitionTag                partition_tag,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    bool                        debug_synchronous)
{

#ifndef CUB_CDP
    *d_cdp_error = hipErrorNotSupported;
#else
    *d_cdp_error = Dispatch(Int2Type<CUB>(), partition_tag, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, 0, debug_synchronous);
    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}


/**
 * Dispatch to CDP kernel
 */
template <typename InputIterator, typename FlagIterator, typename OutputIterator, typename NumSelectedIterator, typename Offset, typename PartitionTag>
hipError_t Dispatch(
    Int2Type<CDP>               dispatch_to,
    PartitionTag                partition_tag,
    int                         timing_timing_iterations,
    size_t                      *d_temp_storage_bytes,
    hipError_t                 *d_cdp_error,

    void                        *d_temp_storage,
    size_t                      &temp_storage_bytes,
    InputIterator               d_in,
    FlagIterator                d_flags,
    OutputIterator              d_out,
    NumSelectedIterator         d_num_selected,
    Offset                      num_items,
    hipStream_t                stream,
    bool                        debug_synchronous)
{
    // Invoke kernel to invoke device-side dispatch
    hipLaunchKernelGGL(CnpDispatchKernel, dim3(1), dim3(1), 0, 0, partition_tag, timing_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, debug_synchronous);

    // Copy out temp_storage_bytes
    CubDebugExit(hipMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t) * 1, hipMemcpyDeviceToHost));

    // Copy out error
    hipError_t retval;
    CubDebugExit(hipMemcpy(&retval, d_cdp_error, sizeof(hipError_t) * 1, hipMemcpyDeviceToHost));
    return retval;
}



//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem
 */
template <typename T>
void Initialize(
    T           *h_in,
    int         num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        // Initialize each item to a randomly selected value from [0..126]
        unsigned int value;
        RandomBits(value, 0, 0, 7);
        if (value == 127)
            value = 126;
        InitValue(INTEGER_SEED, h_in[i], value);
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve selection problem (select if less than compare)
 */
template <
    typename        InputIterator,
    typename        Flag,
    typename        T>
int Solve(
    InputIterator   h_in,
    Flag            *h_flags,
    T               *h_reference,
    T               compare,
    int             num_items)
{
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i)
    {
        if (h_in[i] < compare)
        {
            h_flags[i] = 1;
            h_reference[num_selected] = h_in[i];
            num_selected++;
        }
        else
        {
            h_flags[i] = 0;
            h_reference[num_items - (i - num_selected) - 1] = h_in[i];
        }
    }

    return num_selected;
}



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    bool                PARTITION,
    typename            DeviceInputIterator,
    typename            DeviceFlagIterator,
    typename            T>
void Test(
    DeviceInputIterator d_in,
    DeviceFlagIterator  d_flags,
    T                   *h_reference,
    int                 num_selected,
    int                 num_items,
    char*               type_string)
{
    typedef typename std::iterator_traits<DeviceFlagIterator>::value_type Flag;

    // Allocate device output array and num selected
    T       *d_out            = NULL;
    int     *d_num_selected   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected, sizeof(int)));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    hipError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(hipError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, 0, true));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    CubDebugExit(hipMemset(d_out, 0, sizeof(T) * num_items));
    CubDebugExit(hipMemset(d_num_selected, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, 0, true));

    // Check for correctness (and display results, if specified)
    int compare1 = (PARTITION) ?
        CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose) :
        CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s ", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(&num_selected, d_num_selected, 1, true, g_verbose);
    printf("\t Count %s ", compare2 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<PARTITION>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected, num_items, 0, false));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float grate = float(num_items) / avg_millis / 1000.0 / 1000.0;
        int output_items = (PARTITION) ? num_items : num_selected;
        float gbandwidth = float(((num_items + output_items) * sizeof(T)) + (num_items * sizeof(Flag))) / avg_millis / 1000.0 / 1000.0;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s", avg_millis, grate, gbandwidth);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_num_selected) CubDebugExit(g_allocator.DeviceFree(d_num_selected));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare1 | compare2);
}


/**
 * Test DeviceSelect on pointer type
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void TestPointer(
    int             num_items,
    float           select_ratio,
    char*           type_string)
{
    typedef char Flag;

    // Allocate host arrays
    T       *h_in        = new T[num_items];
    T       *h_reference = new T[num_items];
    Flag    *h_flags     = new Flag[num_items];

    // Initialize input
    Initialize(h_in, num_items);

    // Select a comparison value that is select_ratio through the space of [0,127]
    T compare;
    if (select_ratio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (select_ratio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * select_ratio)));

    // Set flags and solve
    int num_selected = Solve(h_in, h_flags, h_reference, compare, num_items);

    printf("\nPointer %s cub::%s::Flagged %d items, %d selected (select ratio %.3f), %s %d-byte elements\n",
        (PARTITION) ? "DevicePartition" : "DeviceSelect",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_selected, float(num_selected) / num_items, type_string, (int) sizeof(T));
    fflush(stdout);

    // Allocate problem device arrays
    T       *d_in = NULL;
    Flag    *d_flags = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(Flag) * num_items));

    // Initialize device input
    CubDebugExit(hipMemcpy(d_in, h_in, sizeof(T) * num_items, hipMemcpyHostToDevice));
    CubDebugExit(hipMemcpy(d_flags, h_flags, sizeof(Flag) * num_items, hipMemcpyHostToDevice));

    // Run Test
    Test<BACKEND, PARTITION>(d_in, d_flags, h_reference, num_selected, num_items, type_string);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_flags) CubDebugExit(g_allocator.DeviceFree(d_flags));
}


/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    char*           type_string,
    Int2Type<true>  is_number)
{
    typedef char Flag;

    // Use a counting iterator as the input and a constant iterator as flags
    CountingInputIterator<T, int> h_in(0);
    ConstantInputIterator<Flag, int> h_flags(1);

    // Allocate host arrays
    T*  h_reference = new T[num_items];

    // Initialize solution
    int num_selected = num_items;
    for (int i = 0; i < num_items; ++i)
        h_reference[i] = h_in[i];

    printf("\nIterator %s hipcub::DeviceSelect::Flagged %d items, %d selected, %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : (BACKEND == THRUST) ? "Thrust" : "CUB",
        num_items, num_selected, type_string, (int) sizeof(T));
    fflush(stdout);

    // Run Test
    Test<BACKEND, PARTITION>(h_in, h_flags, h_reference, num_selected, num_items, type_string);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    char*           type_string,
    Int2Type<false> is_number)
{}


/**
 * Test different selection ratios
 */
template <
    Backend         BACKEND,
    bool            PARTITION,
    typename        T>
void Test(
    int             num_items,
    char*           type_string)
{
    for (float select_ratio = 0; select_ratio <= 1.0; select_ratio += 0.2)
    {
        TestPointer<BACKEND, PARTITION, T>(num_items, select_ratio, type_string);
    }

    // Iterator test always keeps all items
    TestIterator<BACKEND, PARTITION, T>(num_items, type_string, Int2Type<Traits<T>::CATEGORY != NOT_A_NUMBER>());
}

/**
 * Test select vs. partition
 */
template <
    Backend         BACKEND,
    typename        T>
void TestMethod(
    int             num_items,
    char*           type_string)
{
    Test<BACKEND, false, T>(num_items, type_string);
    Test<BACKEND, true, T>(num_items, type_string);
}


/**
 * Test different dispatch
 */
template <
    typename        T>
void TestOp(
    int             num_items,
    char*           type_string)
{
    TestMethod<CUB, T>(num_items, type_string);
#ifdef CUB_CDP
    TestMethod<CDP, T>(num_items, type_string);
#endif
}


/**
 * Test different input sizes
 */
template <typename T>
void Test(
    int             num_items,
    char*           type_string)
{
    if (num_items < 0)
    {
        TestOp<T>(1,        type_string);
        TestOp<T>(100,      type_string);
        TestOp<T>(10000,    type_string);
        TestOp<T>(1000000,  type_string);
    }
    else
    {
        TestOp<T>(num_items, type_string);
    }
}


/**
 * Test select/partition on pointer types
 */
template <typename T>
void ComparePointer(
    int             num_items,
    float           select_ratio,
    char*           type_string)
{
    printf("-- Select ----------------------------\n");
    TestPointer<CUB, false, T>(num_items, select_ratio, type_string);
    TestPointer<THRUST, false, T>(num_items, select_ratio, type_string);

    printf("-- Partition ----------------------------\n");
    TestPointer<CUB, true, T>(num_items, select_ratio, type_string);
    TestPointer<THRUST, true, T>(num_items, select_ratio, type_string);
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = -1;
    float select_ratio      = 0.5;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("ratio", select_ratio);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--ratio=<selection ratio, default 0.5>] "
            "[--repeat=<repetitions of entire test suite>] "
            "[--v] "
            "[--cdp] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    printf("\n");

    // Get device ordinal
    int device_ordinal;
    CubDebugExit(hipGetDevice(&device_ordinal));

    // Get device SM version
    int sm_version;
    CubDebugExit(SmVersion(sm_version, device_ordinal));

#ifdef QUICK_TEST

    // Compile/run quick tests
    if (num_items < 0) num_items = 32000000;

    ComparePointer<char>(       num_items * ((sm_version <= 130) ? 1 : 4),  select_ratio, CUB_TYPE_STRING(char));
    ComparePointer<short>(      num_items * ((sm_version <= 130) ? 1 : 2),  select_ratio, CUB_TYPE_STRING(short));
    ComparePointer<int>(        num_items,                                  select_ratio, CUB_TYPE_STRING(int));
    ComparePointer<long long>(  num_items / 2,                              select_ratio, CUB_TYPE_STRING(long long));
    ComparePointer<TestFoo>(    num_items / 4,                              select_ratio, CUB_TYPE_STRING(TestFoo));

#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input types
        Test<unsigned char>(num_items, CUB_TYPE_STRING(unsigned char));
        Test<unsigned short>(num_items, CUB_TYPE_STRING(unsigned short));
        Test<unsigned int>(num_items, CUB_TYPE_STRING(unsigned int));
        Test<unsigned long long>(num_items, CUB_TYPE_STRING(unsigned long long));

        Test<uchar2>(num_items, CUB_TYPE_STRING(uchar2));
        Test<ushort2>(num_items, CUB_TYPE_STRING(ushort2));
        Test<uint2>(num_items, CUB_TYPE_STRING(uint2));
        Test<ulonglong2>(num_items, CUB_TYPE_STRING(ulonglong2));

        Test<uchar4>(num_items, CUB_TYPE_STRING(uchar4));
        Test<ushort4>(num_items, CUB_TYPE_STRING(ushort4));
        Test<uint4>(num_items, CUB_TYPE_STRING(uint4));
        Test<ulonglong4>(num_items, CUB_TYPE_STRING(ulonglong4));

        Test<TestFoo>(num_items, CUB_TYPE_STRING(TestFoo));
        Test<TestBar>(num_items, CUB_TYPE_STRING(TestBar));
    }

#endif

    return 0;
}



