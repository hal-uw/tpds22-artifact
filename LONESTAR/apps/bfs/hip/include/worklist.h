/*
   worklist.h

   Implements Worklist classes. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#pragma once

#include "sharedptr.h"
#include <hipcub/hipcub.hpp>
#include "hiputil_subset.h"
#include "bmk2.h"
#include "instr.h"
#include "kernels/mergesort.h"
#include <stdio.h>
#include <stdlib.h>
 
#define SLOTS 1

static int zero = 0;

extern mgpu::ContextPtr mgc;

static
__global__ void reset_wl(volatile int *dindex) {
  *dindex = 0;
}

static
__global__ void init_wl(int size, int *dsize, volatile int *dindex) {
  *dsize = size;
  *dindex = 0;
  *(dindex + 1) = 0;
}

/*   int *dwl;
  int *dindex;
  int *dcounters;
  int currslot;
  int length;
*/
struct Worklist {
  int *dwl;
  int *dindex;
#ifdef SLOTS
  int *dcounters;
  int currslot;
#endif
  int length, index;

  int *wl;
  int *dnsize;

  int *dprio;

#ifdef COUNT_ATOMICS
  int *atomic_counter;
#endif 

#ifdef ATOMIC_DENSITY
  unsigned int *atomic_density;
#endif

  Shared<int> prio;
  bool f_will_write;
  
  Worklist(size_t nsize)
  {
#ifdef SLOTS
    currslot = 0;
#endif
    if(nsize == 0) {
      wl = NULL;
      dwl = NULL;
    } else {
      wl = (int *) calloc(nsize, sizeof(int));
      HIP_SAFE_CALL(hipMalloc(&dwl, nsize * sizeof(int)));
      HIP_SAFE_CALL(hipMalloc(&dnsize, 1 * sizeof(int)));
#ifdef SLOTS      
      HIP_SAFE_CALL(hipMalloc(&dcounters, 2 * sizeof(int)));
      dindex = &dcounters[currslot];
#else
      HIP_SAFE_CALL(hipMalloc(&dindex, 1 * sizeof(int)));
#endif 
      //HIP_SAFE_CALL(hipMalloc(&dindex, 2 * sizeof(int)));

      init_wl<<<1,1>>>(nsize, dnsize, dindex);

      //HIP_SAFE_CALL(hipMemcpy(dnsize, &nsize, 1 * sizeof(int), hipMemcpyHostToDevice));
      //HIP_SAFE_CALL(hipMemcpy((void *) dindex, &zero, 1 * sizeof(zero), hipMemcpyHostToDevice));

#ifdef COUNT_ATOMICS
      HIP_SAFE_CALL(hipMalloc(&atomic_counter, sizeof(int) * 1));
      HIP_SAFE_CALL(hipMemcpy((void *) atomic_counter, &zero, 1 * sizeof(zero), hipMemcpyHostToDevice));
#endif 

#ifdef ATOMIC_DENSITY
      HIP_SAFE_CALL(hipMalloc(&atomic_density, sizeof(unsigned int) * (32+1)));
      HIP_SAFE_CALL(hipMemset(atomic_density, 0, sizeof(unsigned int) * (32+1)));
#endif

      //HIP_SAFE_CALL(hipMalloc(&rcounter, 1 * sizeof(int)));
      //HIP_SAFE_CALL(hipMemcpy((void *) rcounter, &zero, 1 * sizeof(zero), hipMemcpyHostToDevice));

      prio.alloc(nsize);
      //prio.cpu_wr_ptr();
      dprio = prio.gpu_wr_ptr(true);
      length = nsize;
      f_will_write = false;
      index = 0;
    }
  }

  void free() {
    ::free(wl);
    HIP_SAFE_CALL(hipFree(dwl));
    HIP_SAFE_CALL(hipFree(dnsize));
#ifdef SLOTS
    HIP_SAFE_CALL(hipFree(dcounters));
#else
    HIP_SAFE_CALL(hipFree(dindex));
#endif

#ifdef COUNT_ATOMICS
    HIP_SAFE_CALL(hipFree(atomic_counter));
#endif

    prio.free();
  }
  
  void will_write() {
    f_will_write = true;
  }

  void sort() {
    MergesortKeys(dwl, nitems(), mgpu::less<int>(), *mgc); 
  }

  void sort_prio() {
    MergesortPairs(dprio, dwl, nitems(), mgpu::less<int>(), *mgc); 
  }

  void update_gpu(int nsize)
  {
#ifdef SLOTS
    int index[2] = {nsize, 0};
    currslot = 0;
    dindex = &dcounters[currslot];

    HIP_SAFE_CALL(hipMemcpy((void *) dcounters, &index, 2  * sizeof(nsize), hipMemcpyHostToDevice));
#else
    HIP_SAFE_CALL(hipMemcpy((void *) dindex, &nsize, 1  * sizeof(nsize), hipMemcpyHostToDevice));
#endif
    HIP_SAFE_CALL(hipMemcpy(dwl, wl, nsize  * sizeof(int), hipMemcpyHostToDevice));
  }

  void update_cpu()
  {
    int nsize = nitems();
    HIP_SAFE_CALL(hipMemcpy(wl, dwl, nsize  * sizeof(int), hipMemcpyDeviceToHost));
  }

  void display_items()
  {
    int nsize = nitems();
    HIP_SAFE_CALL(hipMemcpy(wl, dwl, nsize  * sizeof(int), hipMemcpyDeviceToHost));

    printf("WL: ");
    for(int i = 0; i < nsize; i++)
      printf("%d %d, ", i, wl[i]);

    printf("\n");
    return;
  }


  void save(const char *f, const unsigned iteration)
  {
    char n[255];
    int ret;

    ret = snprintf(n, 255, "%s%s-%05d-%s.wl", instr_trace_dir(), f, iteration, instr_uniqid());

    if(ret < 0 || ret >= 255) {
      fprintf(stderr, "Error creating filename for kernel '%s', iteration %d\n", f, iteration);
      exit(1);
    }

    int nsize = nitems();
    TRACE of = trace_open(n, "w");
    instr_write_array_gpu(n, of, sizeof(wl[0]), nsize, dwl, wl);
    trace_close(of);
    bmk2_log_collect("ggc/wlcontents", n);
    return;
  }

  void load(const char *f, const unsigned iteration)
  {
    char n[255];
    int ret;

    ret = snprintf(n, 255, "%s%s-%05d-%s.wl", instr_trace_dir(), f, iteration, instr_saved_uniqid());

    if(ret < 0 || ret >= 255) {
      fprintf(stderr, "Error creating filename for kernel '%s', iteration %d\n", f, iteration);
      exit(1);
    }

    TRACE of = trace_open(n, "r");
    int nsize = instr_read_array_gpu(n, of, sizeof(wl[0]), length, dwl, wl);
    HIP_SAFE_CALL(hipMemcpy((void *) dindex, &nsize, 1  * sizeof(nsize), hipMemcpyHostToDevice));
    trace_close(of);
    return;
  }


#ifdef SLOTS
  __device__ __host__ inline
  void reset_next_slot() const
  {
#ifdef __CUDA_ARCH__
    dcounters[1 ^ currslot] = 0;
#else
    reset_wl<<<1,1>>>(&dcounters[1 ^ currslot]);
#endif 
  }

  __device__ __host__ inline
  void set_slot(int slot)
  {
    currslot = slot;
    dindex = &dcounters[currslot];
  }

  __device__ __host__ inline
  void swap_slots()
  {
    currslot ^= 1;
    dindex = &dcounters[currslot];
  }
#endif /* SLOTS */

  __device__ __host__ inline
  void reset()
  {
#ifdef __CUDA_ARCH__
    *(volatile int *) dindex = 0;
    //atomicAdd(rcounter, 1);
#else
    //HIP_SAFE_CALL(hipMemcpy((void *) dindex, &zero, 1 * sizeof(zero), hipMemcpyHostToDevice));
    reset_wl<<<1,1>>>(dindex);
#endif 
  }

  __device__ __host__ inline
  int nitems()
  {
#ifdef __CUDA_ARCH__
    //return atomicAdd(dindex, 0);
    //return *dindex;
    return *((volatile int*) dindex);
#else
    //if(f_will_write)

    HIP_SAFE_CALL(hipMemcpy(&index, (void *) dindex, 1 * sizeof(index), hipMemcpyDeviceToHost));

    //f_will_write = 0;
    return index;
#endif     
  }

  __device__ 
  int push(int item)
  {
    int lindex = atomicAdd((int *) dindex, 1);
    assert(lindex <= *dnsize);

#ifdef ATOMIC_DENSITY
    int first, offset, total;
    warp_active_count(first, offset, total);

    if(offset == 0) {
      atomicAdd(&atomic_density[total], 1);
    }    
#endif

#ifdef COUNT_ATOMICS
    atomicAdd(atomic_counter, 1);
#endif     

    dwl[lindex] = item;
    return 1;
  }

  __device__ 
  int push_range(int nitems) const
  {
    int lindex = atomicAdd((int *) dindex, nitems);
    assert(lindex <= *dnsize);

#ifdef COUNT_ATOMICS
    atomicAdd(atomic_counter, 1);
#endif     

    return lindex;
  }

  __device__ 
  int push(int item, int prio)
  {
    int lindex = atomicAdd((int *) dindex, 1);
    assert(lindex <= *dnsize);

#ifdef COUNT_ATOMICS
    atomicAdd(atomic_counter, 1);
#endif     


#ifdef ATOMIC_DENSITY
    int first, offset, total;
    warp_active_count(first, offset, total);

    if(offset == 0) {
      atomicAdd(&atomic_density[total], 1);
    }    
#endif

    dwl[lindex] = item;
    dprio[lindex] = prio;

    return 1;
  }

  __device__ 
  int push_id(int id, int item)
  {
    assert(id <= *dnsize);
    dwl[id] = item;
    return 1;
  }

  __device__
  int setup_push_warp_one() {
    int first, total, offset, lindex = 0;

    warp_active_count(first, offset, total);
    
    if(offset == 0) {
      lindex = atomicAdd((int *) dindex, total);
      assert(lindex <= *dnsize);
#ifdef COUNT_ATOMICS
      atomicAdd(atomic_counter, 1);
#endif     

      // counting density makes no sense -- it is always 1
    }
    
    lindex = cub::ShuffleBroadcast(lindex, first);
    //lindex = cub::ShuffleIndex(lindex, first); // CUB > 1.3.1

    return lindex + offset;
  }

  __device__
  int setup_push_warp_one_za() {
    int first, total, offset, lindex = 0;

    // test function, not part of API

    warp_active_count_zero_active(first, offset, total);
    
    if(offset == 0) {
      lindex = atomicAdd((int *) dindex, total);
      assert(lindex <= *dnsize);
#ifdef COUNT_ATOMICS
      atomicAdd(atomic_counter, 1);
#endif     
    }
    
    lindex = cub::ShuffleBroadcast(lindex, first);
    //lindex = cub::ShuffleIndex(lindex, first); // CUB > 1.3.1

    return lindex + offset;
  }


  // must be warp uniform ... i.e. all threads in warp must be active
  template <typename T>
  __device__
  int setup_push_warp(typename T::TempStorage *ts, int nitems) {
    int total, offset, lindex;
    T(ts[threadIdx.x / 32]).ExclusiveSum(nitems, offset, total);
    
    if(threadIdx.x % 32 == 0) {
      lindex = atomicAdd((int *) dindex, total);
      assert(lindex <= *dnsize);
#ifdef COUNT_ATOMICS
      atomicAdd(atomic_counter, 1);
#endif     
    }
    
    lindex = cub::ShuffleBroadcast(lindex, 0);
    // lindex = cub::ShuffleIndex(lindex, 0); // CUB > 1.3.1

    return lindex + offset;
  }

  __device__ 
  int do_push(int start, int id, int item) const
  {
    assert(id <= *dnsize);
    dwl[start + id] = item;
    return 1;
  }

  __device__
  int pop(int &item) const
  {
    int lindex = atomicSub((int *) dindex, 1);
    if(lindex <= 0)
      {
	*dindex = 0;
	return 0;
      }

    item = dwl[lindex - 1];
    return 1;
  }
};

struct Worklist2: public Worklist {
  Worklist2(): Worklist(0) {}
  Worklist2(int nsize) : Worklist(nsize) {}

  template <typename T>
  __device__ __forceinline__
    int push_1item(int nitem, int item, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items = 0;
    int thread_data = nitem;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	if(debug) printf("t: %d\n", total_items);
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
#ifdef COUNT_ATOMICS
	atomicAdd(atomic_counter, 1);
#endif     
      }

    __syncthreads();

    if(nitem == 1)
      {
	if(queue_index + thread_data >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d\n", queue_index, thread_data, *dnsize);
	    return 0;
	  }

	//dwl[queue_index + thread_data] = item;
	cub::ThreadStore<cub::STORE_CG>(dwl + queue_index + thread_data, item);
      }

    return total_items;
  }

  template <typename T>
  __device__ __forceinline__
    int push_1item(int nitem, int item, int prio, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items = 0;
    int thread_data = nitem;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	if(debug) printf("t: %d\n", total_items);
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
#ifdef COUNT_ATOMICS
	atomicAdd(atomic_counter, 1);
#endif     
      }

    __syncthreads();

    if(nitem == 1)
      {
	if(queue_index + thread_data >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d\n", queue_index, thread_data, *dnsize);
	    return 0;
	  }

	//dwl[queue_index + thread_data] = item;
	cub::ThreadStore<cub::STORE_CG>(dwl + queue_index + thread_data, item);
	cub::ThreadStore<cub::STORE_CG>(dprio + queue_index + thread_data, prio);
      }

    return total_items;
  }

  template <typename T>
  __device__ __forceinline__
  int push_nitems(int n_items, int *items, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items;

    int thread_data = n_items;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
#ifdef COUNT_ATOMICS
	atomicAdd(atomic_counter, 1);
#endif     
      }

    __syncthreads();

    
    for(int i = 0; i < n_items; i++)
      {
	//printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
	if(queue_index + thread_data + i >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d %d\n", queue_index, thread_data, i, *dnsize);
	    return 0;
	  }

	dwl[queue_index + thread_data + i] = items[i];
      }

    return total_items;
  }

  __device__ 
  int pop_id(int id, int &item) const
  {
    if(id < *dindex)
      {
	item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	//item = dwl[id];
	return 1;
      }
    
    return 0;
  }  

  __device__ 
  int pop_id_len(int id, int len, int &item) const
  {
    if(id < len)
      {
	item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	//item = dwl[id];
	return 1;
      }
    
    return 0;
  }  

};


struct WorklistT: public Worklist2 {
  cudaTextureObject_t tx;

  WorklistT(): Worklist2() {}

  WorklistT(size_t nsize) : Worklist2(nsize) {
    // from here: http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = dwl;
    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = length*sizeof(int);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
    HIP_SAFE_CALL(cudaCreateTextureObject(&tx, &resDesc, &texDesc, NULL));
  } 

  void free() {
    HIP_SAFE_CALL(cudaDestroyTextureObject(tx));
    Worklist2::free();
  }

  __device__ 
  int pop_id(int id, int &item)
  {
    if(id < *dindex)
      {
	item = tex1Dfetch<int>(tx, id);
	//item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	return 1;
      }
    
    return 0;
  }  

  __device__ 
  int pop_id_len(int id, int len, int &item)
  {
    if(id < len)
      {
	item = tex1Dfetch<int>(tx, id);
	//item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	return 1;
      }
    
    return 0;
  }  
};

struct Worklist2Light {
  int *dwl;
  int *dindex;
  int *dcounters;
  int currslot;
  int length;

  __device__
  void fromWL2(Worklist2 wl) {
    dwl = wl.dwl;
    dindex = wl.dindex;
    dcounters = wl.dcounters;
    currslot = wl.currslot;
    length = *wl.dnsize;
  }

  __device__ __host__ inline
  int nitems()
  {
#ifdef __CUDA_ARCH__
    //return atomicAdd(dindex, 0);
    //return *dindex;
    return *((volatile int*) dindex);
#else
    assert(false);
    return 0;
    //if(f_will_write)

    //HIP_SAFE_CALL(hipMemcpy(&index, (void *) dindex, 1 * sizeof(index), hipMemcpyDeviceToHost));

    //f_will_write = 0;
    //return index;
#endif     
  }

#ifdef SLOTS
  __device__ __host__ inline
  void swap_slots()
  {
    currslot ^= 1;
    dindex = &dcounters[currslot];
  }

  __device__ __host__ inline
  void set_slot(int slot)
  {
    currslot = slot;
    dindex = &dcounters[currslot];
  }
#endif /* SLOTS */


#ifdef SLOTS
  __device__ __host__ inline
  void reset_next_slot() const
  {
#ifdef __CUDA_ARCH__
    dcounters[1 ^ currslot] = 0;
#else
    reset_wl<<<1,1>>>(&dcounters[1 ^ currslot]);
#endif 
  }
#endif

  __device__ 
  int do_push(int start, int id, int item)
  {
    assert(id <= length);
    dwl[start + id] = item;
    return 1;
  }

  __device__ 
  int push_range(int nitems) const
  {
    int lindex = atomicAdd((int *) dindex, nitems);
    assert(lindex <= length);

#ifdef COUNT_ATOMICS
    //atomicAdd(atomic_counter, 1);
#endif     

    return lindex;
  }

  __device__
  int setup_push_warp_one() {
    int first, total, offset, lindex = 0;

    warp_active_count(first, offset, total);
    
    if(offset == 0) {
      lindex = atomicAdd((int *) dindex, total);
      assert(lindex <= length);
#ifdef COUNT_ATOMICS
      atomicAdd(atomic_counter, 1);
#endif     

      // counting density makes no sense -- it is always 1
    }
    
    lindex = cub::ShuffleBroadcast(lindex, first);
    //lindex = cub::ShuffleIndex(lindex, first); // CUB > 1.3.1

    return lindex + offset;
  }

  __device__ 
  int pop_id(int id, int &item)
  {
    if(id < *dindex)
      {
    	item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
    	//item = dwl[id];
    	return 1;
      }
    
    return 0;
  }  

};


#ifdef COUNT_ATOMICS 
static 
__device__
__host__
int get_atomic_count(Worklist wl) 
{
#ifdef __CUDA_ARCH__
  return *wl.atomic_counter;
#else
  int count = 0;
  HIP_SAFE_CALL(hipMemcpy(&count, wl.atomic_counter, sizeof(int) * 1, hipMemcpyDeviceToHost));
  return count;
#endif 
}
#endif

#ifdef ATOMIC_DENSITY 
static 
__device__
__host__
void print_atomic_density(const char *name, Worklist wl) 
{
#ifdef __CUDA_ARCH__
  assert(false);
#else
  unsigned count[32+1];
  HIP_SAFE_CALL(hipMemcpy(&count, wl.atomic_density, sizeof(unsigned int) * (32 + 1), hipMemcpyDeviceToHost));

  for(int i = 0; i < 32 + 1; i++) {
    fprintf(stderr, "INSTR atomic_density_%s_%d %u\n", name, i, count[i]);
  }
#endif 
}

static 
__device__
__host__
void print_atomic_density_2(const char *kernel, const int kcount, const char *name, Worklist wl) 
{
#ifdef __CUDA_ARCH__
  assert(false);
#else
  unsigned count[32+1];
  HIP_SAFE_CALL(hipMemcpy(&count, wl.atomic_density, sizeof(unsigned int) * (32 + 1), hipMemcpyDeviceToHost));
  HIP_SAFE_CALL(hipMemset(wl.atomic_density, 0, sizeof(unsigned int) * (32+1)));

  for(int i = 0; i < 32 + 1; i++) {
    fprintf(stderr, "INSTR out_atomic_density:%s:%s:%d_%d %u\n", kernel, name, kcount, i, count[i]);
  }
#endif 
}



#endif
