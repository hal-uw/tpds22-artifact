/*
  sharedptr.h

  Convenience class for shared CPU/GPU allocations.
  Based on the X10 Runtime ideas described in Pai et al. in PACT 2012.
  Also see NVIDIA Hemi's array.h at <https://github.com/harrism/hemi>

  Copyright (C) 2014--2016, The University of Texas at Austin  

  Author: Sreepathi Pai  <sreepai@ices.utexas.edu>
*/

#pragma once
#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <assert.h>
#include "cutil_subset.h"

template<typename T>
class Shared {
  T **ptrs;
  bool *owner;
  bool *isCPU;
  int max_devices;
  size_t nmemb;

public:
  Shared() {
    nmemb = 0;
  }

  Shared(size_t nmemb) {
    this->nmemb = nmemb;
    max_devices = 2;
    ptrs = (T **) calloc(max_devices, sizeof(T *));
    owner = (bool *) calloc(max_devices, sizeof(bool));
    isCPU = (bool *) calloc(max_devices, sizeof(bool));

    isCPU[0] = true;

    for(int i = 0; i < max_devices; i++)
      owner[i] = true;
  }

  size_t size() const {
    return this->nmemb;
  }

  void alloc(size_t nmemb) {
    assert(this->nmemb == 0);

    this->nmemb = nmemb;

    max_devices = 2;
    ptrs = (T **) calloc(max_devices, sizeof(T *));
    owner = (bool *) calloc(max_devices, sizeof(bool));
    isCPU = (bool *) calloc(max_devices, sizeof(bool));

    isCPU[0] = true;

    for(int i = 0; i < max_devices; i++)
      owner[i] = true;   
  }

  void free()
  {
    for(int i = 0; i < max_devices; i++)
      free_device(i);
  }

  bool free_device(int device = 0)
  {
    assert(device < max_devices);

    if(!ptrs[device])
      return true;

    if(isCPU[device])
      ::free(ptrs[device]);
    else
      {
	if(hipFree(ptrs[device]) == hipSuccess)
	  ptrs[device] = NULL;
	else
	  return false;
      }    

    return true;
  }

  bool find_owner(int &o)
  {
    int i;
    for(i = 0; i < max_devices; i++)
      if(owner[i]) {
	o = i;
	break;
      }
    
    return i < max_devices;    
  }

  
  T *cpu_rd_ptr()
  {
    if(ptrs[0] == NULL)
      ptrs[0] = (T *) calloc(nmemb, sizeof(T));

    if(!owner[0])
      {
	int o;
	if(find_owner(o))
	  copy(o, 0);

	owner[0] = true;
      }

    return ptrs[0];
  }

  T *cpu_wr_ptr(bool overwrite = false)
  {
    if(ptrs[0] == NULL)
      ptrs[0] = (T *) calloc(nmemb, sizeof(T));

    if(!owner[0])
      {
	if(!overwrite)
	  {
	    int o;
	    if(find_owner(o))
	      copy(o, 0);
	  }

	owner[0] = true;
      }

    for(int i = 1; i < max_devices; i++)
      owner[i] = false;

    return ptrs[0];
  }

  T *gpu_rd_ptr(int device = 1) /* device >= 1 */
  {
    assert(device >= 1);

    if(ptrs[device] == NULL)
      CUDA_SAFE_CALL(hipMalloc(&ptrs[device], nmemb * sizeof(T)));

    if(!owner[device])
      {
	int o;
	if(find_owner(o))
	  copy(o, device);

	owner[device] = true;
      }

    return ptrs[device];
  }

  T *gpu_wr_ptr(bool overwrite = false, int device = 1)
  {
    assert(device >= 1);

    if(ptrs[device] == NULL) {
      CUDA_SAFE_CALL(hipMalloc(&ptrs[device], nmemb * sizeof(T)));
    }

    if(!owner[device])
      {
	if(!overwrite)
	  {
	    int o;
	    if(find_owner(o))
	      copy(o, device);
	  }

	owner[device] = true;
      }

    for(int i = 0; i < max_devices; i++)
      if(i != device)
	owner[i] = false;

    return ptrs[device];
  }

  T *zero_gpu(int device = 1) { 
    T *p = gpu_wr_ptr(true, device);
    CUDA_SAFE_CALL(hipMemset(p, 0, sizeof(T) * nmemb));
    return p;
  }

  void copy(int src, int dst)
  {
    if(!ptrs[src])
      return;

    assert(ptrs[dst]);

    if(isCPU[dst] && !isCPU[src]) {
      CUDA_SAFE_CALL(hipMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T), hipMemcpyDeviceToHost));
    } else if (!isCPU[dst] && !isCPU[src]) {
      CUDA_SAFE_CALL(hipMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T), hipMemcpyDeviceToDevice)); 
    } else if (!isCPU[dst] && isCPU[src]) {
      CUDA_SAFE_CALL(hipMemcpy(ptrs[dst], ptrs[src], nmemb * sizeof(T), hipMemcpyHostToDevice)); 
    } else
      abort(); // cpu-to-cpu not implemented
  
  }

  __device__ __host__
  T* ptr() {
#ifdef __CUDA_ARCH__
    return ptrs[1];   //TODO: this is invalid beyond one gpu device!
#else
    return ptrs[0];
#endif 
  }
};
  

