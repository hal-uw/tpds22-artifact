#pragma once

#include "hip/hip_runtime.h"
#include "util/util.h"

namespace mgpu {

class HipDevice;

class HipContext;
typedef intrusive_ptr<HipContext> ContextPtr;

////////////////////////////////////////////////////////////////////////////////
// Customizable allocator.

// HipAlloc is the interface class all allocator accesses. Users may derive
// this, implement custom allocators, and set it to the device with 
// HipDevice::SetAllocator.

class HipAlloc : public HipBase {
public:
	virtual hipError_t Malloc(size_t size, void** p) = 0;
	virtual bool Free(void* p) = 0;
	virtual void Clear() = 0;

	virtual ~HipAlloc() { }
	
	HipDevice& Device() { return _device; }
	
protected:
	HipAlloc(HipDevice& device) : _device(device) { }
	HipDevice& _device;
};

// A concrete class allocator that simply calls hipMalloc and hipFree.
class HipAllocSimple : public HipAlloc {
public:
	HipAllocSimple(HipDevice& device) : HipAlloc(device) { }

	virtual hipError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);
	virtual void Clear() { }
	virtual ~HipAllocSimple() { }
};

// A concrete class allocator that uses exponentially-spaced buckets and an LRU
// to reuse allocations. This is the default allocator. It is shared between
// all contexts on the device.
class HipAllocBuckets : public HipAlloc {
public:
	HipAllocBuckets(HipDevice& device);
	virtual ~HipAllocBuckets();

	virtual hipError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);
	virtual void Clear();

	size_t Allocated() const { return _allocated; }
	size_t Committed() const { return _committed; }
	size_t Capacity() const { return _capacity; }

	bool SanityCheck() const;

	void SetCapacity(size_t capacity, size_t maxObjectSize) {
		_capacity = capacity;
		_maxObjectSize = maxObjectSize;
		Clear();
	}

private:
	static const int NumBuckets = 84;
	static const size_t BucketSizes[NumBuckets];

	struct MemNode;
	typedef std::list<MemNode> MemList;
	typedef std::map<void*, MemList::iterator> AddressMap;
	typedef std::multimap<int, MemList::iterator> PriorityMap;

	struct MemNode {
		AddressMap::iterator address;
		PriorityMap::iterator priority;
		int bucket;
	};

	void Compact(size_t extra);
	void FreeNode(MemList::iterator memIt);
	int LocateBucket(size_t size) const;

	AddressMap _addressMap;
	PriorityMap _priorityMap;
	MemList _memLists[NumBuckets + 1];

	size_t _maxObjectSize, _capacity, _allocated, _committed;
	int _counter;
};

} // namespace mgpu
