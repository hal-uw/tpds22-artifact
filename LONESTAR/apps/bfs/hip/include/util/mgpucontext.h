#pragma once

#include "hip/hip_runtime.h"
#include "util/util.h"
#include "util/format.h"
#include "mgpualloc.h"

namespace mgpu {


#ifdef _DEBUG
#define MGPU_SYNC_CHECK(s) {												\
	hipError_t error = hipDeviceSynchronize();							\
	if(hipSuccess != error) {												\
		printf("HIP ERROR %d %s\n%s:%d.\n%s\n",							\
			error, hipGetErrorString(error), __FILE__, __LINE__, s);		\
		exit(0);															\
	}																		\
}
#else
#define MGPU_SYNC_CHECK(s)
#endif

template<typename T>
void copyDtoH(T* dest, const T* source, int count) {
	hipMemcpy(dest, source, sizeof(T) * count, hipMemcpyDeviceToHost);
}
template<typename T>
void copyDtoD(T* dest, const T* source, int count, hipStream_t stream = 0) {
	hipMemcpyAsync(dest, source, sizeof(T) * count, hipMemcpyDeviceToDevice,
		stream);
}
template<typename T>
void copyDtoH(std::vector<T>& dest, const T* source, int count) {
	dest.resize(count);
	if(count) 
		copyDtoH(&dest[0], source, count);
}

template<typename T>
void copyHtoD(T* dest, const T* source, int count) {
	hipMemcpy(dest, source, sizeof(T) * count, hipMemcpyHostToDevice);
}
template<typename T>
void copyHtoD(T* dest, const std::vector<T>& source) {
	if(source.size())
		copyHtoD(dest, &source[0], source.size());
}


////////////////////////////////////////////////////////////////////////////////

class HipContext;
typedef intrusive_ptr<HipContext> ContextPtr;
typedef intrusive_ptr<HipAlloc> AllocPtr;

class HipException : public std::exception {
public:
	hipError_t error;

	HipException() throw() { }
	HipException(hipError_t e) throw() : error(e) { }
	HipException(const HipException& e) throw() : error(e.error) { }

	virtual const char* what() const throw() {
		return "HIP runtime error";
	}
};


////////////////////////////////////////////////////////////////////////////////
// HipEvent and HipTimer. 
// Exception-safe wrappers around hipEvent_t.

class HipEvent : public noncopyable {
public:
	HipEvent() { 
		hipEventCreate(&_event);
	}
	explicit HipEvent(int flags) {
		hipEventCreateWithFlags(&_event, flags);
	}
	~HipEvent() {
		hipEventDestroy(_event);
	}
	operator hipEvent_t() { return _event; }
	void Swap(HipEvent& rhs) {
		std::swap(_event, rhs._event);
	}
private:
	hipEvent_t _event;
};

class HipTimer : noncopyable {
	HipEvent start, end;
public:
	void Start();
	double Split();
	double Throughput(int count, int numIterations);
};


////////////////////////////////////////////////////////////////////////////////

struct DeviceGroup;

class HipDevice : public noncopyable {
	friend struct DeviceGroup;
public:
	static int DeviceCount();
	static HipDevice& ByOrdinal(int ordinal);
	static HipDevice& Selected();

	// Device properties.
	const hipDeviceProp_t & Prop() const { return _prop; }
	int Ordinal() const { return _ordinal; }
	int NumSMs() const { return _prop.multiProcessorCount; }
	int ArchVersion() const { return 100 * _prop.major + 10 * _prop.minor; }

	// LaunchBox properties.
	int PTXVersion() const { return _ptxVersion; }

	std::string DeviceString() const;

	// Set this device as the active device on the thread.
	void SetActive();

private:
	HipDevice() { }		// hide the destructor.
	int _ordinal;
	int _ptxVersion;
	hipDeviceProp_t _prop;
};

////////////////////////////////////////////////////////////////////////////////
// HipDeviceMem
// Exception-safe HIP device memory container. Use the MGPU_MEM(T) macro for
// the type of the reference-counting container.
// HipDeviceMem AddRefs the allocator that returned the memory, releasing the
// pointer when the object is destroyed.

template<typename T>
class HipDeviceMem : public HipBase {
	friend class HipMemSupport;
public:
	~HipDeviceMem();

	const T* get() const { return _p; }
	T* get() { return _p; }

	operator const T*() const { return get(); }
	operator T*() { return get(); }

	// Size is in units of T, not bytes.
	size_t Size() const { return _size; }

	// Copy from this to the argument array.
	hipError_t ToDevice(T* data, size_t count) const;
	hipError_t ToDevice(size_t srcOffest, size_t bytes, void* data) const;
	hipError_t ToHost(T* data, size_t count) const;
	hipError_t ToHost(std::vector<T>& data) const;
	hipError_t ToHost(std::vector<T>& data, size_t count) const;
	hipError_t ToHost(size_t srcOffset, size_t bytes, void* data) const;

	// Copy from the argument array to this.
	hipError_t FromDevice(const T* data, size_t count);
	hipError_t FromDevice(size_t dstOffset, size_t bytes, const void* data);
	hipError_t FromHost(const std::vector<T>& data);
	hipError_t FromHost(const std::vector<T>& data, size_t count);
	hipError_t FromHost(const T* data, size_t count);
	hipError_t FromHost(size_t destOffset, size_t bytes, const void* data);

private:
	friend class HipContext;
	HipDeviceMem(HipAlloc* alloc) : _p(0), _size(0), _alloc(alloc) { }

	AllocPtr _alloc;
	T* _p; 
	size_t _size;
};

typedef intrusive_ptr<HipAlloc> AllocPtr;
#define MGPU_MEM(type) mgpu::intrusive_ptr< mgpu::HipDeviceMem< type > >  

////////////////////////////////////////////////////////////////////////////////
// HipMemSupport
// Convenience functions for allocating device memory and copying to it from
// the host. These functions are factored into their own class for clarity.
// The class is derived by HipContext.

class HipMemSupport : public HipBase {
	friend class HipDevice;
	friend class HipContext;
public:
	HipDevice& Device() { return _alloc->Device(); }

	// Swap out the associated allocator.
	void SetAllocator(HipAlloc* alloc) { 
		assert(alloc->Device().Ordinal() == _alloc->Device().Ordinal());
		_alloc.reset(alloc);
	}

	// Access the associated allocator.
	HipAlloc* GetAllocator() { return _alloc.get(); }	

	// Support for creating arrays.
	template<typename T>
	MGPU_MEM(T) Malloc(size_t count);

	template<typename T>
	MGPU_MEM(T) Malloc(const T* data, size_t count);

	template<typename T>
	MGPU_MEM(T) Malloc(const std::vector<T>& data);

	template<typename T>
	MGPU_MEM(T) Fill(size_t count, T fill);

	template<typename T>
	MGPU_MEM(T) FillAscending(size_t count, T first, T step);

	template<typename T>
	MGPU_MEM(T) GenRandom(size_t count, T min, T max);

	template<typename T>
	MGPU_MEM(T) SortRandom(size_t count, T min, T max);

	template<typename T, typename Func>
	MGPU_MEM(T) GenFunc(size_t count, Func f);

protected:
	HipMemSupport() { }
	AllocPtr _alloc;
};

////////////////////////////////////////////////////////////////////////////////

class HipContext;
typedef mgpu::intrusive_ptr<HipContext> ContextPtr;

// Create a context on the default stream (0).
ContextPtr CreateHipDevice(int ordinal);
ContextPtr CreateHipDevice(int argc, char** argv, bool printInfo = false);

// Create a context on a new stream.
ContextPtr CreateHipDeviceStream(int ordinal);
ContextPtr CreateHipDeviceStream(int argc, char** argv, 
	bool printInfo = false);

// Create a context and attach to an existing stream.
ContextPtr CreateHipDeviceAttachStream(hipStream_t stream);
ContextPtr CreateHipDeviceAttachStream(int ordinal, hipStream_t stream);

struct ContextGroup;

class HipContext : public HipMemSupport {
	friend struct ContextGroup;

	friend ContextPtr CreateHipDevice(int ordinal);
	friend ContextPtr CreateHipDeviceStream(int ordinal);
	friend ContextPtr CreateHipDeviceAttachStream(int ordinal, 
		hipStream_t stream);
public:
	static HipContext& StandardContext(int ordinal = -1);

	// 4KB of page-locked memory per context.
	int* PageLocked() { return _pageLocked; }
	hipStream_t AuxStream() const { return _auxStream; }

	int NumSMs() { return Device().NumSMs(); }
	int ArchVersion() { return Device().ArchVersion(); }
	int PTXVersion() { return Device().PTXVersion(); }
	std::string DeviceString() { return Device().DeviceString(); }

	hipStream_t Stream() const { return _stream; }

	// Set this device as the active device on the thread.
	void SetActive() { Device().SetActive(); }

	// Access the included event.
	HipEvent& Event() { return _event; }

	// Use the included timer.
	HipTimer& Timer() { return _timer; }
	void Start() { _timer.Start(); }
	double Split() { return _timer.Split(); }
	double Throughput(int count, int numIterations) {
		return _timer.Throughput(count, numIterations);
	}

	virtual long AddRef() {
		return _noRefCount ? 1 : HipMemSupport::AddRef();
	}
	virtual void Release() {
		if(!_noRefCount) HipMemSupport::Release();
	}
private:
	HipContext(HipDevice& device, bool newStream, bool standard);
	~HipContext();

	AllocPtr CreateDefaultAlloc(HipDevice& device);

	bool _ownStream;
	hipStream_t _stream;
	hipStream_t _auxStream;
	HipEvent _event;
	HipTimer _timer;
	bool _noRefCount;
	int* _pageLocked;
};

////////////////////////////////////////////////////////////////////////////////
// HipDeviceMem method implementations

template<typename T>
hipError_t HipDeviceMem<T>::ToDevice(T* data, size_t count) const {
	return ToDevice(0, sizeof(T) * count, data);
}
template<typename T>
hipError_t HipDeviceMem<T>::ToDevice(size_t srcOffset, size_t bytes, 
	void* data) const {
	hipError_t error = hipMemcpy(data, (char*)_p + srcOffset, bytes, 
		hipMemcpyDeviceToDevice);
	if(hipSuccess != error) {
		printf("HipDeviceMem::ToDevice copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
hipError_t HipDeviceMem<T>::ToHost(T* data, size_t count) const {
	return ToHost(0, sizeof(T) * count, data);
}
template<typename T>
hipError_t HipDeviceMem<T>::ToHost(std::vector<T>& data, size_t count) const {
	data.resize(count);
	hipError_t error = hipSuccess;
	if(_size) error = ToHost(&data[0], count);
	return error;
}
template<typename T>
hipError_t HipDeviceMem<T>::ToHost(std::vector<T>& data) const {
	return ToHost(data, _size);
}
template<typename T>
hipError_t HipDeviceMem<T>::ToHost(size_t srcOffset, size_t bytes, 
	void* data) const {

	hipError_t error = hipMemcpy(data, (char*)_p + srcOffset, bytes,
		hipMemcpyDeviceToHost);
	if(hipSuccess != error) {
		printf("HipDeviceMem::ToHost copy error %d\n", error);
		exit(0);
	}
	return error;
}

template<typename T>
hipError_t HipDeviceMem<T>::FromDevice(const T* data, size_t count) {
	return FromDevice(0, sizeof(T) * count, data);
}
template<typename T>
hipError_t HipDeviceMem<T>::FromDevice(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return hipErrorInvalidValue;
	hipMemcpy(_p + dstOffset, data, bytes, hipMemcpyDeviceToDevice);
	return hipSuccess;
}
template<typename T>
hipError_t HipDeviceMem<T>::FromHost(const std::vector<T>& data,
	size_t count) {
	hipError_t error = hipSuccess;
	if(data.size()) error = FromHost(&data[0], count);
	return error;
}
template<typename T>
hipError_t HipDeviceMem<T>::FromHost(const std::vector<T>& data) {
	return FromHost(data, data.size());
}
template<typename T>
hipError_t HipDeviceMem<T>::FromHost(const T* data, size_t count) {
	return FromHost(0, sizeof(T) * count, data);
}
template<typename T>
hipError_t HipDeviceMem<T>::FromHost(size_t dstOffset, size_t bytes,
	const void* data) {
	if(dstOffset + bytes > sizeof(T) * _size)
		return hipErrorInvalidValue;
	hipMemcpy(_p + dstOffset, data, bytes, hipMemcpyHostToDevice);
	return hipSuccess;
}
template<typename T>
HipDeviceMem<T>::~HipDeviceMem() {
	_alloc->Free(_p);
}

////////////////////////////////////////////////////////////////////////////////
// HipMemSupport method implementations

template<typename T>
MGPU_MEM(T) HipMemSupport::Malloc(size_t count) {
	MGPU_MEM(T) mem(new HipDeviceMem<T>(_alloc.get()));
	mem->_size = count;
	hipError_t error = _alloc->Malloc(sizeof(T) * count, (void**)&mem->_p);
	if(hipSuccess != error) {
		printf("hipMalloc error %d\n", error);		
		exit(0);
		throw HipException(hipErrorMemoryAllocation);
	}
#ifdef DEBUG
	// Initialize the memory to -1 in debug mode.
//	hipMemset(mem->get(), -1, count);
#endif

	return mem;
}

template<typename T>
MGPU_MEM(T) HipMemSupport::Malloc(const T* data, size_t count) {
	MGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

template<typename T>
MGPU_MEM(T) HipMemSupport::Malloc(const std::vector<T>& data) {
	MGPU_MEM(T) mem = Malloc<T>(data.size());
	if(data.size()) mem->FromHost(&data[0], data.size());
	return mem;
}

template<typename T>
MGPU_MEM(T) HipMemSupport::Fill(size_t count, T fill) {
	std::vector<T> data(count, fill);
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) HipMemSupport::FillAscending(size_t count, T first, T step) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = first + i * step;
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) HipMemSupport::GenRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	return Malloc(data);
}

template<typename T>
MGPU_MEM(T) HipMemSupport::SortRandom(size_t count, T min, T max) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = Rand(min, max);
	std::sort(data.begin(), data.end());
	return Malloc(data);
}

template<typename T, typename Func>
MGPU_MEM(T) HipMemSupport::GenFunc(size_t count, Func f) {
	std::vector<T> data(count);
	for(size_t i = 0; i < count; ++i)
		data[i] = f(i);

	MGPU_MEM(T) mem = Malloc<T>(count);
	mem->FromHost(data, count);
	return mem;
}

////////////////////////////////////////////////////////////////////////////////
// Format methods that operate directly on device mem.

template<typename T, typename Op>
std::string FormatArrayOp(const HipDeviceMem<T>& mem, int count, Op op,
	int numCols) {
	std::vector<T> host;
	mem.ToHost(host, count);
	return FormatArrayOp(host, op, numCols);
}

template<typename T, typename Op>
std::string FormatArrayOp(const HipDeviceMem<T>& mem, Op op, int numCols) {
	return FormatArrayOp(mem, mem.Size(), op, numCols);
}

template<typename T>
void PrintArray(const HipDeviceMem<T>& mem, int count, const char* format, 
	int numCols) {
	std::string s = FormatArrayOp(mem, count, FormatOpPrintf(format), numCols);
	printf("%s", s.c_str());
}

template<typename T>
void PrintArray(const HipDeviceMem<T>& mem, const char* format, int numCols) {
	PrintArray(mem, mem.Size(), format, numCols);
}
template<typename T, typename Op>
void PrintArrayOp(const HipDeviceMem<T>& mem, Op op, int numCols) {
	std::string s = FormatArrayOp(mem, op, numCols);
	printf("%s", s.c_str());
}


////////////////////////////////////////////////////////////////////////////////


} // namespace mgpu
