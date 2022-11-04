/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

#define BLOCK_FACTOR 8
#define NAIVE 0  

__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
    short t;
#if defined(__LP64__) || defined(_WIN64)
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "l"(addr));
#else
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "r"(addr));
#endif
    return (bool)t;
}


inline __device__ void cudaBarrierAtomicNaiveSRB(unsigned int *globalBarr,
  // numBarr represents the number
  // of TBs going to the barrier
  const unsigned int numBarr,
  int backoff,
  const bool isMasterThread,
  bool * volatile global_sense) {
__syncthreads();
__shared__ bool s;
if (isMasterThread) {
s = !(ld_gbl_cg(global_sense));
__threadfence();
// atomicInc effectively adds 1 to atomic for each TB that's part of the
// global barrier.
atomicInc(globalBarr, 0x7FFFFFFF);
}
__syncthreads();

while (ld_gbl_cg(global_sense) != s) {
if (isMasterThread) {
/*
Once the atomic's value == numBarr, then reset the value to 0 and
proceed because all of the TBs have reached the global barrier.
*/
if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
__threadfence();
*global_sense = s;
} else { // increase backoff to avoid repeatedly hammering global barrier
// (capped) exponential backoff
backoff = (((backoff << 1) + 1) & (1024 - 1));
}
}
__syncthreads();

// do exponential backoff to reduce the number of times we pound the global
// barrier
if (ld_gbl_cg(global_sense) != s) {
for (int i = 0; i < backoff; ++i) {
;
}
__syncthreads();
}
}
}

inline __device__ void cudaBarrierAtomicSubSRB(unsigned int * globalBarr,
    // numBarr represents the number of
    // TBs going to the barrier
    const unsigned int numBarr,
    int backoff,
    const bool isMasterThread,
    bool * volatile sense,
    bool * volatile global_sense)
{
__syncthreads();
if (isMasterThread)
{
// atomicInc acts as a store release, need TF to enforce ordering
__threadfence();
// atomicInc effectively adds 1 to atomic for each TB that's part of the
// global barrier.
atomicInc(globalBarr, 0x7FFFFFFF);
}
__syncthreads();

while (*global_sense != *sense)
{
if (isMasterThread)
{
/*
For the tree barrier we expect only 1 TB from each SM to enter the
global barrier.  Since we are assuming an equal amount of work for all
SMs, we can use the # of TBs reaching the barrier for the compare value
here.  Once the atomic's value == numBarr, then reset the value to 0 and
proceed because all of the TBs have reached the global barrier.
*/
if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
__threadfence();
*global_sense = *sense;
//__threadfence();
}
else { // increase backoff to avoid repeatedly hammering global barrier
// (capped) exponential backoff
backoff = (((backoff << 1) + 1) & (1024-1));
}
}
__syncthreads();

// do exponential backoff to reduce the number of times we pound the global
// barrier
if(isMasterThread){
//if (*global_sense != *sense) {
// for (int i = 0; i < backoff; ++i) { ; }
}
__syncthreads();
//}
}
}

inline __device__ void cudaBarrierAtomicSRB(unsigned int * barrierBuffers,
 // numBarr represents the number of
 // TBs going to the barrier
 const unsigned int numBarr,
 const bool isMasterThread,
 bool * volatile sense,
 bool * volatile global_sense)
{
__shared__ int backoff;

if (isMasterThread) {
backoff = 1;
}
__syncthreads();

cudaBarrierAtomicSubSRB(barrierBuffers, numBarr, backoff, isMasterThread, sense, global_sense);
}

inline __device__ void cudaBarrierAtomicSubLocalSRB(unsigned int * perSMBarr,
         const unsigned int numTBs_thisSM,
         const bool isMasterThread,
         bool * sense,
         const int smID,
         unsigned int* last_block)

{
__syncthreads();
__shared__ bool s;
if (isMasterThread)
{
s = !(*sense);
// atomicInc acts as a store release, need TF to enforce ordering locally
__threadfence_block();
/*
atomicInc effectively adds 1 to atomic for each TB that's part of the
barrier.  For the local barrier, this requires using the per-CU
locations.
*/
atomicInc(perSMBarr, 0x7FFFFFFF);
}
__syncthreads();

while (*sense != s)
{
if (isMasterThread)
{
/*
Once all of the TBs on this SM have incremented the value at atomic,
then the value (for the local barrier) should be equal to the # of TBs
on this SM.  Once that is true, then we want to reset the atomic to 0
and proceed because all of the TBs on this SM have reached the local
barrier.
*/
if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == numTBs_thisSM) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
// locally
__threadfence_block();
*sense = s;
*last_block = blockIdx.x;
}
}
__syncthreads();
}
}

//Implements PerSM sense reversing barrier
inline __device__ void cudaBarrierAtomicLocalSRB(unsigned int * perSMBarrierBuffers,
               unsigned int * last_block,
               const unsigned int smID,
               const unsigned int numTBs_thisSM,
               const bool isMasterThread,
               bool* sense)
{
// each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
// start at barrierBuffers[smID*MAX_BLOCKS]
cudaBarrierAtomicSubLocalSRB(perSMBarrierBuffers, numTBs_thisSM, isMasterThread, sense, smID, last_block);
}

/*
Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const unsigned int numBlocksAtBarr,
const int smID,
const int perSM_blockID,
const int numTBs_perSM,
const bool isMasterThread,
bool naive) {                                 
__syncthreads();
if (numTBs_perSM > 1 && naive == false) {
cudaBarrierAtomicLocalSRB(&local_count[smID], &last_block[smID], smID, numTBs_perSM, isMasterThread, &perSMsense[smID]);

// only 1 TB per SM needs to do the global barrier since we synchronized
// the TBs locally first
if (blockIdx.x == last_block[smID]) {
    if(isMasterThread && perSM_blockID == 0){    
    }
    __syncthreads();
cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread , &perSMsense[smID], global_sense);  
//*done = 1;
}
else {
if(isMasterThread){
while (*global_sense != perSMsense[smID]){  
__threadfence();
}
}
__syncthreads();
}    
} else { // if only 1 TB on the SM, no need for the local barriers
  __shared__ int backoff;
  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
cudaBarrierAtomicNaiveSRB(global_count, (numBlocksAtBarr*numTBs_perSM), backoff,  isMasterThread,  global_sense);
}
}


__device__ void kernelAtomicTreeBarrierUniqSRB( bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const int NUM_SM,
bool naive)
{

// local variables
// thread 0 is master thread
const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
(threadIdx.z == 0));
// represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
// fewer TBs than SMs).
const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
NUM_SM);
const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID

// all thread blocks on the same SM access unique locations because the
// barrier can't ensure DRF between TBs
const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
// given the gridDim.x, we can figure out how many TBs are on our SM -- assume
// all SMs have an identical number of TBs

int numTBs_perSM = (int)ceil((float)gridDim.x / numBlocksAtBarr);

joinBarrier_helperSRB(global_sense, perSMsense, done, global_count, local_count, last_block,
numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
isMasterThread, naive);
}

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 32
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
extern int DELTA;
extern int start_node;
bool enable_lb = false;
typedef int edge_data_type;
typedef int node_data_type;
typedef int * gint_p;
extern const node_data_type INF = INT_MAX;
static const int __tb_one = 1;
static const int __tb_gg_main_pipe_1_gpu_gb = 32;
static const int __tb_sssp_kernel = TB_SIZE;
static const int __tb_remove_dups = TB_SIZE;
__global__ void kernel(CSRGraph graph, int src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = (node == src) ? 0 : INF ;
  }
}
__device__ void remove_dups_dev(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb, bool * global_sense = NULL,
  bool * perSMsense = NULL,
  bool * done = NULL,
  unsigned int* global_count = NULL,
  unsigned int* local_count = NULL,
  unsigned int* last_block = NULL,
  const int NUM_SM = 0,
 bool naive = false)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type wlnode_end;
  index_type wlnode2_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    marks[node] = wlnode;
  }
  //gb.Sync();
  kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
  wlnode2_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode2 = 0 + tid; wlnode2 < wlnode2_end; wlnode2 += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode2, node);
    if (marks[node] == wlnode2)
    {
      index_type _start_26;
      _start_26 = (out_wl).setup_push_warp_one();;
      (out_wl).do_push(_start_26, 0, node);
    }
  }
}
__global__ void remove_dups(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  remove_dups_dev(marks, in_wl, out_wl, gb);
}
__global__ void sssp_kernel_dev_TB_LB(CSRGraph graph, int delta, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ unsigned int total_work;
  __shared__ unsigned block_start_src_index;
  __shared__ unsigned block_end_src_index;
  unsigned my_work;
  unsigned node;
  unsigned int offset;
  unsigned int current_work;
  unsigned blockdim_x = BLOCK_DIM_X;
  total_work = thread_prefix_work_wl[num_items - 1];
  my_work = ceilf((float)(total_work) / (float) nthreads);

  __syncthreads();

  if (my_work != 0)
  {
    current_work = tid;
  }
  for (unsigned i =0; i < my_work; i++)
  {
    unsigned int block_start_work;
    unsigned int block_end_work;
    if (threadIdx.x == 0)
    {
      if (current_work < total_work)
      {
        block_start_work = current_work;
        block_end_work=current_work + blockdim_x - 1;
        if (block_end_work >= total_work)
        {
          block_end_work = total_work - 1;
        }
        block_start_src_index = compute_src_and_offset(0, num_items - 1,  block_start_work+1, thread_prefix_work_wl, num_items,offset);
        block_end_src_index = compute_src_and_offset(0, num_items - 1, block_end_work+1, thread_prefix_work_wl, num_items, offset);
      }
    }
    __syncthreads();

    if (current_work < total_work)
    {
      unsigned src_index;
      index_type edge;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      node= thread_src_wl.in_wl().dwl[src_index];
      edge = (graph).getFirstEdge(node)+ offset;
      {
        index_type dst = graph.getAbsDestination(edge);
        edge_data_type wt = graph.getAbsWeight(edge);
        if (graph.node_data[dst] > graph.node_data[node] + wt)
        {
          atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
          if (graph.node_data[node] + wt <= delta)
          {
            index_type _start_67;
            _start_67 = (re_wl).setup_push_warp_one();;
            (re_wl).do_push(_start_67, 0, dst);
          }
          else
          {
            (out_wl).push(dst);
          }
        }
      }
      current_work = current_work + nthreads;
    }
  }
}
__global__ void Inspect_sssp_kernel_dev(CSRGraph graph, int delta, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    int index;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) >= DEGREE_LIMIT)) ? true: false);
    if (pop && graph.node_data[node] == INF)
    {
      continue;
    }
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(node);
      thread_src_wl.in_wl().dwl[index] = node;
    }
  }
}
__device__ void sssp_kernel_dev(CSRGraph graph, int delta, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_sssp_kernel;
  index_type wlnode_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  unsigned d_limit = DEGREE_LIMIT;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) < DEGREE_LIMIT)) ? true: false);
    pop = pop && !(graph.node_data[node] == INF);
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { int node; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].node = node;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    if (threadIdx.x == 0)
    {
    }
    __syncthreads();
    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any(_np.size >= _NP_CROSSOVER_WP))
      {
        if (_np.size >= _NP_CROSSOVER_WP)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;
          nps.warp.src[warpid] = threadIdx.x;
          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        assert(nps.warp.src[warpid] < __kernel_tb_size);
        node = _np_closure[nps.warp.src[warpid]].node;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst = graph.getAbsDestination(edge);
            edge_data_type wt = graph.getAbsWeight(edge);
            if (graph.node_data[dst] > graph.node_data[node] + wt)
            {
              atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
              if (graph.node_data[node] + wt <= delta)
              {
                index_type _start_67;
                _start_67 = (re_wl).setup_push_warp_one();;
                (re_wl).do_push(_start_67, 0, dst);
              }
              else
              {
                (out_wl).push(dst);
              }
            }
          }
        }
      }
      __syncthreads();
    }

    __syncthreads();
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
          assert(nps.fg.src[_np_i] < __kernel_tb_size);
        node = _np_closure[nps.fg.src[_np_i]].node;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst = graph.getAbsDestination(edge);
          edge_data_type wt = graph.getAbsWeight(edge);
          if (graph.node_data[dst] > graph.node_data[node] + wt)
          {
            atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
            if (graph.node_data[node] + wt <= delta)
            {
              index_type _start_67;
              _start_67 = (re_wl).setup_push_warp_one();;
              (re_wl).do_push(_start_67, 0, dst);
            }
            else
            {
              (out_wl).push(dst);
            }
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    node = _np_closure[threadIdx.x].node;
  }
}
__global__ void __launch_bounds__(TB_SIZE, 2) sssp_kernel(CSRGraph graph, int delta, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_sssp_kernel;
  if (tid == 0)
    in_wl.reset_next_slot();

  sssp_kernel_dev(graph, delta, enable_lb, in_wl, out_wl, re_wl);
}
void gg_main_pipe_1(CSRGraph& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      pipe.out_wl().will_write();
      pipe.re_wl().will_write();
      if (enable_lb)
      {
        t_work.reset_thread_work();
        Inspect_sssp_kernel_dev <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb, pipe.in_wl(), pipe.out_wl());
        cudaDeviceSynchronize();
        int num_items = t_work.thread_work_wl.in_wl().nitems();
        if (num_items != 0)
        {
          t_work.compute_prefix_sum();
          cudaDeviceSynchronize();
          sssp_kernel_dev_TB_LB <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
          cudaDeviceSynchronize();
        }
      }
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      pipe.retry2();
    }
    pipe.advance2();
    pipe.out_wl().will_write();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraph gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, int* cl_curdelta, int* cl_i, bool enable_lb, GlobalBarrier gb, bool * global_sense,
    bool * perSMsense,
    bool * done,
    unsigned int* global_count,
    unsigned int* local_count,
    unsigned int* last_block,
    const int NUM_SM,
    bool naive)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D; 
  const unsigned __kernel_tb_size = TB_SIZE;
  curdelta = *cl_curdelta;
  i = *cl_i;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      if (tid == 0)
        pipe.in_wl().reset_next_slot();
      sssp_kernel_dev (gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      pipe.in_wl().swap_slots();
      //gb.Sync();
      kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
      //grid.sync();
      pipe.retry2();
    }
    //__syncthreads();
    //gb.Sync();
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);     
    pipe.advance2();
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    remove_dups_dev (glevel, pipe.in_wl(), pipe.out_wl(), gb, global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
    pipe.in_wl().swap_slots();
    //gb.Sync();
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);     
    //grid.sync();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  //gb.Sync();
  kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
  //grid.sync();
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
}
__global__ void gg_main_pipe_1_gpu(CSRGraph gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_curdelta, int* cl_i, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  curdelta = *cl_curdelta;
  i = *cl_i;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      cudaDeviceSynchronize();
      pipe.retry2();
    }
    cudaDeviceSynchronize();
    pipe.advance2();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
}
void gg_main_pipe_1_wrapper(CSRGraph& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
 // printf("the three values are %d and %d and %d\n", blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency, gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  if (enable_lb)
  {
    gg_main_pipe_1(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  }
  else
  {
    int* cl_curdelta;
    int* cl_i;
    check_cuda(cudaMalloc(&cl_curdelta, sizeof(int) * 1));
    check_cuda(cudaMalloc(&cl_i, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_curdelta, &curdelta, sizeof(int) * 1, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(cl_i, &i, sizeof(int) * 1, cudaMemcpyHostToDevice));
    bool * global_sense;
    bool* perSMsense;
    bool * done;
    unsigned int* global_count;
    unsigned int* local_count; 
    unsigned int *last_block;
    bool naive = true;
    int NUM_SM = ggc_get_nSM();
    cudaMallocManaged((void **)&global_sense,sizeof(bool));
    cudaMallocManaged((void **)&done,sizeof(bool));
    cudaMallocManaged((void **)&perSMsense,NUM_SM*sizeof(bool));
    cudaMallocManaged((void **)&last_block,sizeof(unsigned int)*(NUM_SM));
    cudaMallocManaged((void **)&local_count,  NUM_SM*sizeof(unsigned int));
    cudaMallocManaged((void **)&global_count,sizeof(unsigned int));
    
    cudaMemset(global_sense, false, sizeof(bool));
    cudaMemset(done, false, sizeof(bool));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    for (int i = 0; i < NUM_SM; ++i) {
       cudaMemset(&perSMsense[i], false, sizeof(bool));
       cudaMemset(&local_count[i], 0, sizeof(unsigned int));
       cudaMemset(&last_block[i], 0, sizeof(unsigned int));
     }
     cudaEvent_t start;
     cudaEvent_t stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start);
    // gg_main_pipe_1_gpu<<<1,1>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads,cl_curdelta,cl_i, enable_lb);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,cl_curdelta,cl_i, enable_lb, gg_main_pipe_1_gpu_gb_barrier, global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time (ms) is  " << ms <<   std::endl;
    check_cuda(cudaMemcpy(&curdelta, cl_curdelta, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&i, cl_i, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_curdelta));
    check_cuda(cudaFree(cl_i));
    cudaFree(global_sense);
    cudaFree(perSMsense);
    cudaFree(last_block);
    cudaFree(local_count);
    cudaFree(global_count);
    cudaFree(done);
  }
}
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  std::cout << " Enter Block factor" << std::endl;
  int block_factor;
  std::cin >> block_factor;
  blocks = ggc_get_nSM()*block_factor;
  t_work.init_thread_work(gg.nnodes);
  static GlobalBarrierLifetime remove_dups_barrier;
  static bool remove_dups_barrier_inited;
  gint_p glevel;
  PipeContextT<Worklist2> pipe;
  Shared<int> level (hg.nnodes);
  level.cpu_wr_ptr();
  static const size_t remove_dups_residency = maximum_residency(remove_dups, __tb_remove_dups, 0);
  static const size_t remove_dups_blocks = GG_MIN(blocks.x, ggc_get_nSM() * remove_dups_residency);
  if(!remove_dups_barrier_inited) { remove_dups_barrier.Setup(remove_dups_blocks); remove_dups_barrier_inited = true;};
  kernel <<<blocks, threads>>>(gg, start_node);
  cudaDeviceSynchronize();
  int i = 0;
  int curdelta = 0;
  printf("delta: %d\n", DELTA);
  glevel = level.gpu_wr_ptr();
  pipe = PipeContextT<Worklist2>(gg.nedges*2);
  pipe.in_wl().wl[0] = start_node;
  pipe.in_wl().update_gpu(1);
  gg_main_pipe_1_wrapper(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  printf("iterations: %d\n", i);
}