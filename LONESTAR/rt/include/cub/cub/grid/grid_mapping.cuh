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
 * hipcub::GridMappingStrategy enumerates alternative strategies for mapping constant-sized tiles of device-wide data onto a grid of CUDA thread blocks.
 */

#pragma once

#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup GridModule
 * @{
 */


/******************************************************************************
 * Mapping policies
 *****************************************************************************/


/**
 * \brief hipcub::GridMappingStrategy enumerates alternative strategies for mapping constant-sized tiles of device-wide data onto a grid of CUDA thread blocks.
 */
enum GridMappingStrategy
{
    /**
     * \brief An "even-share" strategy for assigning input tiles to thread blocks.
     *
     * \par Overview
     * The input is evenly partitioned into \p p segments, where \p p is
     * constant and corresponds loosely to the number of thread blocks that may
     * actively reside on the target device. Each segment is comprised of
     * consecutive tiles, where a tile is a small, constant-sized unit of input
     * to be processed to completion before the thread block terminates or
     * obtains more work.  The kernel invokes \p p thread blocks, each
     * of which iteratively consumes a segment of <em>n</em>/<em>p</em> elements
     * in tile-size increments.
     */
    GRID_MAPPING_EVEN_SHARE,

    /**
     * \brief A dynamic "queue-based" strategy for assigning input tiles to thread blocks.
     *
     * \par Overview
     * The input is treated as a queue to be dynamically consumed by a grid of
     * thread blocks.  Work is atomically dequeued in tiles, where a tile is a
     * unit of input to be processed to completion before the thread block
     * terminates or obtains more work.  The grid size \p p is constant,
     * loosely corresponding to the number of thread blocks that may actively
     * reside on the target device.
     */
    GRID_MAPPING_DYNAMIC,
};


/** @} */       // end group GridModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

