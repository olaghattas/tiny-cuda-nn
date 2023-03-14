/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   cuda_graph.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of a CUDA graph capture/update with subsequent execution
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <cuda.h>

#include <deque>
#include <functional>

TCNN_NAMESPACE_BEGIN

class CudaGraph;

inline std::deque<CudaGraph*>& current_captures() {
	static thread_local std::deque<CudaGraph*> s_current_captures;
	return s_current_captures;
}

inline CudaGraph* current_capture() {
	return current_captures().empty() ? nullptr : current_captures().front();
}

class CudaGraph {
public:
	~CudaGraph();

	ScopeGuard capture_guard(cudaStream_t stream);
	void reset();
	void schedule_synchronize();

private:
	cudaGraph_t m_graph = nullptr;
	cudaGraphExec_t m_graph_instance = nullptr;

	bool m_synchronize_when_capture_done = false;
};

TCNN_NAMESPACE_END
