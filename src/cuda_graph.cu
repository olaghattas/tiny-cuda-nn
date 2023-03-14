#include <tiny-cuda-nn/cuda_graph.h>

TCNN_NAMESPACE_BEGIN

    CudaGraph::~CudaGraph() {
		try {
			reset();
		} catch (std::runtime_error error) {
			// Don't need to report on destruction problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				std::cerr << "Could not destroy cuda graph: " << error.what() << std::endl;
			}
		}
	}

	ScopeGuard CudaGraph::capture_guard(cudaStream_t stream) {
		// Can't capture on the global stream
		if (stream == nullptr || stream == cudaStreamLegacy) {
			return {};
		}

		// If the caller is already capturing, no need for a nested capture.
		cudaStreamCaptureStatus capture_status;
		CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &capture_status));
		if (capture_status != cudaStreamCaptureStatusNone) {
			return {};
		}

		cudaError_t capture_result = cudaStreamIsCapturing(cudaStreamLegacy, &capture_status);
		if (capture_result == cudaErrorStreamCaptureImplicit) {
			return {};
		}

		CUDA_CHECK_THROW(capture_result);
		if (capture_status != cudaStreamCaptureStatusNone) {
			return {};
		}

		// Start capturing
		if (m_graph) {
			CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
			m_graph = nullptr;
		}

		CUDA_CHECK_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
		current_captures().push_back(this);

		// Stop capturing again once the returned object goes out of scope
		return ScopeGuard{[this, stream]() {
			CUDA_CHECK_THROW(cudaStreamEndCapture(stream, &m_graph));

			if (current_captures().back() != this) {
				throw std::runtime_error{"CudaGraph: must end captures in reverse order of creation."};
			}
			current_captures().pop_back();

			if (m_synchronize_when_capture_done) {
				CUDA_CHECK_THROW(cudaDeviceSynchronize());
				m_synchronize_when_capture_done = false;
			}

			// Capture failed for some reason. Reset state and don't execute anything.
			// A corresponding exception is likely already in flight.
			if (!m_graph) {
				if (m_graph_instance) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
				}

				m_graph = nullptr;
				m_graph_instance = nullptr;
				return;
			}

			// If we previously created a graph instance, try to update it with the newly captured graph.
			// This is cheaper than creating a new instance from scratch (and may involve just updating
			// pointers rather than changing the topology of the graph.)
			if (m_graph_instance) {
				cudaGraphExecUpdateResult update_result;
				cudaGraphNode_t error_node;
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &error_node, &update_result));

				// If the update failed, reset graph instance. We will create a new one next.
				if (update_result != cudaGraphExecUpdateSuccess) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
					m_graph_instance = nullptr;
				}
			}

			if (!m_graph_instance) {
				CUDA_CHECK_THROW(cudaGraphInstantiate(&m_graph_instance, m_graph, NULL, NULL, 0));
			}

			CUDA_CHECK_THROW(cudaGraphLaunch(m_graph_instance, stream));
		}};
	}

	void CudaGraph::reset() {
		if (m_graph) {
			CUDA_CHECK_THROW(cudaGraphDestroy(m_graph));
			m_graph = nullptr;
		}

		if (m_graph_instance) {
			CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
			m_graph_instance = nullptr;
		}
	}

	void CudaGraph::schedule_synchronize() {
		m_synchronize_when_capture_done = true;
	}

TCNN_NAMESPACE_END
