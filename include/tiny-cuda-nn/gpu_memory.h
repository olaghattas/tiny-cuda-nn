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

/** @file   gpu_memory.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Managed memory on the GPU. Like a std::vector, memory is allocated either explicitly (resize/enlarge)
 *          or implicitly (resize_and_copy_from_host etc). Memory is always and automatically released in the destructor.
 *          Also contains a GPU memory arena for light-weight stream-ordered allocations of temporary memory. The
 *          memory arena makes use of virtual memory when available to avoid re-allocations during progressive growing.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cuda_graph.h>

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

TCNN_NAMESPACE_BEGIN

#define DEBUG_GUARD_SIZE 0

    inline std::atomic<size_t> &total_n_bytes_allocated() {
        static std::atomic<size_t> s_total_n_bytes_allocated{0};
        return s_total_n_bytes_allocated;
    }

/// Managed memory on the Device
    template<typename T>
    class GPUMemory {
    private:
        T *m_data = nullptr;
        size_t m_size = 0; // Number of elements
        bool m_managed = false;

    public:
        GPUMemory();

        GPUMemory(size_t size, bool managed = false);

        GPUMemory<T> &operator=(GPUMemory<T> &&other);

        GPUMemory(GPUMemory<T> &&other);

        // Don't permit copy assignment to prevent performance accidents.
        // Copy is permitted through an explicit copy constructor.
        GPUMemory<T> &operator=(const GPUMemory<T> &other) = delete;

        explicit GPUMemory(const GPUMemory<T> &other);

        void check_guards() const;

        void allocate_memory(size_t n_bytes);

        void free_memory();

        /// Frees memory again
        TCNN_HOST_DEVICE ~GPUMemory();
        /** @name Resizing/enlargement
         *  @{
         */
        /// Resizes the array to the exact new size, even if it is already larger
        void resize(const size_t size);

        /// Enlarges the array if its size is smaller
        void enlarge(const size_t size);

        /** @name Memset
         *  @{
         */
        /// Sets the memory of the first num_elements to value
        void memset(const int value, const size_t num_elements, const size_t offset = 0);

        /// Sets the memory of the all elements to value
        void memset(const int value);

        /** @name Copy operations
         *  @{
         */
        /// Copy data of num_elements from the raw pointer on the host
        void copy_from_host(const T *host_data, const size_t num_elements);

        /// Copy num_elements from the host vector
        void copy_from_host(const std::vector<T> &data, const size_t num_elements);

        /// Copies data from the raw host pointer to fill the entire array
        void copy_from_host(const T *data);

        /// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const T *data, const size_t num_elements);

        /// Copies num_elements from the host vector after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const std::vector<T> &data, const size_t num_elements);

        /// Copies the entire host vector after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const std::vector<T> &data);

        /// Copies num_elements of data from the raw host pointer after resizing the array
        void resize_and_copy_from_host(const T *data, const size_t num_elements);

        /// Copies num_elements from the host vector after resizing the array
        void resize_and_copy_from_host(const std::vector<T> &data, const size_t num_elements);

        /// Copies the entire host vector after resizing the array
        void resize_and_copy_from_host(const std::vector<T> &data);

        /// Copies the entire host vector to the device. Fails if there is not enough space available.
        void copy_from_host(const std::vector<T> &data);

        /// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space available.
        void copy_to_host(T *host_data, const size_t num_elements) const;

        /// Copies num_elements from the device to a vector on the host
        void copy_to_host(std::vector<T> &data, const size_t num_elements) const;

        /// Copies num_elements from the device to a raw pointer on the host
        void copy_to_host(T *data) const;

        /// Copies all elements from the device to a vector on the host
        void copy_to_host(std::vector<T> &data) const;

        /// Copies size elements from another device array to this one, automatically resizing it
        void copy_from_device(const GPUMemory<T> &other, const size_t size);

        /// Copies data from another device array to this one, automatically resizing it
        void copy_from_device(const GPUMemory<T> &other);

        // Created an (owned) copy of the data
        GPUMemory<T> copy(size_t size) const;

        GPUMemory<T> copy() const;

        T *data() const;

        bool managed() const;

        T &at(size_t idx) const;

        TCNN_HOST_DEVICE T &operator[](size_t idx) const;

        TCNN_HOST_DEVICE T &operator[](uint32_t idx) const;

        size_t get_num_elements() const;

        size_t size() const;

        size_t get_bytes() const;

        size_t bytes() const;
    };

    struct Interval {
        // Inclusive start, exclusive end
        size_t start, end;

        bool operator<(const Interval &other) const;

        bool overlaps(const Interval &other) const;

        Interval intersect(const Interval &other) const;

        bool valid() const;

        bool empty() const;

        size_t size() const;
    };

    class GPUMemoryArena {
    public:
        GPUMemoryArena();

        GPUMemoryArena(GPUMemoryArena &&other) = default;

        GPUMemoryArena(const GPUMemoryArena &other) = delete;

        GPUMemoryArena &operator=(GPUMemoryArena &&other) = delete;

        GPUMemoryArena &operator=(const GPUMemoryArena &other) = delete;

        ~GPUMemoryArena();

        uint8_t *data();

        std::shared_ptr<GPUMemory<uint8_t>> backing_memory();

        // Finds the smallest interval of free memory in the GPUMemoryArena that's
        // large enough to hold the requested number of bytes. Then allocates
        // that memory.
        size_t allocate(size_t n_bytes);

        void free(size_t start);

        void enlarge(size_t n_bytes);

        size_t size() const;

        bool in_use() const;

        class Allocation {
        public:
            Allocation() = default;

            Allocation(cudaStream_t stream, size_t offset, const std::shared_ptr<GPUMemoryArena> &workspace);

            ~Allocation();

            Allocation(const Allocation &other) = delete;

            Allocation &operator=(Allocation &&other);

            Allocation(Allocation &&other);

            uint8_t *data();

            const uint8_t *data() const;

            cudaStream_t stream() const;

        private:
            cudaStream_t m_stream = nullptr;
            uint8_t *m_data = nullptr;
            size_t m_offset = 0;
            std::shared_ptr<GPUMemoryArena> m_workspace = nullptr;

            // Backing GPUMemory (if backed by a GPUMemory). Ensures that
            // the backing memory is only freed once all allocations that
            // use it were destroyed.
            std::shared_ptr<GPUMemory<uint8_t>> m_backing_memory = nullptr;
        };

    private:
        void merge_adjacent_intervals();

        std::vector<Interval> m_free_intervals;
        std::unordered_map<size_t, size_t> m_allocated_intervals;

        int m_device = 0;
        CUdeviceptr m_base_address = {};
        size_t m_size = 0;

        std::vector<CUmemGenericAllocationHandle> m_handles;

        // Used then virtual memory isn't supported.
        // Requires more storage + memcpy, but is more portable.
        std::shared_ptr<GPUMemory<uint8_t>> m_fallback_memory = nullptr;

        size_t m_alignment;
        size_t m_max_size;
    };

    std::unordered_map<cudaStream_t, std::shared_ptr<GPUMemoryArena>> &stream_gpu_memory_arenas();

    std::unordered_map<int, std::shared_ptr<GPUMemoryArena>> &global_gpu_memory_arenas();

    GPUMemoryArena::Allocation allocate_workspace(cudaStream_t stream, size_t n_bytes);


    size_t align_to_cacheline(size_t bytes);


    template<typename First, typename FirstSize>
    std::tuple<First *>
    allocate_workspace_and_distribute(cudaStream_t stream, GPUMemoryArena::Allocation *alloc, size_t offset,
                                      FirstSize first_size) {
        *alloc = allocate_workspace(stream, offset + align_to_cacheline(first_size * sizeof(First)));
        return std::make_tuple<First *>((First *) (alloc->data() + offset));
    }


    template<typename First, typename ...Types, typename FirstSize, typename ...Sizes, std::enable_if_t<
            sizeof...(Types) != 0 && sizeof...(Types) == sizeof...(Sizes), int> = 0>
    std::tuple<First *, Types *...>
    allocate_workspace_and_distribute(cudaStream_t stream, GPUMemoryArena::Allocation *alloc, size_t offset,
                                      FirstSize first_size, Sizes... sizes) {
        auto nested = allocate_workspace_and_distribute < Types...>(stream, alloc, offset + align_to_cacheline(
                first_size * sizeof(First)), sizes...);
        return std::tuple_cat(std::make_tuple<First *>((First *) (alloc->data() + offset)), nested);
    }

    template<typename ...Types, typename ...Sizes, std::enable_if_t<sizeof...(Types) == sizeof...(Sizes), int> = 0>
    std::tuple<Types *...>
    allocate_workspace_and_distribute(cudaStream_t stream, GPUMemoryArena::Allocation *alloc, Sizes... sizes) {
        return allocate_workspace_and_distribute < Types...>(stream, alloc, (size_t) 0, sizes...);
    }


    void free_gpu_memory_arena(cudaStream_t stream);

    void free_all_gpu_memory_arenas();


template class GPUMemory<double>;
template class GPUMemory<float>;
template class GPUMemory<char>;
template class GPUMemory<half>;
template class GPUMemory<unsigned int>;
template class GPUMemory<unsigned char>;

TCNN_NAMESPACE_END
