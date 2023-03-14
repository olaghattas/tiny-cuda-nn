#include <tiny-cuda-nn/gpu_memory.h>

//TCNN_NAMESPACE_BEGIN

/// Managed memory on the Device
template<typename T>
tcnn::GPUMemory<T>::GPUMemory() {}

template<typename T>
tcnn::GPUMemory<T>::GPUMemory(size_t size, bool managed) : m_managed{managed} {
    resize(size);
}

template<typename T>
tcnn::GPUMemory<T> &tcnn::GPUMemory<T>::operator=(GPUMemory<T> &&other) {
    std::swap(m_data, other.m_data);
    std::swap(m_size, other.m_size);
    std::swap(m_managed, other.m_managed);
    return *this;
}

template<typename T>
tcnn::GPUMemory<T>::GPUMemory(GPUMemory<T> &&other) {
    *this = std::move(other);
}

// Don't permit copy assignment to prevent performance accidents.
// Copy is permitted through an explicit copy constructor.
template<typename T>
tcnn::GPUMemory<T>::GPUMemory(const GPUMemory<T> &other) {
    m_managed = other.managed();
    copy_from_device(other);
}

template<typename T>
void tcnn::GPUMemory<T>::check_guards() const {
#if DEBUG_GUARD_SIZE > 0
    if (!m_data)
        return;
    uint8_t buf[DEBUG_GUARD_SIZE];
    const uint8_t *rawptr=(const uint8_t *)m_data;
    cudaMemcpy(buf, rawptr-DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
    for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i] != 0xff) {
        printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected 0xff!\n", i, m_data, buf[i] );
        break;
    }
    cudaMemcpy(buf, rawptr+m_size*sizeof(T), DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
    for (int i=0;i<DEBUG_GUARD_SIZE;++i) if (buf[i] != 0xfe) {
        printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i] );
        break;
    }
#endif
}

template<typename T>
void tcnn::GPUMemory<T>::allocate_memory(size_t n_bytes) {
    if (n_bytes == 0) {
        return;
    }

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
    std::cout << "GPUMemory: Allocating " << bytes_to_string(n_bytes) << "." << std::endl;
#endif

    uint8_t *rawptr = nullptr;
    if (m_managed) {
        CUDA_CHECK_THROW(cudaMallocManaged(&rawptr, n_bytes + DEBUG_GUARD_SIZE * 2));
    } else {
        CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes + DEBUG_GUARD_SIZE * 2));
    }
#if DEBUG_GUARD_SIZE > 0
    CUDA_CHECK_THROW(cudaMemset(rawptr, 0xff, DEBUG_GUARD_SIZE));
    CUDA_CHECK_THROW(cudaMemset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xfe, DEBUG_GUARD_SIZE));
#endif
    if (rawptr) rawptr += DEBUG_GUARD_SIZE;
    m_data = (T *) (rawptr);
    total_n_bytes_allocated() += n_bytes;
}

template<typename T>
void tcnn::GPUMemory<T>::free_memory() {
    if (!m_data) {
        return;
    }

    uint8_t *rawptr = (uint8_t *) m_data;
    if (rawptr) rawptr -= DEBUG_GUARD_SIZE;
    CUDA_CHECK_THROW(cudaFree(rawptr));

    total_n_bytes_allocated() -= get_bytes();

    m_data = nullptr;
    m_size = 0;
}

/// Frees memory again
template<typename T>
TCNN_HOST_DEVICE tcnn::GPUMemory<T>::~GPUMemory() {
#ifndef __CUDA_ARCH__
    try {
        if (m_data) {
            free_memory();
            m_size = 0;
        }
    } catch (std::runtime_error error) {
        // Don't need to report on memory-free problems when the driver is shutting down.
        if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
            std::cerr << "Could not free memory: " << error.what() << std::endl;
        }
    }
#endif
}

/** @name Resizing/enlargement
 *  @{
 */
/// Resizes the array to the exact new size, even if it is already larger
template<typename T> void tcnn::GPUMemory<T>::resize(const size_t size) {
    if (m_size != size) {
        if (m_size) {
            try {
                free_memory();
            } catch (std::runtime_error error) {
                throw std::runtime_error{fmt::format("Could not free memory: {}", error.what())};
            }
        }

        if (size > 0) {
            try {
                allocate_memory(size * sizeof(T));
            } catch (std::runtime_error error) {
                throw std::runtime_error{fmt::format("Could not allocate memory: {}", error.what())};
            }
        }

        m_size = size;
    }
}

/// Enlarges the array if its size is smaller
template<typename T>
void tcnn::GPUMemory<T>::enlarge(const size_t size) {
    if (size > m_size) {
        resize(size);
    }
}
/** @} */

/** @name Memset
 *  @{
 */
/// Sets the memory of the first num_elements to value
template<typename T>
void tcnn::GPUMemory<T>::memset(const int value, const size_t num_elements, const size_t offset) {
    if (num_elements + offset > m_size) {
        throw std::runtime_error{
                fmt::format("Could not set memory: Number of elements {}+{} larger than allocated memory {}.",
                            num_elements, offset, m_size)};
    }

    CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
}

/// Sets the memory of the all elements to value
template<typename T>
void tcnn::GPUMemory<T>::memset(const int value) {
    memset(value, m_size);
}
/** @} */

/** @name Copy operations
 *  @{
 */
/// Copy data of num_elements from the raw pointer on the host
template<typename T>
void tcnn::GPUMemory<T>::copy_from_host(const T *host_data, const size_t num_elements) {
    CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
}

/// Copy num_elements from the host vector
template<typename T>
void tcnn::GPUMemory<T>::copy_from_host(const std::vector<T> &data, const size_t num_elements) {
    if (data.size() < num_elements) {
        throw std::runtime_error{
                fmt::format("Trying to copy {} elements, but vector size is only {}.", num_elements, data.size())};
    }
    copy_from_host(data.data(), num_elements);
}

/// Copies data from the raw host pointer to fill the entire array
template<typename T>
void tcnn::GPUMemory<T>::copy_from_host(const T *data) {
    copy_from_host(data, m_size);
}

/// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
template<typename T>
void tcnn::GPUMemory<T>::enlarge_and_copy_from_host(const T *data, const size_t num_elements) {
    enlarge(num_elements);
    copy_from_host(data, num_elements);
}

/// Copies num_elements from the host vector after enlarging the array so that everything fits in
template<typename T>
void tcnn::GPUMemory<T>::enlarge_and_copy_from_host(const std::vector<T> &data, const size_t num_elements) {
    enlarge_and_copy_from_host(data.data(), num_elements);
}

/// Copies the entire host vector after enlarging the array so that everything fits in
template<typename T>
void tcnn::GPUMemory<T>::enlarge_and_copy_from_host(const std::vector<T> &data) {
    enlarge_and_copy_from_host(data.data(), data.size());
}

/// Copies num_elements of data from the raw host pointer after resizing the array
template<typename T>
void tcnn::GPUMemory<T>::resize_and_copy_from_host(const T *data, const size_t num_elements) {
    resize(num_elements);
    copy_from_host(data, num_elements);
}

/// Copies num_elements from the host vector after resizing the array
template<typename T>
void tcnn::GPUMemory<T>::resize_and_copy_from_host(const std::vector<T> &data, const size_t num_elements) {
    resize_and_copy_from_host(data.data(), num_elements);
}

/// Copies the entire host vector after resizing the array
template<typename T>
void tcnn::GPUMemory<T>::resize_and_copy_from_host(const std::vector<T> &data) {
    resize_and_copy_from_host(data.data(), data.size());
}

/// Copies the entire host vector to the device. Fails if there is not enough space available.
template<typename T>
void tcnn::GPUMemory<T>::copy_from_host(const std::vector<T> &data) {
    if (data.size() < m_size) {
        throw std::runtime_error{
                fmt::format("Trying to copy {} elements, but vector size is only {}.", m_size, data.size())};
    }
    copy_from_host(data.data(), m_size);
}

/// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space available.
template<typename T>
void tcnn::GPUMemory<T>::copy_to_host(T *host_data, const size_t num_elements) const {
    if (num_elements > m_size) {
        throw std::runtime_error{
                fmt::format("Trying to copy {} elements, but memory size is only {}.", num_elements, m_size)};
    }

    CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
}

/// Copies num_elements from the device to a vector on the host
template<typename T>
void tcnn::GPUMemory<T>::copy_to_host(std::vector<T> &data, const size_t num_elements) const {
    if (data.size() < num_elements) {
        throw std::runtime_error{
                fmt::format("Trying to copy {} elements, but vector size is only {}.", num_elements, data.size())};
    }

    copy_to_host(data.data(), num_elements);
}

/// Copies num_elements from the device to a raw pointer on the host
template<typename T>
void tcnn::GPUMemory<T>::copy_to_host(T *data) const {
    copy_to_host(data, m_size);
}

/// Copies all elements from the device to a vector on the host
template<typename T>
void tcnn::GPUMemory<T>::copy_to_host(std::vector<T> &data) const {
    if (data.size() < m_size) {
        throw std::runtime_error{
                fmt::format("Trying to copy {} elements, but vector size is only {}", m_size, data.size())};
    }

    copy_to_host(data.data(), m_size);
}

/// Copies size elements from another device array to this one, automatically resizing it
template<typename T>
void tcnn::GPUMemory<T>::copy_from_device(const tcnn::GPUMemory<T> &other, const size_t size) {
    if (size == 0) {
        return;
    }

    if (m_size < size) {
        resize(size);
    }

    CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

/// Copies data from another device array to this one, automatically resizing it
template<typename T>
void tcnn::GPUMemory<T>::copy_from_device(const tcnn::GPUMemory<T> &other) {
    copy_from_device(other, other.m_size);
}

// Created an (owned) copy of the data
template<typename T>
tcnn::GPUMemory<T> tcnn::GPUMemory<T>::copy(size_t size) const {
    GPUMemory<T> result{size};
    result.copy_from_device(*this);
    return result;
}

template<typename T>
tcnn::GPUMemory<T> tcnn::GPUMemory<T>::copy() const {
    return copy(m_size);
}

template<typename T>
T *tcnn::GPUMemory<T>::data() const {
    check_guards();
    return m_data;
}

template<typename T>
bool tcnn::GPUMemory<T>::managed() const {
    return m_managed;
}

template<typename T>
T &tcnn::GPUMemory<T>::at(size_t idx) const {
    if (!m_managed) {
        throw std::runtime_error{fmt::format("GPUMemory::at() not permitted if not managed.")};
    }

    if (idx > m_size) {
        throw std::runtime_error{fmt::format("GPUMemory our of bounds: idx={} size={}", idx, m_size)};
    }

    return m_data[idx];
}

template<typename T>
TCNN_HOST_DEVICE T &tcnn::GPUMemory<T>::operator[](size_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
    if (idx > m_size) {
        printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
    }
#endif
    return m_data[idx];
}

template<typename T>
TCNN_HOST_DEVICE T &tcnn::GPUMemory<T>::operator[](uint32_t idx) const {
#ifdef DEBUG_BUFFER_OVERRUN
    if (idx > m_size) {
        printf("WARNING: buffer overrun of %p at idx %u\n", idx);
    }
#endif
    return m_data[idx];
}

template<typename T>
size_t tcnn::GPUMemory<T>::get_num_elements() const {
    return m_size;
}

template<typename T>
size_t tcnn::GPUMemory<T>::size() const {
    return get_num_elements();
}

template<typename T>
size_t tcnn::GPUMemory<T>::get_bytes() const {
    return m_size * sizeof(T);
}

template<typename T>
size_t tcnn::GPUMemory<T>::bytes() const {
    return get_bytes();
}


bool tcnn::Interval::operator<(const Interval &other) const {
    // This operator is used to sort non-overlapping intervals. Since intervals
    // may be empty, the second half of the following expression is required to
    // resolve ambiguity when `end` of adjacent empty intervals is equal.
    return end < other.end || (end == other.end && start < other.start);
}

bool tcnn::Interval::overlaps(const Interval &other) const {
    return !intersect(other).empty();
}

tcnn::Interval tcnn::Interval::intersect(const Interval &other) const {
    return {std::max(start, other.start), std::min(end, other.end)};
}

bool tcnn::Interval::valid() const {
    return end >= start;
}

bool tcnn::Interval::empty() const {
    return end <= start;
}

size_t tcnn::Interval::size() const {
    return end - start;
}


tcnn::GPUMemoryArena::GPUMemoryArena() {
    m_device = cuda_device();

    // Align memory at least by a cache line (128 bytes).
    m_alignment = (size_t) 128;
    m_max_size = next_multiple(cuda_memory_info().total, cuda_memory_granularity());

    m_free_intervals = {{0, m_max_size}};

    if (!cuda_supports_virtual_memory()) {
        // Use regular memory as fallback
        m_fallback_memory = std::make_shared<GPUMemory<uint8_t>>();

        static bool printed_warning = false;
        if (!printed_warning) {
            printed_warning = true;
            std::cout
                    << "GPUMemoryArena: Warning: GPU " << m_device << " does not support virtual memory. "
                    << "Falling back to regular allocations, which will be larger and can cause occasional stutter."
                    << std::endl;
        }
        return;
    }

    // Reserve an address range that would be sufficient for housing the entire
    // available GPU RAM (if nothing else was using the GPU). This is unlikely
    // to exhaust all available addresses (even if multiple GPUMemoryArenas are
    // used simultaneously), while also ensuring that we never exhaust the
    // reserved address range without running out of physical memory beforehand.
    CU_CHECK_THROW(cuMemAddressReserve(&m_base_address, m_max_size, 0, 0, 0));
}


tcnn::GPUMemoryArena::~GPUMemoryArena() {
    if (in_use()) {
        std::cerr << "Attempting to free memory arena while it is still in use." << std::endl;
    }

    try {
        // Make sure we're clearing the GPU memory arena on the correct device.
        int previous_device = cuda_device();
        set_cuda_device(m_device);
        ScopeGuard revert_device = {[&]() { set_cuda_device(previous_device); }};

        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        if (m_base_address) {
            total_n_bytes_allocated() -= m_size;

            CU_CHECK_THROW(cuMemUnmap(m_base_address, m_size));

            for (const auto &handle: m_handles) {
                CU_CHECK_THROW(cuMemRelease(handle));
            }

            CU_CHECK_THROW(cuMemAddressFree(m_base_address, m_max_size));
        }
    } catch (std::runtime_error error) {
        // Don't need to report on memory-free problems when the driver is shutting down.
        if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
            std::cerr << "Could not free memory arena: " << error.what() << std::endl;
        }
    }
}

uint8_t *tcnn::GPUMemoryArena::data() {
    return m_fallback_memory ? m_fallback_memory->data() : (uint8_t *) m_base_address;
}

std::shared_ptr<tcnn::GPUMemory<uint8_t>> tcnn::GPUMemoryArena::backing_memory() {
    return m_fallback_memory;
}

// Finds the smallest interval of free memory in the GPUMemoryArena that's
// large enough to hold the requested number of bytes. Then allocates
// that memory.
size_t tcnn::GPUMemoryArena::allocate(size_t n_bytes) {
    // Permitting zero-sized allocations is error prone
    if (n_bytes == 0) {
        n_bytes = m_alignment;
    }

    // Align allocations with the nearest cache line (at least the granularity of the memory allocations)
    n_bytes = next_multiple(n_bytes, m_alignment);

    Interval *best_candidate = &m_free_intervals.back();
    for (auto &f: m_free_intervals) {
        if (f.size() >= n_bytes && f.size() < best_candidate->size()) {
            best_candidate = &f;
        }
    }

    size_t start = best_candidate->start;

    // Note: the += operator can turn `best_candidate` into an empty interval, which is fine because it will
    // be absorbed into adjacent free intervals in later calls to `merge_adjacent_intervals`.
    m_allocated_intervals[start] = best_candidate->start += n_bytes;

    enlarge(size());

    return start;
}

void tcnn::GPUMemoryArena::free(size_t start) {
    if (m_allocated_intervals.count(start) == 0) {
        throw std::runtime_error{"Attempted to free arena memory that was not allocated."};
    }

    Interval interval = {start, m_allocated_intervals[start]};
    m_allocated_intervals.erase(start);

    m_free_intervals.insert(
            std::upper_bound(std::begin(m_free_intervals), std::end(m_free_intervals), interval),
            interval
    );

    merge_adjacent_intervals();
}

void tcnn::GPUMemoryArena::enlarge(size_t n_bytes) {
    if (n_bytes <= m_size) {
        return;
    }

    if (cuda_device() != m_device) {
        throw std::runtime_error{
                fmt::format("Attempted to use a GPUMemoryArena of device {} from the wrong device {}.", m_device,
                            cuda_device())};
    }

    if (m_fallback_memory) {
        static const double GROWTH_FACTOR = 1.5;

        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        m_size = next_multiple((size_t) (n_bytes * GROWTH_FACTOR), cuda_memory_granularity());
        m_fallback_memory = std::make_shared<GPUMemory<uint8_t>>(m_fallback_memory->copy(m_size));

        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        return;
    }

    size_t n_bytes_to_allocate = n_bytes - m_size;
    n_bytes_to_allocate = next_multiple(n_bytes_to_allocate, cuda_memory_granularity());

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = m_device;

    m_handles.emplace_back();
    CU_CHECK_THROW(cuMemCreate(&m_handles.back(), n_bytes_to_allocate, &prop, 0));

    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = prop.location.id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK_THROW(cuMemMap(m_base_address + m_size, n_bytes_to_allocate, 0, m_handles.back(), 0));
    CU_CHECK_THROW(cuMemSetAccess(m_base_address + m_size, n_bytes_to_allocate, &access_desc, 1));
    m_size += n_bytes_to_allocate;

    total_n_bytes_allocated() += n_bytes_to_allocate;

    // Need to synchronize the device to make sure memory is available to all streams.
    if (current_capture()) {
        current_capture()->schedule_synchronize();
    } else {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
}

size_t tcnn::GPUMemoryArena::size() const {
    return m_free_intervals.back().start;
}

bool tcnn::GPUMemoryArena::in_use() const {
    return m_free_intervals.size() != 1 || m_free_intervals.front().size() != m_max_size;
}

tcnn::GPUMemoryArena::Allocation::Allocation(cudaStream_t stream, size_t offset,
                                       const std::shared_ptr<tcnn::GPUMemoryArena> &workspace)
        : m_stream{stream}, m_data{workspace->data() + offset}, m_offset{offset}, m_workspace{workspace},
          m_backing_memory{workspace->backing_memory()} {}

tcnn::GPUMemoryArena::Allocation::~Allocation() {
    if (m_workspace) {
        m_workspace->free(m_offset);
    }
}

tcnn::GPUMemoryArena::Allocation &tcnn::GPUMemoryArena::Allocation::operator=(tcnn::GPUMemoryArena::Allocation &&other) {
    std::swap(m_stream, other.m_stream);
    std::swap(m_data, other.m_data);
    std::swap(m_offset, other.m_offset);
    std::swap(m_workspace, other.m_workspace);
    std::swap(m_backing_memory, other.m_backing_memory);
    return *this;
}

tcnn::GPUMemoryArena::Allocation::Allocation(tcnn::GPUMemoryArena::Allocation &&other) {
    *this = std::move(other);
}

uint8_t *tcnn::GPUMemoryArena::Allocation::data() {
    return m_data;
}

const uint8_t *tcnn::GPUMemoryArena::Allocation::data() const {
    return m_data;
}

cudaStream_t tcnn::GPUMemoryArena::Allocation::stream() const {
    return m_stream;
}

void tcnn::GPUMemoryArena::merge_adjacent_intervals() {
    size_t j = 0;
    for (size_t i = 1; i < m_free_intervals.size(); ++i) {
        Interval &prev = m_free_intervals[j];
        Interval &cur = m_free_intervals[i];

        if (prev.end == cur.start) {
            prev.end = cur.end;
        } else {
            ++j;
            m_free_intervals[j] = m_free_intervals[i];
        }
    }
    m_free_intervals.resize(j + 1);
}


std::unordered_map<cudaStream_t, std::shared_ptr<tcnn::GPUMemoryArena>> & tcnn::stream_gpu_memory_arenas() {
    static auto *stream_gpu_memory_arenas = new std::unordered_map<cudaStream_t, std::shared_ptr<tcnn::GPUMemoryArena>>{};
    return *stream_gpu_memory_arenas;
}

std::unordered_map<int, std::shared_ptr<tcnn::GPUMemoryArena>> & tcnn::global_gpu_memory_arenas() {
    static auto *global_gpu_memory_arenas = new std::unordered_map<int, std::shared_ptr<tcnn::GPUMemoryArena>>{};
    return *global_gpu_memory_arenas;
}

tcnn::GPUMemoryArena::Allocation tcnn::allocate_workspace(cudaStream_t stream, size_t n_bytes) {
    if (n_bytes == 0) {
        // Return a null allocation if no bytes were requested.
        return {};
    }

    auto &arena = stream ? stream_gpu_memory_arenas()[stream] : global_gpu_memory_arenas()[tcnn::cuda_device()];
    if (!arena) {
        arena = std::make_shared<tcnn::GPUMemoryArena>();
    }
    return tcnn::GPUMemoryArena::Allocation{stream, arena->allocate(n_bytes), arena};
}

size_t tcnn::align_to_cacheline(size_t bytes) {
    return next_multiple(bytes, (size_t) 128);
}

//size_t align_to_cacheline(size_t bytes) {
//    return tcnn::next_multiple(bytes, (size_t) 128);
//}

//template<typename First, typename FirstSize>
//std::tuple<First *>
//allocate_workspace_and_distribute(cudaStream_t stream, tcnn::GPUMemoryArena::Allocation *alloc, size_t offset,
//                                  FirstSize first_size) {
//    *alloc = allocate_workspace(stream, offset + align_to_cacheline(first_size * sizeof(First)));
//    return std::make_tuple<First *>((First *) (alloc->data() + offset));
//}
//
//template<typename First, typename ...Types, typename FirstSize, typename ...Sizes, std::enable_if_t<
//        sizeof...(Types) != 0 && sizeof...(Types) == sizeof...(Sizes), int> >
//std::tuple<First *, Types *...>
//tcnn::allocate_workspace_and_distribute(cudaStream_t stream, tcnn::GPUMemoryArena::Allocation *alloc, size_t offset,
//                                  FirstSize first_size, Sizes... sizes) {
//    auto nested = allocate_workspace_and_distribute<Types...>(stream, alloc,
//                                                              offset + align_to_cacheline(first_size * sizeof(First)),
//                                                              sizes...);
//    return std::tuple_cat(std::make_tuple<First *>((First *) (alloc->data() + offset)), nested);
//}
//
//template<typename ...Types, typename ...Sizes, std::enable_if_t<sizeof...(Types) == sizeof...(Sizes), int>>
//std::tuple<Types *...>
//tcnn::allocate_workspace_and_distribute(cudaStream_t stream, GPUMemoryArena::Allocation *alloc, Sizes... sizes) {
//    return allocate_workspace_and_distribute<Types...>(stream, alloc, (size_t) 0, sizes...);
//}

void tcnn::free_gpu_memory_arena(cudaStream_t stream) {
    if (stream) {
        stream_gpu_memory_arenas().erase(stream);
    } else {
        global_gpu_memory_arenas().erase(cuda_device());
    }
}

void tcnn::free_all_gpu_memory_arenas() {
    stream_gpu_memory_arenas().clear();
    global_gpu_memory_arenas().clear();
}

void annoyingFuncThatinstantiates(){
    tcnn::GPUMemory<float> tmp1;
    tcnn::GPUMemory<char> tmp2;
    tcnn::GPUMemory<half> tmp3;
    tcnn::GPUMemory<unsigned int> tmp4;
}


//TCNN_NAMESPACE_END