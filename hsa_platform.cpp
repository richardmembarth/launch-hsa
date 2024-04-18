#include "hsa_platform.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <sstream>
#include <thread>

inline std::string read_stream(std::istream& stream) {
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void store_file(const std::string& filename, const std::string& str) {
    const std::byte* data = reinterpret_cast<const std::byte*>(str.data());
    std::ofstream dst_file(filename, std::ofstream::binary);
    if (!dst_file)
        error("Can't open destination file '%'", filename);
    dst_file.write(reinterpret_cast<const char*>(data), str.length());
}

std::string load_file(const std::string& filename) {
    std::ifstream src_file(filename);
    if (!src_file)
        error("Can't open source file '%'", filename);
    return read_stream(src_file);
}

#define CHECK_HSA(err, name) check_hsa_error(err, name, __FILE__, __LINE__)
#define CODE_OBJECT_VERSION 3

inline void check_hsa_error(hsa_status_t err, const char* name, const char* file, const int line) {
    if (err != HSA_STATUS_SUCCESS) {
        const char* status_string;
        hsa_status_t ret = hsa_status_string(err, &status_string);
        if (ret != HSA_STATUS_SUCCESS)
            info("hsa_status_string failed: %", ret);
        error("HSA API function % (%) [file %, line %]: %", name, err, file, line, status_string);
    }
}

std::string get_device_profile_str(hsa_profile_t profile) {
    #define HSA_PROFILE_TYPE(TYPE) case TYPE: return #TYPE;
    switch (profile) {
        HSA_PROFILE_TYPE(HSA_PROFILE_BASE)
        HSA_PROFILE_TYPE(HSA_PROFILE_FULL)
        default: return "unknown HSA profile";
    }
}

std::string get_device_type_str(hsa_device_type_t device_type) {
    #define HSA_DEVICE_TYPE(TYPE) case TYPE: return #TYPE;
    switch (device_type) {
        HSA_DEVICE_TYPE(HSA_DEVICE_TYPE_CPU)
        HSA_DEVICE_TYPE(HSA_DEVICE_TYPE_GPU)
        HSA_DEVICE_TYPE(HSA_DEVICE_TYPE_DSP)
        default: return "unknown HSA device type";
    }
}

std::string get_region_segment_str(hsa_region_segment_t region_segment) {
    #define HSA_REGION_SEGMENT(TYPE) case TYPE: return #TYPE;
    switch (region_segment) {
        HSA_REGION_SEGMENT(HSA_REGION_SEGMENT_GLOBAL)
        HSA_REGION_SEGMENT(HSA_REGION_SEGMENT_READONLY)
        HSA_REGION_SEGMENT(HSA_REGION_SEGMENT_PRIVATE)
        HSA_REGION_SEGMENT(HSA_REGION_SEGMENT_GROUP)
        HSA_REGION_SEGMENT(HSA_REGION_SEGMENT_KERNARG)
        default: return "unknown HSA region segment";
    }
}

std::string get_memory_pool_segment_str(hsa_amd_segment_t amd_segment) {
    #define HSA_AMD_SEGMENT(TYPE) case TYPE: return #TYPE;
    switch (amd_segment) {
        HSA_AMD_SEGMENT(HSA_AMD_SEGMENT_GLOBAL)
        HSA_AMD_SEGMENT(HSA_AMD_SEGMENT_READONLY)
        HSA_AMD_SEGMENT(HSA_AMD_SEGMENT_PRIVATE)
        HSA_AMD_SEGMENT(HSA_AMD_SEGMENT_GROUP)
        default: return "unknown HSA AMD segment";
    }
}

hsa_status_t HSAPlatform::iterate_agents_callback(hsa_agent_t agent, void* data) {
    auto devices_ = static_cast<std::vector<DeviceData>*>(data);
    hsa_status_t status;

    char agent_name[64] = { 0 };
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("  (%) Device Name: %", devices_->size(), agent_name);
    char product_name[64] = { 0 };
    status = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, product_name);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      Device Product Name: %", product_name);
    char vendor_name[64] = { 0 };
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME, vendor_name);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      Device Vendor: %", vendor_name);

    hsa_profile_t profile;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &profile);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      Device profile: %", get_device_profile_str(profile));

    hsa_default_float_rounding_mode_t float_mode;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &float_mode);
    CHECK_HSA(status, "hsa_agent_get_info()");

    hsa_isa_t isa;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
    CHECK_HSA(status, "hsa_agent_get_info()");
    uint32_t name_length;
    status = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_length);
    char isa_name[64] = { 0 };
    status = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
    debug("      Device ISA: %", isa_name);

    hsa_device_type_t device_type;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      Device Type: %", get_device_type_str(device_type));

    uint16_t version_major, version_minor;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_VERSION_MAJOR, &version_major);
    CHECK_HSA(status, "hsa_agent_get_info()");
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_VERSION_MINOR, &version_minor);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      HSA Runtime Version: %.%", version_major, version_minor);

    uint32_t queue_size = 0;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    CHECK_HSA(status, "hsa_agent_get_info()");
    debug("      Queue Size: %", queue_size);

    hsa_queue_t* queue = nullptr;
    if (queue_size > 0) {
        status = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
        CHECK_HSA(status, "hsa_queue_create()");

        status = hsa_amd_profiling_set_profiler_enabled(queue, 1);
        CHECK_HSA(status, "hsa_amd_profiling_set_profiler_enabled()");
    }

    auto dev = devices_->size();
    devices_->resize(dev + 1);
    DeviceData* device = &(*devices_)[dev];
    device->agent = agent;
    device->profile = profile;
    device->float_mode = float_mode;
    device->isa = agent_name;
    device->queue = queue;
    device->kernarg_region.handle = { 0 };
    device->finegrained_region.handle = { 0 };
    device->coarsegrained_region.handle = { 0 };
    device->amd_kernarg_pool.handle = { 0 };
    device->amd_finegrained_pool.handle = { 0 };
    device->amd_coarsegrained_pool.handle = { 0 };
    device->name = product_name;

    status = hsa_signal_create(0, 0, nullptr, &device->signal);
    CHECK_HSA(status, "hsa_signal_create()");
    status = hsa_agent_iterate_regions(agent, iterate_regions_callback, device);
    CHECK_HSA(status, "hsa_agent_iterate_regions()");
    status = hsa_amd_agent_iterate_memory_pools(agent, iterate_memory_pools_callback, device);
    CHECK_HSA(status, "hsa_amd_agent_iterate_memory_pools()");

    return HSA_STATUS_SUCCESS;
}

hsa_status_t HSAPlatform::iterate_regions_callback(hsa_region_t region, void* data) {
    DeviceData* device = static_cast<DeviceData*>(data);
    hsa_status_t status;

    hsa_region_segment_t segment;
    status = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    CHECK_HSA(status, "hsa_region_get_info()");
    debug("      Region Segment: %", get_region_segment_str(segment));

    hsa_region_global_flag_t flags;
    status = hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    CHECK_HSA(status, "hsa_region_get_info()");

    std::string global_flags;
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
        global_flags += "HSA_REGION_GLOBAL_FLAG_KERNARG ";
        device->kernarg_region = region;
    }
    if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
        global_flags += "HSA_REGION_GLOBAL_FLAG_FINE_GRAINED ";
        device->finegrained_region = region;
    }
    if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
        global_flags += "HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED ";
        device->coarsegrained_region = region;
    }
    debug("      Region Global Flags: %", global_flags);

    bool runtime_alloc_allowed;
    status = hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &runtime_alloc_allowed);
    CHECK_HSA(status, "hsa_region_get_info()");
    debug("      Region Runtime Alloc Allowed: %", runtime_alloc_allowed);

    return HSA_STATUS_SUCCESS;
}

hsa_status_t HSAPlatform::iterate_memory_pools_callback(hsa_amd_memory_pool_t memory_pool, void* data) {
    DeviceData* device = static_cast<DeviceData*>(data);
    hsa_status_t status;

    hsa_amd_segment_t segment;
    status = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    CHECK_HSA(status, "hsa_amd_memory_pool_get_info()");
    debug("      AMD Memory Pool Segment: %", get_memory_pool_segment_str(segment));

    hsa_amd_memory_pool_global_flag_t flags;
    status = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    CHECK_HSA(status, "hsa_amd_memory_pool_get_info()");

    std::string global_flags;
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
        global_flags += "HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT ";
        device->amd_kernarg_pool = memory_pool;
    }
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        global_flags += "HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED ";
        device->amd_finegrained_pool = memory_pool;
    }
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
        global_flags += "HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED ";
        device->amd_coarsegrained_pool = memory_pool;
    }
    debug("      AMD Memory Pool Global Flags: %", global_flags);

    bool runtime_alloc_allowed;
    status = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &runtime_alloc_allowed);
    CHECK_HSA(status, "hsa_amd_memory_pool_get_info()");
    debug("      AMD Memory Pool Runtime Alloc Allowed: %", runtime_alloc_allowed);

    return HSA_STATUS_SUCCESS;
}

HSAPlatform::HSAPlatform() {
    hsa_status_t status = hsa_init();
    //if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
    //    info("HSA runtime failed to initialize (HSA_STATUS_ERROR_OUT_OF_RESOURCES). This is likely caused by a lack of suitable HSA devices and may be ignored.");
    //    return;
    //}
    CHECK_HSA(status, "hsa_init()");

    uint16_t version_major, version_minor;
    status = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major);
    CHECK_HSA(status, "hsa_system_get_info()");
    status = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &version_minor);
    CHECK_HSA(status, "hsa_system_get_info()");
    debug("HSA System Runtime Version: %.%", version_major, version_minor);

    status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_);
    CHECK_HSA(status, "hsa_system_get_info()");

    status = hsa_iterate_agents(iterate_agents_callback, &devices_);
    CHECK_HSA(status, "hsa_iterate_agents()");
}

HSAPlatform::~HSAPlatform() {
    hsa_status_t status;

    for (size_t i = 0; i < devices_.size(); i++) {
        for (auto& it : devices_[i].programs) {
            status = hsa_executable_destroy(it.second);
            CHECK_HSA(status, "hsa_executable_destroy()");
        }
        if (auto queue = devices_[i].queue) {
            status = hsa_queue_destroy(queue);
            CHECK_HSA(status, "hsa_queue_destroy()");
        }
        status = hsa_signal_destroy(devices_[i].signal);
        CHECK_HSA(status, "hsa_signal_destroy()");
        for (auto kernel_pair : devices_[i].kernels) {
            for (auto kernel : kernel_pair.second) {
                if (kernel.second.kernarg_segment) {
                    status = hsa_memory_free(kernel.second.kernarg_segment);
                    CHECK_HSA(status, "hsa_memory_free()");
                }
            }
        }
    }

    hsa_shut_down();
}

void* HSAPlatform::alloc_hsa(int64_t size, hsa_region_t region) {
    if (!size)
        return nullptr;

    char* mem;
    hsa_status_t status = hsa_memory_allocate(region, size, (void**) &mem);
    CHECK_HSA(status, "hsa_memory_allocate()");

    return (void*)mem;
}

void* HSAPlatform::alloc_hsa(int64_t size, hsa_amd_memory_pool_t memory_pool) {
    if (!size)
        return nullptr;

    char* mem;
    hsa_status_t status = hsa_amd_memory_pool_allocate(memory_pool, size, 0, (void**) &mem);
    CHECK_HSA(status, "hsa_amd_memory_pool_allocate()");

    return (void*)mem;
}

void HSAPlatform::release(DeviceId, void* ptr) {
    hsa_status_t status = hsa_amd_memory_pool_free(ptr);
    CHECK_HSA(status, "hsa_amd_memory_pool_free()");
}

void HSAPlatform::launch_kernel(DeviceId dev, const LaunchParams& launch_params) {
    auto queue = devices_[dev].queue;
    if (!queue)
        error("The selected HSA device '%' cannot execute kernels", dev);

    auto& kernel_info = load_kernel(dev, launch_params.file_name, launch_params.kernel_name);

    auto align_up = [&] (unsigned int start, unsigned int align) -> unsigned int {
        return (start + align - 1U) & -align;
    };

    // set up arguments
    if (launch_params.num_args) {
        if (!kernel_info.kernarg_segment) {
            size_t total_size = 0;
            for (uint32_t i = 0; i < launch_params.num_args; i++)
                total_size = (total_size + launch_params.args.aligns[i] - 1) /
                    launch_params.args.aligns[i] * launch_params.args.aligns[i] + launch_params.args.alloc_sizes[i];
            kernel_info.kernarg_segment_size = total_size;
            hsa_status_t status = hsa_memory_allocate(devices_[dev].kernarg_region, kernel_info.kernarg_segment_size, &kernel_info.kernarg_segment);
            CHECK_HSA(status, "hsa_memory_allocate()");
        }
        void*  cur   = kernel_info.kernarg_segment;
        size_t space = kernel_info.kernarg_segment_size;
        for (uint32_t i = 0; i < launch_params.num_args; i++) {
            // align base address for next kernel argument
            if (!std::align(launch_params.args.aligns[i], launch_params.args.alloc_sizes[i], cur, space))
                error("Incorrect kernel argument alignment detected");
            std::memcpy(cur, launch_params.args.data[i], launch_params.args.sizes[i]);
            cur = reinterpret_cast<uint8_t*>(cur) + launch_params.args.alloc_sizes[i];
        }

        size_t total = reinterpret_cast<uint8_t*>(cur) - reinterpret_cast<uint8_t*>(kernel_info.kernarg_segment);
        if (align_up(total, sizeof(int64_t)) == kernel_info.kernarg_segment_size - 32) {
            // implicit kernel args: global offset x, y, z
            for (int i=0; i<3; ++i) {
                if (!std::align(sizeof(int64_t), sizeof(int64_t), cur, space))
                    error("Incorrect kernel argument alignment detected");
                std::memset(cur, 0, sizeof(int64_t));
                cur = reinterpret_cast<uint8_t*>(cur) + sizeof(int64_t);
            }
            cur = reinterpret_cast<uint8_t*>(cur) + sizeof(int64_t);
        }

        total = reinterpret_cast<uint8_t*>(cur) - reinterpret_cast<uint8_t*>(kernel_info.kernarg_segment);
        if (total != kernel_info.kernarg_segment_size) {
            error("HSA kernarg segment size for kernel '%' differs from argument size: % vs. %",
                launch_params.kernel_name, kernel_info.kernarg_segment_size, total);
        }
    }

    auto signal = devices_[dev].signal;
    hsa_signal_add_relaxed(signal, 1);

    // construct aql packet
    hsa_kernel_dispatch_packet_t aql;
    std::memset(&aql, 0, sizeof(aql));

    aql.header =
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) |
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER);
    aql.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    aql.workgroup_size_x = (uint16_t)launch_params.block[0];
    aql.workgroup_size_y = (uint16_t)launch_params.block[1];
    aql.workgroup_size_z = (uint16_t)launch_params.block[2];
    aql.grid_size_x = launch_params.grid[0];
    aql.grid_size_y = launch_params.grid[1];
    aql.grid_size_z = launch_params.grid[2];
    aql.completion_signal    = signal;
    aql.kernel_object        = kernel_info.kernel;
    aql.kernarg_address      = kernel_info.kernarg_segment;
    aql.private_segment_size = kernel_info.private_segment_size;
    aql.group_segment_size   = kernel_info.group_segment_size;

    // write to command queue
    const uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    const uint32_t queue_mask = queue->size - 1;
    ((hsa_kernel_dispatch_packet_t*)(queue->base_address))[index & queue_mask] = aql;
    hsa_queue_store_write_index_relaxed(queue, index + 1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);
}

void HSAPlatform::synchronize(DeviceId dev) {
    auto signal = devices_[dev].signal;
    info("synchronize: signal: %", hsa_signal_load_scacquire(signal));
    hsa_signal_value_t completion = hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    if (completion != 0)
        error("HSA signal completion failed: %", completion);
}

void HSAPlatform::copy(const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) {
    hsa_status_t status = hsa_memory_copy((char*)dst + offset_dst, (char*)src + offset_src, size);
    CHECK_HSA(status, "hsa_memory_copy()");
}

HSAPlatform::KernelInfo& HSAPlatform::load_kernel(DeviceId dev, const std::string& filename, const std::string& kernelname) {
    auto& hsa_dev = devices_[dev];
    hsa_status_t status;

    hsa_dev.lock();

    hsa_executable_t executable = { 0 };
    auto canonical = std::filesystem::weakly_canonical(filename);
    auto& prog_cache = hsa_dev.programs;
    auto prog_it = prog_cache.find(canonical.string());
    if (prog_it == prog_cache.end()) {
        hsa_dev.unlock();

        if (canonical.extension() != ".gcn")
            error("Incorrect extension for kernel file '%' (should be '.gcn')", canonical.string());

        // load file from disk
        std::string gcn = load_file(canonical.string());

        hsa_code_object_reader_t reader;
        status = hsa_code_object_reader_create_from_memory(gcn.data(), gcn.size(), &reader);
        CHECK_HSA(status, "hsa_code_object_reader_create_from_file()");

        debug("Compiling '%' on HSA device %", canonical.string(), dev);

        status = hsa_executable_create_alt(HSA_PROFILE_FULL /* hsa_dev.profile */, hsa_dev.float_mode, nullptr, &executable);
        CHECK_HSA(status, "hsa_executable_create_alt()");

        // TODO
        //hsa_loaded_code_object_t program_code_object;
        //status = hsa_executable_load_program_code_object(executable, reader, "", &program_code_object);
        //CHECK_HSA(status, "hsa_executable_load_program_code_object()");
        // -> hsa_executable_global_variable_define()
        // -> hsa_executable_agent_variable_define()
        // -> hsa_executable_readonly_variable_define()

        hsa_loaded_code_object_t agent_code_object;
        status = hsa_executable_load_agent_code_object(executable, hsa_dev.agent, reader, nullptr, &agent_code_object);
        CHECK_HSA(status, "hsa_executable_load_agent_code_object()");

        status = hsa_executable_freeze(executable, nullptr);
        CHECK_HSA(status, "hsa_executable_freeze()");

        status = hsa_code_object_reader_destroy(reader);
        CHECK_HSA(status, "hsa_code_object_reader_destroy()");

        uint32_t validated;
        status = hsa_executable_validate(executable, &validated);
        CHECK_HSA(status, "hsa_executable_validate()");

        if (validated != 0)
            debug("HSA executable validation failed: %", validated);

        hsa_dev.lock();
        prog_cache[canonical.string()] = executable;
    } else {
        executable = prog_it->second;
    }

    // checks that the kernel exists
    auto& kernel_cache = hsa_dev.kernels;
    auto& kernel_map = kernel_cache[executable.handle];
    auto kernel_it = kernel_map.find(kernelname);
    if (kernel_it == kernel_map.end()) {
        hsa_dev.unlock();

        hsa_executable_symbol_t kernel_symbol = { 0 };
        std::string symbol_name = kernelname + ".kd";
        // DEPRECATED: use hsa_executable_get_symbol_by_linker_name if available
        status = hsa_executable_get_symbol_by_name(executable, symbol_name.c_str(), &hsa_dev.agent, &kernel_symbol);
        CHECK_HSA(status, "hsa_executable_get_symbol_by_name()");

        KernelInfo kernel_info;
        kernel_info.kernarg_segment = nullptr;

        status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_info.kernel);
        CHECK_HSA(status, "hsa_executable_symbol_get_info()");
        status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernel_info.kernarg_segment_size);
        CHECK_HSA(status, "hsa_executable_symbol_get_info()");
        status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kernel_info.group_segment_size);
        CHECK_HSA(status, "hsa_executable_symbol_get_info()");
        status = hsa_executable_symbol_get_info(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kernel_info.private_segment_size);
        CHECK_HSA(status, "hsa_executable_symbol_get_info()");

        #if CODE_OBJECT_VERSION > 3
        // metadata are not yet extracted from code object version 3
        // https://github.com/RadeonOpenCompute/ROCR-Runtime/blob/master/src/loader/executable.cpp#L1428
        if (kernel_info.kernarg_segment_size) {
            status = hsa_memory_allocate(hsa_dev.kernarg_region, kernel_info.kernarg_segment_size, &kernel_info.kernarg_segment);
            CHECK_HSA(status, "hsa_memory_allocate()");
        }
        #endif

        hsa_dev.lock();
        std::tie(kernel_it, std::ignore) = kernel_cache[executable.handle].emplace(kernelname, kernel_info);
    }

    // We need to get the reference now, while we have the lock, since re-hashing
    // may impact the validity of the iterator (but references are *not* invalidated)
    KernelInfo& kernel_info = kernel_it->second;
    hsa_dev.unlock();

    return kernel_info;
}

const char* HSAPlatform::device_name(DeviceId dev) const {
    return devices_[dev].name.c_str();
}
