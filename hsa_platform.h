#ifndef HSA_PLATFORM_H
#define HSA_PLATFORM_H

#include <cstdint>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

enum DeviceId : uint32_t {};

enum class KernelArgType : uint8_t { Val = 0, Ptr, Struct };

struct ParamsArgs {
    void** data;
    const uint32_t* sizes;
    const uint32_t* aligns;
    const uint32_t* alloc_sizes;
    const KernelArgType* types;
};

struct LaunchParams {
    const char* file_name;
    const char* kernel_name;
    const uint32_t* grid;
    const uint32_t* block;
    ParamsArgs args;
    uint32_t num_args;
};


/// HSA platform. Has the same number of devices as that of the HSA implementation.
class HSAPlatform {//: public Platform {
public:
    HSAPlatform();//Runtime* runtime);
    ~HSAPlatform();

    void* alloc(DeviceId dev, int64_t size) { return alloc_hsa(size, devices_[dev].amd_coarsegrained_pool); }
    void* alloc_host(DeviceId dev, int64_t size) { return alloc_hsa(size, devices_[dev].amd_coarsegrained_pool); }
    void* alloc_unified(DeviceId dev, int64_t size) { return alloc_hsa(size, devices_[dev].finegrained_region); }
    void* get_device_ptr(DeviceId, void* ptr) { return ptr; }
    void release(DeviceId dev, void* ptr);
    void release_host(DeviceId dev, void* ptr) { release(dev, ptr); }

    void launch_kernel(DeviceId dev, const LaunchParams& launch_params);
    void synchronize(DeviceId dev);

    void copy(const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size);
    void copy(DeviceId, const void* src, int64_t offset_src, DeviceId, void* dst, int64_t offset_dst, int64_t size) { copy(src, offset_src, dst, offset_dst, size); }
    void copy_from_host(const void* src, int64_t offset_src, DeviceId, void* dst, int64_t offset_dst, int64_t size) { copy(src, offset_src, dst, offset_dst, size); }
    void copy_to_host(DeviceId, const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) { copy(src, offset_src, dst, offset_dst, size); }

    size_t dev_count() const { return devices_.size(); }
    std::string name() const { return "HSA"; }
    const char* device_name(DeviceId dev) const;
    bool device_check_feature_support(DeviceId, const char*) const { return false; }

    struct KernelInfo {
        uint64_t kernel;
        uint32_t kernarg_segment_size;
        uint32_t group_segment_size;
        uint32_t private_segment_size;
        void*    kernarg_segment;
    };

    typedef std::unordered_map<std::string, KernelInfo> KernelMap;

    struct DeviceData {
        hsa_agent_t agent;
        hsa_profile_t profile;
        hsa_default_float_rounding_mode_t float_mode;
        std::string isa;
        hsa_queue_t* queue;
        hsa_signal_t signal;
        hsa_region_t kernarg_region, finegrained_region, coarsegrained_region;
        hsa_amd_memory_pool_t amd_kernarg_pool, amd_finegrained_pool, amd_coarsegrained_pool;
        std::atomic_flag locked = ATOMIC_FLAG_INIT;
        std::unordered_map<std::string, hsa_executable_t> programs;
        std::unordered_map<uint64_t, KernelMap> kernels;
        std::string name;

        DeviceData() {}
        DeviceData(const DeviceData&) = delete;
        DeviceData(DeviceData&& data)
            : agent(data.agent)
            , profile(data.profile)
            , float_mode(data.float_mode)
            , isa(data.isa)
            , queue(data.queue)
            , signal(data.signal)
            , kernarg_region(data.kernarg_region)
            , finegrained_region(data.finegrained_region)
            , coarsegrained_region(data.coarsegrained_region)
            , amd_kernarg_pool(data.amd_kernarg_pool)
            , amd_finegrained_pool(data.amd_finegrained_pool)
            , amd_coarsegrained_pool(data.amd_finegrained_pool)
            , programs(std::move(data.programs))
            , kernels(std::move(data.kernels))
            , name(data.name)
        {}

        void lock() {
            while (locked.test_and_set(std::memory_order_acquire)) ;
        }

        void unlock() {
            locked.clear(std::memory_order_release);
        }
    };

    uint64_t frequency_;
    std::vector<DeviceData> devices_;

    void* alloc_hsa(int64_t, hsa_region_t);
    void* alloc_hsa(int64_t, hsa_amd_memory_pool_t);
    static hsa_status_t iterate_agents_callback(hsa_agent_t, void*);
    static hsa_status_t iterate_regions_callback(hsa_region_t, void*);
    static hsa_status_t iterate_memory_pools_callback(hsa_amd_memory_pool_t, void*);
    KernelInfo& load_kernel(DeviceId, const std::string&, const std::string&);
};

#endif
