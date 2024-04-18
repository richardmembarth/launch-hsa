#include "hsa_platform.h"
#include "log.h"

#include <string>

int main(int argc, char** argv) {
    HSAPlatform hsa;

    std::string file_name = "dummy.amdgpu.gcn";
    std::string kernel_name = "dummy_kernel";
    uint32_t grid[] = { 1, 1, 1 };
    uint32_t block[] = { 1, 1, 1 };
    struct ParamsArgs kernel_args;

    LaunchParams launch_params = {
        file_name.c_str(),
        kernel_name.c_str(),
        &grid[0],
        &block[0],
        kernel_args,
        0 // num_args
    };

    DeviceId dev = DeviceId(1); // 0: host, 1: first GPU in the system

    // this triggers a synchronization bug in the HSA runtime:
    // - the runtime uses a shared completion signal for all kernel launches
    // - when created, the shared completion signal is initialized to 0
    // - before each kernel launch, the completion signal is incremented by 1 
    // - the GPU decrements the completion signal by 1 after kernel execution
    // - synchronize() waits until the shared completion signal is 0 (all
    //   kernels have finished execution
    // - in some cases, the shared completion signal takes the value of -1 and
    //   synchronize() will wait forever
    for (int i = 0; i < 1000; ++i) {
        hsa.launch_kernel(dev, launch_params);
        if (i % 10 == 0)
            hsa.synchronize(dev);
    }
    hsa.synchronize(dev);

    return EXIT_SUCCESS;
}
