#include <torch/script.h>

#include <iostream>
#include <memory>
#include <chrono>


// cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH="/home/shreyansh/libtorch;/usr/local/cuda" ..
// cmake --build . --config Release
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: torchscript_gpu_v5 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        module.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the module\n";
        return -1;
    }

    std::cout << "model loaded.\n";

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    at::Tensor example_tensor = torch::rand({1, 3, 224, 224}, at::kCUDA);
    inputs.push_back(example_tensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';


    for (int i=0; i<10; i++) {
        at::Tensor output = module.forward(inputs).toTensor();
    }

    int repetitions = 1000;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_point, end_point; // creating time points

    start_point = std::chrono::high_resolution_clock::now();

    for (int i=0; i<repetitions; i++) {
        at::Tensor output = module.forward(inputs).toTensor();
    }

    end_point = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_point).time_since_epoch().count(); 
	// casting the time point to milliseconds and measuring the time since time epoch
    auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_point).time_since_epoch().count();
	
    std::cout<<"Mean time: "<<(end-start)/float(repetitions)<<"ms"<<"\n";
	std::cout<<"Total time: "<<(end-start)<<"ms"<<"\n";

    return 0;
}
