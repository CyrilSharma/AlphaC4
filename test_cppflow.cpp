#include <stdio.h>
#include <vector>
#include <tensorflow/c/c_api.h>
#include <numeric>
#include "cppflow/cppflow.h"

static void DeallocateTensor(void* data, std::size_t, void*) {
    std::free(data);
    std::cout << "Deallocate tensor" << std::endl;
}

int main() {

    cppflow::model model("../my_model");

    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*));

    std::vector<float> vals1 = {
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
    };

    std::vector<float> vals2 = {
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
    };

    std::vector<std::vector<float>> states = {vals1, vals2};

    const std::vector<std::int64_t> dims = {static_cast<long long>(states.size()), 6, 7, 1};

    const auto data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{});

    auto data = static_cast<float*>(std::malloc(data_size));

    std::vector<float> statevec;

    for (unsigned int i = 0; i < states.size(); i++) {
        statevec.insert(statevec.end(), states[i].begin(), states[i].end());
    }

    std::copy(statevec.begin(), statevec.end(), data);

    for (auto thing: statevec) {
        std::cout << " " << thing <<  std::endl;
    }

    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), static_cast<int>(dims.size()),
                                           data, data_size, DeallocateTensor, nullptr);

    auto output_tensor = model({{"serving_default_input_1:0", input_tensor}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

    std::cout << "thing_tensor: " << output_tensor[0] << std::endl;
    std::cout << "thing_tensor: " << output_tensor[1] << std::endl;

    std::cout << "thing_tensor: " << output_tensor[0] << std::endl;
    std::cout << "thing_tensor: " << output_tensor[1] << std::endl;

    std::vector<float> probs = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    std::vector<float> state_vals = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    std::vector<float> all_probs = output_tensor[0].get_data<float>();

    std::vector<float> all_state_vals = output_tensor[1].get_data<float>();

    for (unsigned int i = 0; i < states.size(); i++) {
        for (unsigned int j = 0; j < 7; j++) {
            probs[i] = all_probs[i * 7 + j];
        }
        for (unsigned int k = 0; k < 2; k++) {
            state_vals[i] = all_state_vals[i * 2 + k];
        }
    }

    int z = 0;
    for (auto i : probs) {
        z += 1;
    }

    std::cout << "things: " << z << std::endl;

    std::cout << "probs: " << probs.at(0) << probs.at(1) << std::endl;

    return 0;
}