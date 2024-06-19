#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <iostream>
#include "src/kernels/fused_addresidual_norm.h"

#include <stdio.h>
// (RussWong)note:
// `./test_fused_addresidual_norm` to test fp32 GPU kernel
// (RussWong)note: this kernel's CPU implementation is absolutely right.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel

#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

void CPUfusedresidandRMSNorm(float *h_residual, float *h_decoder_out, float *h_bias,
                             float *h_scale, float eps, int hidden_units, int num_tokens)
{
    for (int b = 0; b < num_tokens; b++)
    {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float input = 0.0f;
        for (int i = 0; i < hidden_units; i++)
        {
            input = h_decoder_out[b * hidden_units + i] +
                    h_residual[b * hidden_units + i] + h_bias[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < hidden_units; i++)
        {
            sum += input * input;
        }

        mean = (float)(sum / hidden_units);
        inv_fenmu = rsqrt(mean + eps);

        for (int i = 0; i < hidden_units; i++)
        {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
        }
    }
}

template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size)
{
    for (int i = 0; i < output_size; i++)
    {
	float f_GPUoutput = (float)GPUoutput[i];
        if (fabs(CPUoutput[i] - f_GPUoutput) > 1e-6)
        {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], f_GPUoutput);
            return false;
        }
    }
    return true;
}
#define TEST_FUSED_ADD_RMS(dtype)                                                                      \
    dtype *h_residual;                                                                                   \
    dtype *d_residual;                                                                                   \
    h_residual = (dtype *)malloc(sizeof(dtype) * total_size);                                            \
    cudaMalloc((void **)&d_residual, sizeof(dtype) * total_size);                                        \
    for (int i = 0; i < total_size; i++)                                                                 \
    {                                                                                                    \
        h_residual[i] = 0.0f;                                                                            \
    }                                                                                                    \
    dtype *h_decoder_out = (dtype *)malloc(sizeof(dtype) * total_size);                                  \
    dtype *decoder_out = (dtype *)malloc(sizeof(dtype) * total_size);                                    \
    dtype *d_decoder_out;                                                                                \
    cudaMalloc((void **)&d_decoder_out, sizeof(dtype) * total_size);                                     \
    for (int i = 0; i < total_size; i++)                                                                 \
    {                                                                                                    \
        h_decoder_out[i] = 1.0f;                                                                         \
    }                                                                                                    \
    dtype *h_bias = (dtype *)malloc(sizeof(dtype) * hidden_units);                                       \
    dtype *d_bias;                                                                                       \
    cudaMalloc((void **)&d_bias, sizeof(dtype) * hidden_units);                                          \
    for (int i = 0; i < hidden_units; i++)                                                               \
    {                                                                                                    \
        h_bias[i] = 0.0f;                                                                                \
    }                                                                                                    \
    dtype *h_scale = (dtype *)malloc(sizeof(dtype) * hidden_units);                                      \
    dtype *d_scale;                                                                                      \
    cudaMalloc((void **)&d_scale, sizeof(dtype) * hidden_units);                                         \
    for (int i = 0; i < hidden_units; i++)                                                               \
    {                                                                                                    \
        h_scale[i] = 1.0f;                                                                               \
    }                                                                                                    \
    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(dtype) * total_size, cudaMemcpyHostToDevice));       \
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(dtype) * total_size, cudaMemcpyHostToDevice)); \
    CHECK(cudaMemcpy(d_bias, h_bias, sizeof(dtype) * hidden_units, cudaMemcpyHostToDevice));             \
    CHECK(cudaMemcpy(d_scale, h_scale, sizeof(dtype) * hidden_units, cudaMemcpyHostToDevice));           \
    DataType type_dtype = getTensorType<dtype>();                                                        \
    DataType type_int = getTensorType<int>();                                                            \
    TensorWrapper<dtype> *decoder_out_tensor = new TensorWrapper<dtype>(Device::GPU,                     \
                                                                        type_dtype,                      \
                                                                        {num_tokens, hidden_units},      \
                                                                        d_decoder_out);                  \
    TensorWrapper<dtype> *residual_tensor = new TensorWrapper<dtype>(Device::GPU,                        \
                                                                     type_dtype,                         \
                                                                     {num_tokens, hidden_units},         \
                                                                     d_residual);                        \
    BaseWeight<dtype> norm;                                                                              \                    
    std::cout << "before launch kernel" << std::endl;                                                    \
    launchFusedAddBiasResidualRMSNorm(residual_tensor,                                                   \
                                      decoder_out_tensor,                                                \
                                      norm,                                                              \
                                      d_scale,                                                           \
                                      eps);                                                              \
    std::cout << "after launch kernel" << std::endl;                                                     \
    std::cout << "cuda memcpy device to host" << std::endl;                                              \
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(dtype) * total_size, cudaMemcpyDeviceToHost));   \
    float* CPU_residual = (float *)malloc(sizeof(float) * total_size);                                   \
    for (int i = 0; i < total_size; i++)                                                                 \
    {                                                                                                    \
        CPU_residual[i] = 0.0f;                                                                          \
    }                                                                                                    \
    float *CPU_decoder_out = (float *)malloc(sizeof(float) * total_size);                                \
    for (int i = 0; i < total_size; i++)                                                                 \
    {                                                                                                    \
        CPU_decoder_out[i] = 1.0f;                                                                       \
    }                                                                                                    \
    float *CPU_bias = (float *)malloc(sizeof(float) * hidden_units);                                     \
    for (int i = 0; i < hidden_units; i++)                                                               \
    {                                                                                                    \
        CPU_bias[i] = 0.0f;                                                                              \
    }                                                                                                    \
    float *CPU_scale = (float *)malloc(sizeof(float) * hidden_units);                                    \
    for (int i = 0; i < hidden_units; i++)                                                               \
    {                                                                                                    \
        CPU_scale[i] = 1.0f;                                                                             \
    }                                                                                                    \
    CPUfusedresidandRMSNorm(CPU_residual, CPU_decoder_out, CPU_bias,                                     \
                            CPU_scale, eps, hidden_units, num_tokens);                                   \
    bool is_right = CheckResult<dtype>(CPU_decoder_out, decoder_out, total_size);                        \
    std::cout << "before free" << std::endl;                                                             \
    std::cout << "fused rmsnorm and add residual passed" << std::endl;                                   \
    free(h_residual);                                                                                    \
    free(h_decoder_out);                                                                                 \
    free(h_bias);                                                                                        \
    free(h_scale);                                                                                       \
    free(CPU_residual);                                                                                  \
    free(CPU_decoder_out);                                                                               \
    free(CPU_bias);                                                                                      \
    free(CPU_scale);                                                                                     \
    free(decoder_out);                                                                                   \
    cudaFree(d_residual);                                                                                \
    cudaFree(d_decoder_out);                                                                             \
    cudaFree(d_bias);                                                                                    \
    cudaFree(d_scale);

int main(int argc, char *argv[])
{
    const int num_tokens = 2;
    const int hidden_units = 32;
    const int total_size = num_tokens * hidden_units;
    float eps = 0.5f;
    if (argv[1])
    {
        TEST_FUSED_ADD_RMS(half);
    }
    else
    {
        TEST_FUSED_ADD_RMS(float);
    }    
}
