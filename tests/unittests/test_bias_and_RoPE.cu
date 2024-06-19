#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/weights/llama/attention_weights.h"
#include "src/utils/macro.h"
// (RussWong)note: not sure CPU implementation is absolutely right and the GPU kernel is right compared with HF.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel and test its correctness
void CPUfunc(float *q,
             float *k,
             float *v,
             float *QKV,
             const float *qkv_bias,
             const int *padding_offset,
             const int *history_length,
             const int *input_length,
             const int batch_size,
             const int seq_len,
             const int token_num,
             const int head_num,
             const int kv_head_num,
             const int head_size,
             const int rotary_embedding_dim,
             float rotary_embedding_base)
{
    int qbatchstride = seq_len * head_num * head_size;
    int kvbatchstride = seq_len * kv_head_num * head_size;
    for (int b = 0; b < batch_size; b++)
    {
        for (int s = 0; s < seq_len; s++)
        {
            int timestep = history_length[b] + s;
            for (int head = 0; head < head_num; head++)
            {
                for (int d = 0; d < head_size; d++)
                {
                    // q bias
                    q[b * qbatchstride + s * head_num * head_size + head * head_size + d] =
                        QKV[b * qbatchstride + s * head_num * head_size + head * head_size + d];
                }
                // RoPE
                for (int d = 0; d < head_size / 2; d++)
                {
                    float x0 = q[b * qbatchstride + s * head_num * head_size + head * head_size + d];
                    float x1 = q[b * qbatchstride + s * head_num * head_size + head * head_size + d + 64];
                    // refer to https://zhuanlan.zhihu.com/p/647109286, d=0,2,4,dim-1
                    float inv_freq = timestep / powf(rotary_embedding_base, (d * 2) / (float)rotary_embedding_dim);
                    q[b * qbatchstride + s * head_num * head_size + head * head_size + d] =
                        x0 * cos(inv_freq) - x1 * sin(inv_freq);

                    q[b * qbatchstride + s * head_num * head_size + head * head_size + d + 64] =
                        x1 * cos(inv_freq) + x0 * sin(inv_freq);
                }
            }
            for (int head = 0; head < kv_head_num; head++)
            {
                for (int d = 0; d < head_size; d++)
                {
                    // k bias
                    k[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d] =
                        QKV[b * kvbatchstride + s * (head_num + kv_head_num) * head_size + head * head_size + d]; // + qkv_bias[(head_num + kv_head_num)  * head_size + d];
                    v[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d] =
                        QKV[b * kvbatchstride + s * (head_num + kv_head_num * 2) * head_size + head * head_size + d]; // + qkv_bias[(head_num + 2 * kv_head_num)  * head_size + d];
                }
                // RoPE
                for (int d = 0; d < head_size / 2; d++)
                {
                    float x0 = k[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d];
                    float x1 = k[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d + 64];
                    float inv_freq = timestep / powf(rotary_embedding_base, (d * 2) / (float)rotary_embedding_dim);
                    k[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d] =
                        x0 * cos(inv_freq) - x1 * sin(inv_freq);

                    k[b * kvbatchstride + s * kv_head_num * head_size + head * head_size + d + 64] =
                        x1 * cos(inv_freq) + x0 * sin(inv_freq);
                }
            }
        }
    }
}

template <typename T>
bool CheckResult(T *q, T *k, float *hq, float *hk,
                 const int q_size, const int k_size)
{
    for (int i = 0; i < q_size; i++)
    {
        if (fabs((float)q[i] - hq[i]) > 1e-6)
        {
            printf("the %dth q is wrong, q = %f, hq = %f\n", i, q[i], hq[i]);
            return false;
        }
    }
    for (int i = 0; i < k_size; i++)
    {
        if (fabs((float)k[i] - hk[i]) > 1e-6)
        {
            printf("the %dth k is wrong, k = %f, hk = %f\n", i, k[i], hk[i]);
            return false;
        }
    }
    return true;
}

#define TEST_ROPE(dtype)                                                                                                                   \
    dtype *q = (dtype *)malloc(sizeof(dtype) * batch_size * seq_len * head_num * head_size);                                               \
    dtype *k = (dtype *)malloc(sizeof(dtype) * batch_size * seq_len * kv_head_num * head_size);                                            \
    dtype *v = (dtype *)malloc(sizeof(dtype) * batch_size * seq_len * kv_head_num * head_size);                                            \
    dtype *QKV = (dtype *)malloc(sizeof(dtype) * token_num * (head_num + 2 * kv_head_num) * head_size);                                    \
    dtype *qkv_bias = (dtype *)malloc(sizeof(dtype) * (head_num + 2 * kv_head_num) * head_size);                                           \
    for (int i = 0; i < token_num * (head_num + 2 * kv_head_num) * head_size; i++)                                                         \
    {                                                                                                                                      \
        QKV[i] = (dtype)32.0;                                                                                                              \
    }                                                                                                                                      \
    for (int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++)                                                                     \
    {                                                                                                                                      \
        qkv_bias[i] = (dtype)2.0;                                                                                                          \
    }                                                                                                                                      \
    for (int i = 0; i < batch_size; i++)                                                                                                   \
    {                                                                                                                                      \
        input_length[i] = 7;                                                                                                               \
        history_length[i] = 0;                                                                                                             \
    }                                                                                                                                      \
    for (int i = 0; i < batch_size * seq_len; i++)                                                                                         \
    {                                                                                                                                      \
        padding_offset[i] = 0;                                                                                                             \
    }                                                                                                                                      \
    int *dpadding_offset;                                                                                                                  \
    int *dhistory_length;                                                                                                                  \
    int *dinput_length;                                                                                                                    \
    dtype *dq;                                                                                                                             \
    dtype *dk;                                                                                                                             \
    dtype *dv;                                                                                                                             \
    dtype *dQKV;                                                                                                                           \
    dtype *dqkv_bias;                                                                                                                      \
    cudaMalloc((void **)&dpadding_offset, sizeof(int) * batch_size * seq_len);                                                             \
    cudaMalloc((void **)&dhistory_length, sizeof(int) * batch_size);                                                                       \
    cudaMalloc((void **)&dinput_length, sizeof(int) * batch_size);                                                                         \
    cudaMalloc((void **)&dq, sizeof(dtype) * batch_size * seq_len * head_num * head_size);                                                 \
    cudaMalloc((void **)&dk, sizeof(dtype) * batch_size * seq_len * kv_head_num * head_size);                                              \
    cudaMalloc((void **)&dv, sizeof(dtype) * batch_size * seq_len * kv_head_num * head_size);                                              \
    cudaMalloc((void **)&dQKV, sizeof(dtype) * token_num * (head_num + 2 * kv_head_num) * head_size);                                      \
    cudaMalloc((void **)&dqkv_bias, sizeof(dtype) * (head_num + 2 * kv_head_num) * head_size);                                             \
    cudaMemcpy(dinput_length, input_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);                                             \
    cudaMemcpy(dhistory_length, history_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);                                         \
    cudaMemcpy(dpadding_offset, padding_offset, sizeof(int) * seq_len * batch_size, cudaMemcpyHostToDevice);                               \
    cudaMemcpy(dQKV, QKV, sizeof(dtype) * token_num * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice);                   \
    cudaMemcpy(dqkv_bias, qkv_bias, sizeof(dtype) * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice);                     \
    DataType type = getTensorType<dtype>();                                                                                                \
    TensorWrapper<dtype> *q_buf = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, head_num, seq_len, head_size}, dq);             \
    TensorWrapper<dtype> *k_buf = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, dk);          \
    TensorWrapper<dtype> *v_buf = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, dv);          \
    TensorWrapper<dtype> *QKV_buf = new TensorWrapper<dtype>(Device::GPU, type, {token_num, head_num + 2 * kv_head_num, head_size}, dQKV); \
    LLaMAattentionWeights<dtype> attn_weights;                                                                                             \
    attn_weights.qkv.bias = dqkv_bias;                                                                                                     \
    DataType type_int = getTensorType<int>();                                                                                              \
    TensorWrapper<int> *input_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dinput_length);                     \
    TensorWrapper<int> *history_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dhistory_length);                 \
    TensorWrapper<int> *padding_offset_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, seq_len}, dpadding_offset);        \
    LLaMAAttentionStaticParams params;                                                                                                     \
    params.rotary_embedding_dim = rotary_embedding_dim;                                                                                    \
    params.rotary_embedding_base = rotary_embedding_base;                                                                                  \
    params.max_position_embeddings = max_position_embeddings;                                                                              \
    params.use_dynamic_ntk = false;                                                                                                        \
    std::cout << "before launch kernel" << std::endl;                                                                                      \
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf,                                                                                           \
                                          k_buf,                                                                                           \
                                          v_buf,                                                                                           \
                                          QKV_buf,                                                                                         \
                                          attn_weights.qkv,                                                                                \
                                          padding_offset_buf,                                                                              \
                                          history_length_buf,                                                                              \
                                          input_length_buf,                                                                                \
                                          params);                                                                                         \
    std::cout << "after launch kernel" << std::endl;                                                                                       \
    std::cout << "cuda memcpy device to host" << std::endl;                                                                                \
    CHECK(cudaMemcpy(q, dq, sizeof(dtype) * batch_size * seq_len * head_num * head_size, cudaMemcpyDeviceToHost));                         \
    CHECK(cudaMemcpy(k, dk, sizeof(dtype) * batch_size * seq_len * kv_head_num * head_size, cudaMemcpyDeviceToHost));                      \
    std::cout << "after memcpyd2h, dq[0] = " << (float)q[0] << std::endl;                                                                         \
    std::cout << "before CPU function" << std::endl;                                                                                       \
    float *hq = (float *)malloc(sizeof(float) * batch_size * seq_len * head_num * head_size);                                              \
    float *hk = (float *)malloc(sizeof(float) * batch_size * seq_len * kv_head_num * head_size);                                           \
    float *hv = (float *)malloc(sizeof(float) * batch_size * seq_len * kv_head_num * head_size);                                           \
    float *hQKV = (float *)malloc(sizeof(float) * token_num * (head_num + 2 * kv_head_num) * head_size);                                   \
    float *hqkv_bias = (float *)malloc(sizeof(float) * (head_num + 2 * kv_head_num) * head_size);                                          \
    for (int i = 0; i < token_num * (head_num + 2 * kv_head_num) * head_size; i++)                                                         \
    {                                                                                                                                      \
        hQKV[i] = 32.0f;                                                                                                                   \
    }                                                                                                                                      \
    for (int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++)                                                                     \
    {                                                                                                                                      \
        hqkv_bias[i] = 2.0f;                                                                                                               \
    }                                                                                                                                      \
    CPUfunc(hq,                                                                                                                            \
            hk,                                                                                                                            \
            hv,                                                                                                                            \
            hQKV,                                                                                                                          \
            hqkv_bias,                                                                                                                     \
            padding_offset,                                                                                                                \
            history_length,                                                                                                                \
            input_length,                                                                                                                  \
            batch_size,                                                                                                                    \
            seq_len,                                                                                                                       \
            token_num,                                                                                                                     \
            head_num,                                                                                                                      \
            kv_head_num,                                                                                                                   \
            head_size,                                                                                                                     \
            rotary_embedding_dim,                                                                                                          \
            rotary_embedding_base);                                                                                                        \
    std::cout << "after CPU function" << std::endl;                                                                                        \
    bool is_right = CheckResult<dtype>(q, k, hq, hk,                                                                                       \
                                       batch_size * seq_len * head_num * head_size,                                                        \
                                       batch_size * seq_len * kv_head_num * head_size);                                                    \
    std::cout << "before free" << std::endl;                                                                                               \
    std::cout << "passed" << std::endl;                                                                                                    \
    free(q);                                                                                                                               \
    free(k);                                                                                                                               \
    free(v);                                                                                                                               \
    free(QKV);                                                                                                                             \
    free(qkv_bias);                                                                                                                        \
    free(padding_offset);                                                                                                                  \
    free(history_length);                                                                                                                  \
    free(input_length);                                                                                                                    \
    free(hq);                                                                                                                              \
    free(hk);                                                                                                                              \
    free(hv);                                                                                                                              \
    free(hQKV);                                                                                                                            \
    free(hqkv_bias);                                                                                                                       \
    cudaFree(dq);                                                                                                                          \
    cudaFree(dk);                                                                                                                          \
    cudaFree(dv);                                                                                                                          \
    cudaFree(dQKV);                                                                                                                        \
    cudaFree(dqkv_bias);                                                                                                                   \
    cudaFree(dpadding_offset);                                                                                                             \
    cudaFree(dhistory_length);                                                                                                             \
    cudaFree(dinput_length);

// (RussWong)note:
// `./biasRope` to test fp32 GPU kernel
// half GPU kernel test is not implemented now
int main(int argc, char *argv[])
{
    const int batch_size = 1;
    const int seq_len = 32;
    int *padding_offset = (int *)malloc(sizeof(int) * batch_size * seq_len);
    int *history_length = (int *)malloc(sizeof(int) * batch_size);
    int *input_length = (int *)malloc(sizeof(int) * batch_size);
    const int token_num = batch_size * seq_len;
    const int head_num = 32;
    const int kv_head_num = 32;
    const int head_size = 128;
    const int rotary_embedding_dim = 128;
    const int rotary_embedding_base = 10000;
    const int max_position_embeddings = 2048;
    if (argv[1])
    {
        TEST_ROPE(half);
    } else {
        TEST_ROPE(float);
    }
}
