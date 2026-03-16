
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(tensor) {\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory"); \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous"); }
void CHECK_KERNEL() {auto error = cudaGetLastError(); TORCH_CHECK( error == cudaSuccess, cudaGetErrorString(error));}


template < typename scalar_t  >
__global__ void rope_2d_cuda_kernel(
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> tokens,
        const int64_t* __restrict__ pos,
        const float base,
        const float fwd )
{
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);

    extern __shared__ float shared[];
    float* shared_inv_freq = shared + D;

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    const int Q = D / 4;

    if (threadIdx.x < Q)
        shared_inv_freq[threadIdx.x] = fwd / powf(base, threadIdx.x/float(Q));
    __syncthreads();

    const int X = threadIdx.x < D/2 ? 0 : 1;
    const int m = (X*D/2) + (threadIdx.x % Q);   // index of u_Y or u_X

    const float freq = pos[blockIdx.x*2+X] * shared_inv_freq[threadIdx.x % Q];
    const float cos = cosf(freq);
    const float sin = sinf(freq);

    for (int h = 0; h < H; h++)
    {
        shared[threadIdx.x] = tokens[b][n][h][threadIdx.x];
        __syncthreads();

        const float u = shared[m];
        const float v = shared[m+Q];

        if ((threadIdx.x % (D/2)) < Q)
            tokens[b][n][h][threadIdx.x] = u*cos - v*sin;
        else
            tokens[b][n][h][threadIdx.x] = v*cos + u*sin;
    }
}

void rope_2d_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd )
{
    const int B = tokens.size(0); // batch size
    const int N = tokens.size(1); // sequence length
    const int H = tokens.size(2); // number of heads
    const int D = tokens.size(3); // dimension per head

    TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D, "tokens are not contiguous");
    TORCH_CHECK(pos.is_contiguous(), "positions are not contiguous");
    TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == 2, "bad pos.shape");
    TORCH_CHECK(D % 4 == 0, "token dim must be multiple of 4");

    const int THREADS_PER_BLOCK = D;
    const int N_BLOCKS = B * N; // each block takes care of H*D values
    const int SHARED_MEM = sizeof(float) * (D + D/4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
        rope_2d_cuda_kernel<scalar_t> <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>> (
            tokens.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            pos.data_ptr<int64_t>(),
            base, fwd); //, N, H, D );
    }));
}
