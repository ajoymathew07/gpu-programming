#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.
#define STATE_NONE 0
#define STATE_NEAR 1
#define STATE_FAR 2

#define CNT_NEAR_NEXT 0
#define CNT_SETTLED 1
#define CNT_FAR 2
#define CNT_TOTAL 3


struct Graph{
    int N;
    int E;
    int *offsets;
    int *neighs;
    int *weights;
};

struct Workspace{
    int* tent;
    int* state;
    int* near_cur;
    int* near_next;
    int* mark;
    int* settled;
    int* far_queue;
    int* far_keys;
    int* counters;
    int window_id;
};

static inline int num_blocks(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
// ============================================================================
// CUDA KERNELS
// ============================================================================
__global__ void init_source_kernel(int *tent, int* state, int* near_cur, int N, int source) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= N) return;

    if(v == source){
        tent[v] = 0;
        state[v] = STATE_NEAR;
        near_cur[0] = v;
    }else{
        
        tent[v] = INF;
        state[v] = STATE_NONE;
    }
}

__global__ void relax_light_kernel(Graph g, Workspace ws, int delta, int near_size, int cutoff, int light_delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= near_size) return;

    int u = ws.near_cur[i];
    atomicExch(&ws.state[u], STATE_NONE);
    __threadfence();

    if(atomicExch(&ws.mark[u], ws.window_id) != ws.window_id){
        int pos = atomicAdd(&ws.counters[CNT_SETTLED], 1);
        ws.settled[pos] = u;
    }

    int du = ((volatile int*) ws.tent)[u];
    int start = g.offsets[u];
    int end = g.offsets[u + 1];

    for(int e = start; e < end; ++e){
        int w = g.weights[e];
        if(w > light_delta) continue;

        int v = g.neighs[e];
        int nd = du + w;

        if(nd >= ws.tent[v]) continue;

        if(nd < atomicMin(&ws.tent[v], nd)){

            if(nd <= cutoff){
                if(atomicExch(&ws.state[v], STATE_NEAR) != STATE_NEAR){
                    int pos = atomicAdd(&ws.counters[CNT_NEAR_NEXT], 1);
                    ws.near_next[pos] = v;
                }
            }else{
                if(atomicCAS(&ws.state[v], STATE_NONE, STATE_FAR) == STATE_NONE){
                    int pos = atomicAdd(&ws.counters[CNT_FAR], 1);
                    ws.far_queue[pos] = v;
                   // ws.far_keys[pos] = nd;  can change, so better to compute at the end
                }
            }

        }
    }
}

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(const Graph& g, Workspace&ws, int source, int delta_mode, int K, std::ofstream& outfile)
{
  init_source_kernel<<<num_blocks(g.N), BLOCK_SIZE>>>(ws.tent, ws.state, ws.near_cur, g.N, source);
  cudaMemset(ws.counters, 0, CNT_TOTAL * sizeof(int));

  int near_size = 1;
  int light_delta = (delta_mode >= 0) ? delta_mode : 0;
  int cutoff = light_delta;

  ws.window_id++;

  while(near_size > 0){
    relax_light_kernel<<<num_blocks(near_size), BLOCK_SIZE>>>(g, ws, light_delta, near_size, cutoff, light_delta);
    cudaMemcpy(&near_size, ws.counters + CNT_NEAR_NEXT, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(ws.counters + CNT_NEAR_NEXT, 0, sizeof(int));
    std::swap(ws.near_cur, ws.near_next);
  }

  cudaDeviceSynchronize();
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
    {
        infile >> sources[i];
    }

    int *offsets = new int[ N + 1 ] { 0 };
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (size_t)(N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, (size_t)E * sizeof(int));
    cudaMalloc(&d_weights, (size_t)E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (size_t)(N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, (size_t)E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, (size_t)E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance arry. tent means tentative distance
    int *d_tent;
    cudaMalloc(&d_tent, (size_t)N * sizeof(int));

    thrust::device_vector<int> d_state(N);
    thrust::device_vector<int> d_near_cur(N);
    thrust::device_vector<int> d_near_next(N);
    thrust::device_vector<int> d_settled(N);
    thrust::device_vector<int> d_mark(N, -1);
    thrust::device_vector<int> d_far_queue(N);
    thrust::device_vector<int> d_far_keys(N);
    thrust::device_vector<int> d_counters(CNT_TOTAL, 0);

    Graph g;
    g.N = N;
    g.E = E;
    g.offsets = d_offsets;
    g.neighs = d_neighs;
    g.weights = d_weights;

    Workspace ws;
    ws.tent = d_tent;
    ws.state = thrust::raw_pointer_cast(d_state.data());
    ws.near_cur = thrust::raw_pointer_cast(d_near_cur.data());
    ws.near_next = thrust::raw_pointer_cast(d_near_next.data());
    ws.mark = thrust::raw_pointer_cast(d_mark.data());
    ws.settled = thrust::raw_pointer_cast(d_settled.data());
    ws.far_queue = thrust::raw_pointer_cast(d_far_queue.data());
    ws.far_keys = thrust::raw_pointer_cast(d_far_keys.data());
    ws.counters = thrust::raw_pointer_cast(d_counters.data());
    ws.window_id = 0;


    /*
    ToDo








    */

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        run_delta_stepping_single_source(
            g, ws, source, delta_mode, K, outfile
        );

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    delete[] offsets;
    delete[] neighs;
    delete[] weights;

    delete[] h_tent;
    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);

    outfile.close();
    return 0;
}