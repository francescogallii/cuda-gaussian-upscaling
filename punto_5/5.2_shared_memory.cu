#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// --- CONFIGURAZIONE ---
#define MAX_KERNEL_SIZE 25
__constant__ float c_kernel[MAX_KERNEL_SIZE];

// Parametri Tiling
#define RADIUS 1
#define BLOCK_W 16
#define BLOCK_H 16

// La tile deve contenere i pixel del blocco PIÙ il bordo (Halo)
// Tile Width = 16 + 1 (sx) + 1 (dx) = 18
#define TILE_W (BLOCK_W + 2 * RADIUS)
#define TILE_H (BLOCK_H + 2 * RADIUS)

// ========================================================================
// KERNEL CUDA (SHARED MEMORY + CONSTANT MEMORY)
// ========================================================================
__global__ void upscale_kernel_shared(unsigned char* d_input, unsigned char* d_output, 
                                      int w_in, int h_in, int w_out, int h_out, 
                                      int channels) {
    
    // Allocazione statica Shared Memory: ~972 bytes
    // (18 * 18 * 3 bytes). Piccolissima, entra facile in L1.
    __shared__ unsigned char smem[TILE_H][TILE_W][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Coordinate globali output
    int x_out = blockIdx.x * blockDim.x + tx;
    int y_out = blockIdx.y * blockDim.y + ty;

    // --- FASE 1: CARICAMENTO COLLABORATIVO IN SHARED MEMORY ---
    // Ogni thread del blocco aiuta a caricare un pezzetto della "mattonella" di input.
    // La mattonella è più grande del blocco (18x18 vs 16x16), quindi serve un ciclo.
    
    // Calcolo coordinate base della tile nell'input
    // (Il pixel 0,0 della tile corrisponde al pixel input necessario per il thread 0,0 meno il raggio)
    int global_base_x = (blockIdx.x * blockDim.x) / 2 - RADIUS;
    int global_base_y = (blockIdx.y * blockDim.y) / 2 - RADIUS;

    // Indici linearizzati per distribuire il lavoro di carico
    int tid = ty * blockDim.x + tx;      // ID thread nel blocco (0..255)
    int num_threads = blockDim.x * blockDim.y; // 256
    int num_elements = TILE_W * TILE_H;        // 324 (18*18)

    // Ciclo di caricamento: i thread coprono tutti i 324 elementi
    for (int i = tid; i < num_elements; i += num_threads) {
        int local_y = i / TILE_W;
        int local_x = i % TILE_W;

        int global_x = global_base_x + local_x;
        int global_y = global_base_y + local_y;

        // Boundary Check (Clamping)
        global_x = max(0, min(global_x, w_in - 1));
        global_y = max(0, min(global_y, h_in - 1));

        int idx = (global_y * w_in + global_x) * channels;
        
        // Caricamento in Shared Memory
        smem[local_y][local_x][0] = d_input[idx + 0];
        smem[local_y][local_x][1] = d_input[idx + 1];
        smem[local_y][local_x][2] = d_input[idx + 2];
    }

    // Barriera: Aspettiamo che TUTTI abbiano finito di caricare
    __syncthreads();

    // --- FASE 2: CALCOLO (Tutto in Shared Memory) ---
    
    if (x_out < w_out && y_out < h_out) {
        
        // Mappatura inversa precisa (sub-pixel)
        float src_x = (x_out + 0.5f) / 2.0f - 0.5f;
        float src_y = (y_out + 0.5f) / 2.0f - 0.5f;

        // Coordinate relative alla Tile
        float rel_x = src_x - global_base_x;
        float rel_y = src_y - global_base_y;

        int center_x = roundf(rel_x);
        int center_y = roundf(rel_y);

        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float total_weight = 0.0f;
        int k_idx = 0;

        // Convoluzione usando SOLO dati dalla Shared Memory
        for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
            for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                
                float weight = c_kernel[k_idx++]; // Constant Mem

                int smem_x = center_x + kx;
                int smem_y = center_y + ky;

                r_sum += smem[smem_y][smem_x][0] * weight; // Shared Mem
                g_sum += smem[smem_y][smem_x][1] * weight;
                b_sum += smem[smem_y][smem_x][2] * weight;
                total_weight += weight;
            }
        }

        if (total_weight > 0.0f) {
            int out_idx = (y_out * w_out + x_out) * channels;
            d_output[out_idx + 0] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r_sum / total_weight));
            d_output[out_idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, g_sum / total_weight));
            d_output[out_idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b_sum / total_weight));
        }
    }
}

// Funzione Host pre-computazione (Ottimizzata come discusso)
void precompute_and_upload_kernel(int radius, float sigma) {
    float h_kernel[MAX_KERNEL_SIZE];
    int idx = 0;
    float sigma_sq_2 = 2.0f * sigma * sigma; // Calcolato fuori dal loop
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            float dist_sq = kx*kx + ky*ky;
            h_kernel[idx++] = expf(-dist_sq / sigma_sq_2);
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel, idx * sizeof(float)));
}

int main(int argc, char** argv) {
    
    std::string input_path;
    std::cout << "Inserisci percorso immagine: ";
    std::cin >> input_path;

    std::string output_path;
    std::cout << "Inserisci percorso dell'immagine di output: ";
    std::cin >> output_path;

    int w_in, h_in, channels;
    unsigned char* h_img_in = stbi_load(input_path.c_str(), &w_in, &h_in, &channels, 3);
    if (!h_img_in) {
        std::cerr << "ERRORE: Immagine non trovata!" << std::endl;
        return -1;
    }

    int w_out = w_in * 2;
    int h_out = h_in * 2;
    size_t in_size = w_in * h_in * 3;
    size_t out_size = w_out * h_out * 3;

    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    CUDA_CHECK(cudaMemcpy(d_input, h_img_in, in_size, cudaMemcpyHostToDevice));

    precompute_and_upload_kernel(RADIUS, 0.8f);

    dim3 blockSize(BLOCK_W, BLOCK_H);
    dim3 gridSize((w_out + blockSize.x - 1) / blockSize.x, 
                  (h_out + blockSize.y - 1) / blockSize.y);

    // Warmup
    upscale_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out, 3);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Lancio Kernel
    upscale_kernel_shared<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out, 3);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tempo Ottimizzazione 2 (Shared Mem): " << milliseconds << " ms" << std::endl;

    // Download e Salvataggio
    std::vector<unsigned char> h_img_out(out_size);
    CUDA_CHECK(cudaMemcpy(h_img_out.data(), d_output, out_size, cudaMemcpyDeviceToHost));
    stbi_write_png(output_path.c_str(), w_out, h_out, 3, h_img_out.data(), w_out * 3);
    
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_img_in);
    return 0;
}