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

#define RADIUS 1
#define BLOCK_W 16
#define BLOCK_H 16

// Tile Width ora gestisce PIXEL interi, non byte sparsi
#define TILE_W (BLOCK_W + 2 * RADIUS)
#define TILE_H (BLOCK_H + 2 * RADIUS)

// ========================================================================
// KERNEL CUDA (VECTORIZED - RGBA)
// ========================================================================
// Nota: Usiamo uchar4* per input e output. Leggiamo 4 byte alla volta.
__global__ void upscale_kernel_vectorized(uchar4* d_input, uchar4* d_output, 
                                          int w_in, int h_in, int w_out, int h_out) {
    
    // Shared Memory ora è un array di uchar4 (molto più pulito)
    // Dimensione: 18x18 uchar4 ~= 1.2 KB
    __shared__ uchar4 smem[TILE_H][TILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Coordinate globali output
    int x_out = blockIdx.x * blockDim.x + tx;
    int y_out = blockIdx.y * blockDim.y + ty;

    // --- FASE 1: CARICAMENTO VETTORIALE (COALESCED) ---
    // Ogni thread carica pixel uchar4. Essendo 4 byte (32 bit), l'accesso è 
    // nativamente allineato e perfettamente coalesced.
    
    int global_base_x = (blockIdx.x * blockDim.x) / 2 - RADIUS;
    int global_base_y = (blockIdx.y * blockDim.y) / 2 - RADIUS;

    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y; 
    int num_elements = TILE_W * TILE_H;

    for (int i = tid; i < num_elements; i += num_threads) {
        int local_y = i / TILE_W;
        int local_x = i % TILE_W;

        int global_x = global_base_x + local_x;
        int global_y = global_base_y + local_y;

        // Boundary Check (Clamping)
        global_x = max(0, min(global_x, w_in - 1));
        global_y = max(0, min(global_y, h_in - 1));

        // LETTURA VETTORIALE: 1 istruzione per caricare RGBA
        // Indice lineare per array di uchar4
        int idx = global_y * w_in + global_x;
        smem[local_y][local_x] = d_input[idx];
    }

    __syncthreads();

    // --- FASE 2: CALCOLO ---
    
    if (x_out < w_out && y_out < h_out) {
        float src_x = (x_out + 0.5f) / 2.0f - 0.5f;
        float src_y = (y_out + 0.5f) / 2.0f - 0.5f;

        float rel_x = src_x - global_base_x;
        float rel_y = src_y - global_base_y;

        int center_x = roundf(rel_x);
        int center_y = roundf(rel_y);

        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f; 
        // Ignoriamo Alpha (o lo copiamo se servisse trasparenza)
        
        float total_weight = 0.0f;
        int k_idx = 0;

        for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
            for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
                float weight = c_kernel[k_idx++];
                
                // Accesso diretto alla struct uchar4 in shared memory
                uchar4 pixel = smem[center_y + ky][center_x + kx];

                r_sum += pixel.x * weight; // .x = Red
                g_sum += pixel.y * weight; // .y = Green
                b_sum += pixel.z * weight; // .z = Blue
                total_weight += weight;
            }
        }

        if (total_weight > 0.0f) {
            int out_idx = y_out * w_out + x_out;
            uchar4 out_pixel;
            out_pixel.x = (unsigned char)fminf(255.0f, fmaxf(0.0f, r_sum / total_weight));
            out_pixel.y = (unsigned char)fminf(255.0f, fmaxf(0.0f, g_sum / total_weight));
            out_pixel.z = (unsigned char)fminf(255.0f, fmaxf(0.0f, b_sum / total_weight));
            out_pixel.w = 255; // Alpha fisso a 255 (Opaco)
            
            // SCRITTURA VETTORIALE
            d_output[out_idx] = out_pixel;
        }
    }
}

void precompute_and_upload_kernel(int radius, float sigma) {
    float h_kernel[MAX_KERNEL_SIZE];
    int idx = 0;
    float sigma_sq_2 = 2.0f * sigma * sigma;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            h_kernel[idx++] = expf(-(kx*kx + ky*ky) / sigma_sq_2);
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

    int w_in, h_in, channels_in_file;
    // --- TRUCCO HOST: Carichiamo forzatamente a 4 canali (RGBA) ---
    // Il '4' finale dice a stbi_load di convertire tutto in RGBA automaticamente.
    unsigned char* h_img_in = stbi_load(input_path.c_str(), &w_in, &h_in, &channels_in_file, 4);
    
    if (!h_img_in) {
        std::cerr << "ERRORE LOADING" << std::endl;
        return -1;
    }

    int w_out = w_in * 2;
    int h_out = h_in * 2;
    // Nota: Ora la size è * 4, non * 3
    size_t in_size = w_in * h_in * 4; 
    size_t out_size = w_out * h_out * 4;

    // Usiamo uchar4 per i puntatori device per aritmetica dei puntatori corretta
    uchar4 *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    // Copia Host -> Device (tutto il blocco RGBA)
    CUDA_CHECK(cudaMemcpy(d_input, h_img_in, in_size, cudaMemcpyHostToDevice));

    precompute_and_upload_kernel(RADIUS, 0.8f);

    dim3 blockSize(BLOCK_W, BLOCK_H);
    dim3 gridSize((w_out + blockSize.x - 1) / blockSize.x, 
                  (h_out + blockSize.y - 1) / blockSize.y);

    // Warmup
    upscale_kernel_vectorized<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    upscale_kernel_vectorized<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tempo Ottimizzazione 3 (Vectorized RGBA): " << milliseconds << " ms" << std::endl;

    // Salvataggio
    std::vector<unsigned char> h_img_out(out_size);
    CUDA_CHECK(cudaMemcpy(h_img_out.data(), d_output, out_size, cudaMemcpyDeviceToHost));
    // Salviamo come PNG a 4 canali
    stbi_write_png(output_path.c_str(), w_out, h_out, 4, h_img_out.data(), w_out * 4);
    
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_img_in);
    return 0;
}