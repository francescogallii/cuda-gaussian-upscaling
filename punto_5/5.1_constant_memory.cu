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

// --- OTTIMIZZAZIONE 1: MEMORIA COSTANTE ---
// Definiamo una dimensione massima per il kernel (es. raggio 2 -> 5x5 = 25 float)
// La Constant Memory è limitata (64KB), ma per 25 float è perfetta.
#define MAX_KERNEL_SIZE 25
__constant__ float c_kernel[MAX_KERNEL_SIZE];

// KERNEL CUDA (Pre-Computed Weights + Naive Memory)

__global__ void upscale_kernel_constant(unsigned char* d_input, unsigned char* d_output, 
                                        int w_in, int h_in, int w_out, int h_out, 
                                        int channels, int radius) {
    
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out < w_out && y_out < h_out) {
        
        float src_x = (x_out + 0.5f) / 2.0f - 0.5f;
        float src_y = (y_out + 0.5f) / 2.0f - 0.5f;

        int center_x = roundf(src_x);
        int center_y = roundf(src_y);

        float total_weight = 0.0f;
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;

        // Indice per scorrere l'array in constant memory
        int k_idx = 0;

        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                
                // --- MODIFICA RISPETTO AL NAIVE ---
                // Invece di calcolare expf(), leggiamo dalla Constant Memory
                float weight = c_kernel[k_idx];
                k_idx++; 
                // ----------------------------------

                int nx = max(0, min(center_x + kx, w_in - 1));
                int ny = max(0, min(center_y + ky, h_in - 1));

                // Accesso alla memoria: ANCORA NAIVE (Global Memory diretta)
                // Non abbiamo toccato questo aspetto per isolare l'ottimizzazione
                int idx = (ny * w_in + nx) * channels;
                
                r_sum += d_input[idx + 0] * weight;
                g_sum += d_input[idx + 1] * weight;
                b_sum += d_input[idx + 2] * weight;
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

// Funzione Host per pre-calcolare i pesi
void precompute_and_upload_kernel(int radius, float sigma) {
    float h_kernel[MAX_KERNEL_SIZE];
    int idx = 0;
    // L'ordine dei cicli DEVE corrispondere a quello nel kernel GPU (ky, poi kx)
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            float dist_sq = kx*kx + ky*ky;
            h_kernel[idx++] = expf(-dist_sq / (2.0f * sigma * sigma));
        }
    }
    // Copia speciale in memoria costante
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel, idx * sizeof(float)));
}

int main() {
    std::string input_path;
    std::cout << "Inserisci percorso immagine: ";
    std::cin >> input_path;

    std::string output_path;
    std::cout << "Inserisci percorso dell'immagine di output: ";
    std::cin >> output_path;

    int w_in, h_in, channels;
    unsigned char* h_img_in = stbi_load(input_path.c_str(), &w_in, &h_in, &channels, 3);
    if (!h_img_in) return -1;

    int w_out = w_in * 2;
    int h_out = h_in * 2;
    size_t in_size = w_in * h_in * 3;
    size_t out_size = w_out * h_out * 3;

    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    CUDA_CHECK(cudaMemcpy(d_input, h_img_in, in_size, cudaMemcpyHostToDevice));

    // --- SETUP COSTANTE ---
    int radius = 1; // Raggio fisso per il test (Kernel 3x3)
    precompute_and_upload_kernel(radius, 0.8f); 

    dim3 blockSize(16, 16);
    dim3 gridSize((w_out + blockSize.x - 1) / blockSize.x, 
                  (h_out + blockSize.y - 1) / blockSize.y);

    // Warmup
    upscale_kernel_constant<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out, 3, radius);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Lancio Kernel Modificato
    upscale_kernel_constant<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out, 3, radius);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tempo Ottimizzazione 1 (Constant Mem): " << milliseconds << " ms" << std::endl;
    
    // Salvataggio e free identici alla versione precedente
    std::vector<unsigned char> h_img_out(out_size);
    CUDA_CHECK(cudaMemcpy(h_img_out.data(), d_output, out_size, cudaMemcpyDeviceToHost));
    stbi_write_png(output_path.c_str(), w_out, h_out, 3, h_img_out.data(), w_out * 3);
    
    cudaFree(d_input); cudaFree(d_output); stbi_image_free(h_img_in);
    return 0;
}