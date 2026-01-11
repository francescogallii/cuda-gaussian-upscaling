#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Macro per il controllo degli errori CUDA
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// ========================================================================
// KERNEL CUDA (NAÏVE)
// ========================================================================
__global__ void upscale_kernel_naive(unsigned char* d_input, unsigned char* d_output, 
                                     int w_in, int h_in, int w_out, int h_out, 
                                     int channels, int radius, float sigma) {
    
    // Calcolo coordinate globali del pixel di output
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check per la griglia di output
    if (x_out < w_out && y_out < h_out) {
        
        // Mappatura inversa verso l'immagine di input (Scale Factor 2)
        float src_x = (x_out + 0.5f) / 2.0f - 0.5f;
        float src_y = (y_out + 0.5f) / 2.0f - 0.5f;

        int center_x = roundf(src_x);
        int center_y = roundf(src_y);

        float total_weight = 0.0f;
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;

        // Loop del Kernel Gaussiano (Accesso diretto a Global Memory)
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                
                // Clamping per i bordi dell'input
                int nx = max(0, min(center_x + kx, w_in - 1));
                int ny = max(0, min(center_y + ky, h_in - 1));

                float dx = src_x - (center_x + kx);
                float dy = src_y - (center_y + ky);
                
                // Calcolo peso
                float weight = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));

                // Lettura RGB e accumulo pesato
                r_sum += d_input[(ny * w_in + nx) * channels + 0] * weight;
                g_sum += d_input[(ny * w_in + nx) * channels + 1] * weight;
                b_sum += d_input[(ny * w_in + nx) * channels + 2] * weight;
                total_weight += weight;
            }
        }

        // Normalizzazione e scrittura nell'immagine di output
        if (total_weight > 0.0f) {
            d_output[(y_out * w_out + x_out) * channels + 0] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r_sum / total_weight));
            d_output[(y_out * w_out + x_out) * channels + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, g_sum / total_weight));
            d_output[(y_out * w_out + x_out) * channels + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b_sum / total_weight));
        }
    }
}

// ========================================================================
// MAIN (HOST CODE)
// ========================================================================
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
    size_t in_size = w_in * h_in * 3 * sizeof(unsigned char);
    size_t out_size = w_out * h_out * 3 * sizeof(unsigned char);

    // 1. Allocazione memoria sulla GPU (Device)
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));

    // 2. Copia Host -> Device
    CUDA_CHECK(cudaMemcpy(d_input, h_img_in, in_size, cudaMemcpyHostToDevice));

    // 3. Configurazione Griglia e Blocchi (Passo 3)
    dim3 blockSize(16, 16);
    dim3 gridSize((w_out + blockSize.x - 1) / blockSize.x, 
                  (h_out + blockSize.y - 1) / blockSize.y);

    // 4. Lancio Kernel e misurazione tempo con eventi CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    upscale_kernel_naive<<<gridSize, blockSize>>>(d_input, d_output, w_in, h_in, w_out, h_out, 3, 1, 0.8f);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tempo Kernel Naive: " << milliseconds << " ms" << std::endl;

    // 5. Copia Device -> Host
    std::vector<unsigned char> h_img_out(w_out * h_out * 3);
    CUDA_CHECK(cudaMemcpy(h_img_out.data(), d_output, out_size, cudaMemcpyDeviceToHost));

    // 6. Salvataggio e pulizia
    stbi_write_png(output_path.c_str(), w_out, h_out, 3, h_img_out.data(), w_out * 3);
    
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_img_in);

    return 0;
}