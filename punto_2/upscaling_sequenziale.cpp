#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>

// Definizioni necessarie per le librerie stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ==========================================
// CONFIGURAZIONE
// ==========================================
const int SCALE_FACTOR = 2;
const float SIGMA = 0.8f; 

// ==========================================
// STRUTTURE DATI
// ==========================================

struct Image {
    int width;
    int height;
    int channels; // 3 per RGB, 4 per RGBA
    std::vector<unsigned char> data; // Dati raw (0-255) linearizzati

    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}

    // Accede al valore di un pixel specifico in un canale specifico
    // Gestisce il "Clamping" (se usciamo dai bordi, ripete l'ultimo pixel valido)
    unsigned char get_pixel_safe(int x, int y, int c) const {
        int clamped_x = std::max(0, std::min(x, width - 1));
        int clamped_y = std::max(0, std::min(y, height - 1));
        
        // Indice = (y * width + x) * channels + canale
        int index = (clamped_y * width + clamped_x) * channels + c;
        return data[index];
    }

    void set_pixel(int x, int y, int c, unsigned char value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int index = (y * width + x) * channels + c;
            data[index] = value;
        }
    }
};

// ==========================================
// LOGICA DI CALCOLO
// ==========================================

int get_dynamic_radius(int width, int height) {
    int min_dim = std::min(width, height);
    //if (min_dim < 50) return 1;    
    //if (min_dim < 500) return 2;   
    //return 3;
    return 1;                      
}

float calculate_weight(float dx, float dy) {
    float distance_sq = dx * dx + dy * dy;
    return std::exp(-distance_sq / (2 * SIGMA * SIGMA));
}

Image upscale_image(const Image& input) {
    int new_width = input.width * SCALE_FACTOR;
    int new_height = input.height * SCALE_FACTOR;
    
    // Creiamo l'immagine di output con lo stesso numero di canali dell'input
    Image output(new_width, new_height, input.channels);

    int radius = get_dynamic_radius(input.width, input.height);
    std::cout << ">> Inizio Upscaling...\n";
    std::cout << "   Dimensione Input: " << input.width << "x" << input.height << "\n";
    std::cout << "   Dimensione Output: " << new_width << "x" << new_height << "\n";
    std::cout << "   Raggio Kernel: " << radius << "\n";

    // Ciclo sui pixel di OUTPUT
    for (int y_out = 0; y_out < new_height; ++y_out) {
        // Piccola barra di progresso testuale
        if (y_out % 100 == 0) std::cout << "\rElaborazione riga: " << y_out << "/" << new_height << std::flush;

        for (int x_out = 0; x_out < new_width; ++x_out) {
            
            // Mappatura inversa
            float src_x = (x_out + 0.5f) / SCALE_FACTOR - 0.5f;
            float src_y = (y_out + 0.5f) / SCALE_FACTOR - 0.5f;

            int center_x = std::round(src_x);
            int center_y = std::round(src_y);

            // Accumulatori per ogni canale (R, G, B...)
            // Usiamo vector float per accumulare somme pesate precise
            std::vector<float> channel_sums(input.channels, 0.0f);
            float total_weight = 0.0f;

            // Kernel Loop
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    
                    int neighbor_x = center_x + kx;
                    int neighbor_y = center_y + ky;
                    
                    float dx = src_x - neighbor_x;
                    float dy = src_y - neighbor_y;
                    float weight = calculate_weight(dx, dy);

                    // Per ogni canale del pixel vicino
                    for (int c = 0; c < input.channels; ++c) {
                        unsigned char val = input.get_pixel_safe(neighbor_x, neighbor_y, c);
                        channel_sums[c] += val * weight;
                    }
                    total_weight += weight;
                }
            }

            // Normalizzazione e salvataggio
            if (total_weight > 0.0f) {
                for (int c = 0; c < input.channels; ++c) {
                    // Cast a int per pulizia, poi a unsigned char
                    int final_val = (int)(channel_sums[c] / total_weight);
                    // Clamp finale per sicurezza (0-255)
                    final_val = std::max(0, std::min(255, final_val));
                    output.set_pixel(x_out, y_out, c, (unsigned char)final_val);
                }
            }
        }
    }
    std::cout << "\n>> Upscaling completato.\n";
    return output;
}

// ==========================================
// MAIN
// ==========================================

int main() {
    std::string input_path;
    std::string output_name;

    std::cout << "==========================================\n";
    std::cout << "    IMAGE UPSCALER (Gaussian Kernel)      \n";
    std::cout << "==========================================\n";

    // Input Utente
    std::cout << "Inserisci il percorso dell'immagine (es. foto.jpg): ";
    std::cin >> input_path;

    // Caricamento Immagine
    int w, h, ch;
    // stbi_load restituisce un puntatore a unsigned char con i dati raw
    unsigned char* img_data = stbi_load(input_path.c_str(), &w, &h, &ch, 0);

    if (img_data == nullptr) {
        std::cerr << "ERRORE: Impossibile caricare l'immagine. Controlla il percorso.\n";
        return 1;
    }

    // Copiamo i dati nella nostra struttura C++
    Image input_img(w, h, ch);
    // std::copy è efficiente per copiare blocchi di memoria
    std::copy(img_data, img_data + (w * h * ch), input_img.data.begin());
    
    // Liberiamo la memoria allocata da stb_image (non ci serve più il puntatore raw)
    stbi_image_free(img_data);

    // Campionamento tempo iniziale 
    auto start = std::chrono::high_resolution_clock::now();
    
    // Elaborazione
    Image output_img = upscale_image(input_img);
    
    // Campionamento tempo finale
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "\nTempo impiegato dalla CPU: " << duration.count() << " ms\n";

    // Output Utente
    std::cout << "Inserisci il nome del file di output (senza estensione): ";
    std::cin >> output_name;
    std::string output_filename = output_name + ".png"; // Salviamo sempre in PNG per qualità

    // Salvataggio
    // stbi_write_png vuole: filename, w, h, canali, puntatore ai dati, stride (bytes per riga)
    int stride = output_img.width * output_img.channels;
    int success = stbi_write_png(output_filename.c_str(), output_img.width, output_img.height, output_img.channels, output_img.data.data(), stride);

    if (success) {
        std::cout << ">> Successo! Immagine salvata come: " << output_filename << "\n";
    } else {
        std::cerr << ">> ERRORE durante il salvataggio del file.\n";
    }

    return 0;
}