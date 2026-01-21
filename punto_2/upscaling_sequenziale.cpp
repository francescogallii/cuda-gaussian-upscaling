#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>

// STB Libraries: Header-only libraries per leggere/scrivere immagini (JPG, PNG, ecc.)
// Definiamo IMPLEMENTATION per includere il codice sorgente qui.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- CONFIGURAZIONE ---

const int SCALE_FACTOR = 2;
const float SIGMA = 0.8f; // Bilanciamento nitidezza/sfocatura

// STRUTTURE DATI

struct Image {
    int width;
    int height;
    int channels; // Gestisce dinamicamente RGB (3) o RGBA (4)
    std::vector<unsigned char> data; // Byte raw (0-255) linearizzati

    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}

    // Accesso sicuro ai pixel con supporto multicanale.
    // Indexing: (riga * larghezza + colonna) * canali + offset_canale
    unsigned char get_pixel_safe(int x, int y, int c) const {
        // Clamping: se le coordinate escono dai bordi, ripetiamo l'ultimo pixel valido
        int clamped_x = std::max(0, std::min(x, width - 1));
        int clamped_y = std::max(0, std::min(y, height - 1));
        
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


// Determina la dimensione del kernel. 
// Attualmente fissato a raggio 1 (kernel 3x3) per semplicità in questo test.
int get_dynamic_radius(int width, int height) {
    // (Logica dinamica disabilitata per testare un kernel fisso leggero)
    return 1;                      
}

float calculate_weight(float dx, float dy) {
    float distance_sq = dx * dx + dy * dy;
    return std::exp(-distance_sq / (2 * SIGMA * SIGMA));
}

Image upscale_image(const Image& input) {
    int new_width = input.width * SCALE_FACTOR;
    int new_height = input.height * SCALE_FACTOR;
    
    // Output eredita lo stesso numero di canali dell'input
    Image output(new_width, new_height, input.channels);

    int radius = get_dynamic_radius(input.width, input.height);
    std::cout << ">> Elaborazione: Input " << input.width << "x" << input.height 
              << " -> Output " << new_width << "x" << new_height << " (Kernel r=" << radius << ")\n";

    // Loop principale sui pixel di output
    for (int y_out = 0; y_out < new_height; ++y_out) {
        
        // Feedback visuale per immagini grandi
        if (y_out % 100 == 0) std::cout << "\rRiga: " << y_out << "/" << new_height << std::flush;

        for (int x_out = 0; x_out < new_width; ++x_out) {
            
            // Inverse Mapping: centro del pixel di output riportato nello spazio input
            float src_x = (x_out + 0.5f) / SCALE_FACTOR - 0.5f;
            float src_y = (y_out + 0.5f) / SCALE_FACTOR - 0.5f;

            // Pixel intero di riferimento (Nearest Neighbor)
            int center_x = std::round(src_x);
            int center_y = std::round(src_y);

            // Accumulatori float per ogni canale per evitare overflow/arrotondamenti prematuri
            std::vector<float> channel_sums(input.channels, 0.0f);
            float total_weight = 0.0f;

            // Convoluzione
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    
                    int neighbor_x = center_x + kx;
                    int neighbor_y = center_y + ky;
                    
                    float dx = src_x - neighbor_x;
                    float dy = src_y - neighbor_y;
                    
                    float weight = calculate_weight(dx, dy);

                    // Accumulo pesato per ogni canale (R, G, B, A) separatamente
                    for (int c = 0; c < input.channels; ++c) {
                        unsigned char val = input.get_pixel_safe(neighbor_x, neighbor_y, c);
                        channel_sums[c] += val * weight;
                    }
                    total_weight += weight;
                }
            }

            // Normalizzazione e scrittura
            if (total_weight > 0.0f) {
                for (int c = 0; c < input.channels; ++c) {
                    // Divisione per peso totale + cast sicuro a uint8
                    int final_val = (int)(channel_sums[c] / total_weight);
                    final_val = std::max(0, std::min(255, final_val)); // Clamp 0-255
                    
                    output.set_pixel(x_out, y_out, c, (unsigned char)final_val);
                }
            }
        }
    }
    std::cout << "\n>> Upscaling completato.\n";
    return output;
}

int main() {
    std::string input_path, output_name;

    std::cout << "=== CPU IMAGE UPSCALER ===\n";
    std::cout << "Inserisci immagine (es. input.jpg): ";
    std::cin >> input_path;

    // Caricamento dati raw tramite stb_image
    int w, h, ch;
    unsigned char* img_data = stbi_load(input_path.c_str(), &w, &h, &ch, 0);

    if (!img_data) {
        std::cerr << "ERRORE: Impossibile aprire " << input_path << "\n";
        return 1;
    }

    // Trasferimento dati in struttura C++ gestita
    Image input_img(w, h, ch);
    std::copy(img_data, img_data + (w * h * ch), input_img.data.begin());
    stbi_image_free(img_data); // Pulizia memoria C-style

    // Misurazione Tempo
    auto start = std::chrono::high_resolution_clock::now();
    
    Image output_img = upscale_image(input_img);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    
    std::cout << "Tempo CPU: " << duration.count() << " ms\n";

    // Salvataggio
    std::cout << "Nome output (no estensione): ";
    std::cin >> output_name;
    std::string output_filename = output_name + ".png";

    // Stride = larghezza in byte di una riga
    int stride = output_img.width * output_img.channels;
    stbi_write_png(output_filename.c_str(), output_img.width, output_img.height, output_img.channels, output_img.data.data(), stride);

    std::cout << "Salvato in: " << output_filename << "\n";

    return 0;
}