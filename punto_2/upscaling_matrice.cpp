#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// --- CONFIGURAZIONE ---

const int SCALE_FACTOR = 2; // Fattore di upscaling (2x = 4 volte i pixel totali)

// Sigma controlla lo spread della distribuzione. 
// 0.75f bilancia nitidezza e smoothing per evitare aliasing.
const float SIGMA = 0.75f; 

// STRUTTURE DATI

struct Image {
    int width;
    int height;
    std::vector<int> data; // Vettore linearizzato per massimizzare la cache locality

    Image(int w, int h) : width(w), height(h), data(w * h, 0) {}

    // Accesso sicuro ai pixel: gestisce i bordi tramite clamping (ripetizione dell'ultimo pixel valido)
    int get_pixel_safe(int x, int y) const {
        int clamped_x = std::max(0, std::min(x, width - 1));
        int clamped_y = std::max(0, std::min(y, height - 1));
        return data[clamped_y * width + clamped_x];
    }

    void set_pixel(int x, int y, int value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = value;
        }
    }
};

// LOGICA DI CALCOLO

// Adatta la dimensione del kernel in base alla risoluzione input.
// Evita overhead inutile su immagini molto piccole.
int get_dynamic_radius(int width, int height) {
    int min_dim = std::min(width, height);
    if (min_dim < 10) return 1;      // Kernel 3x3
    if (min_dim < 100) return 2;     // Kernel 5x5
    return 3;                        // Kernel 7x7 (Standard per Hi-Res)
}

// Calcola il peso gaussiano basato sulla distanza euclidea al quadrato
float calculate_weight(float dx, float dy) {
    float distance_sq = dx * dx + dy * dy;
    return std::exp(-distance_sq / (2 * SIGMA * SIGMA));
}

Image upscale_image(const Image& input) {
    int new_width = input.width * SCALE_FACTOR;
    int new_height = input.height * SCALE_FACTOR;
    
    Image output(new_width, new_height);

    int radius = get_dynamic_radius(input.width, input.height);
    std::cout << "INFO: Kernel size: " << (radius*2+1) << "x" << (radius*2+1) << "\n";

    for (int y_out = 0; y_out < new_height; ++y_out) {
        for (int x_out = 0; x_out < new_width; ++x_out) {
            
            // Mapping inverso Output -> Input.
            // L'offset 0.5 garantisce l'allineamento corretto tra i centri dei pixel delle due griglie.
            float src_x = (x_out + 0.5f) / SCALE_FACTOR - 0.5f;
            float src_y = (y_out + 0.5f) / SCALE_FACTOR - 0.5f;

            // Nearest integer pixel (centro del kernel)
            int center_x = std::round(src_x);
            int center_y = std::round(src_y);

            float total_weight = 0.0f;
            float weighted_sum = 0.0f;

            // Convoluzione sul vicinato
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    
                    int neighbor_x = center_x + kx;
                    int neighbor_y = center_y + ky;

                    int pixel_val = input.get_pixel_safe(neighbor_x, neighbor_y);

                    // Distanza sub-pixel per il calcolo del peso
                    float dx = src_x - neighbor_x;
                    float dy = src_y - neighbor_y;

                    float weight = calculate_weight(dx, dy);

                    weighted_sum += pixel_val * weight;
                    total_weight += weight;
                }
            }

            // Normalizzazione (Media Ponderata)
            if (total_weight > 0.0f) {
                output.set_pixel(x_out, y_out, (int)(weighted_sum / total_weight));
            }
        }
    }

    return output;
}

void print_matrix(const Image& img, const std::string& title) {
    std::cout << "\n--- " << title << " (" << img.width << "x" << img.height << ") ---\n";
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            std::cout << std::setw(4) << img.data[y * img.width + x] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Test case 4x4
    int w = 4, h = 4;
    Image input_img(w, h);

    // Pattern di test
    int values[] = {
        10, 20, 10, 20,
        20, 80, 80, 20,
        20, 80, 80, 20,
        10, 20, 10, 20
    };

    for(int i=0; i < w*h; i++) input_img.data[i] = values[i];

    print_matrix(input_img, "Input Matrix");

    // Esecuzione algoritmo (Sequenziale CPU)
    Image output_img = upscale_image(input_img);

    print_matrix(output_img, "Upscaled Matrix");

    return 0;
}