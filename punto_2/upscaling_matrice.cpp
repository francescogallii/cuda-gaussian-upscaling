#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// ==========================================
// SEZIONE CONFIGURAZIONE (MODULARE)
// ==========================================

// Fattore di scala (2 = raddoppia larghezza e altezza -> quadruplica i pixel totali)
const int SCALE_FACTOR = 2;

// Sigma per la distribuzione Gaussiana (controlla quanto velocemente scende il peso)
// Un valore più basso da molto peso al centro, un valore più alto ammorbidisce l'immagine.
const float SIGMA = 0.75f; 

// ==========================================
// STRUTTURE DATI
// ==========================================

struct Image {
    int width;
    int height;
    std::vector<int> data; // Vettore linearizzato

    // Costruttore per inizializzare un'immagine vuota
    Image(int w, int h) : width(w), height(h), data(w * h, 0) {}

    // Metodo helper per accedere ai pixel (Gestione bordi / Clamping)
    // Se chiediamo un pixel fuori dai bordi, restituisce il pixel del bordo più vicino.
    int get_pixel_safe(int x, int y) const {
        int clamped_x = std::max(0, std::min(x, width - 1));
        int clamped_y = std::max(0, std::min(y, height - 1));
        return data[clamped_y * width + clamped_x];
    }

    // Metodo per settare un pixel
    void set_pixel(int x, int y, int value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = value;
        }
    }
};

// ==========================================
// LOGICA DI CALCOLO
// ==========================================

// 1. Funzione per decidere la dimensione della sottomatrice (Raggio del kernel)
// Se l'immagine è molto piccola, usiamo un raggio piccolo. Se è grande, possiamo permetterci più precisione.
int get_dynamic_radius(int width, int height) {
    int min_dim = std::min(width, height);
    if (min_dim < 10) return 1;      // Sottomatrice 3x3 (Raggio 1)
    if (min_dim < 100) return 2;     // Sottomatrice 5x5 (Raggio 2)
    return 3;                        // Sottomatrice 7x7 (Raggio 3) per immagini grandi
}

// 2. Calcolo del Peso (Funzione Gaussiana)
// Restituisce un peso alto se la distanza è 0, basso se la distanza è alta.
float calculate_weight(float dx, float dy) {
    float distance_sq = dx * dx + dy * dy;
    // Formula Gaussiana: e^(-(x^2 + y^2) / (2 * sigma^2))
    return std::exp(-distance_sq / (2 * SIGMA * SIGMA));
}

// 3. Funzione di Upscaling Principale
Image upscale_image(const Image& input) {
    int new_width = input.width * SCALE_FACTOR;
    int new_height = input.height * SCALE_FACTOR;
    
    Image output(new_width, new_height);

    // Calcolo dinamico del raggio della sottomatrice
    int radius = get_dynamic_radius(input.width, input.height);
    std::cout << "DEBUG: Raggio Kernel calcolato: " << radius 
              << " (Sottomatrice " << (radius*2+1) << "x" << (radius*2+1) << ")\n";

    // Ciclo su ogni pixel dell'immagine di OUTPUT
    for (int y_out = 0; y_out < new_height; ++y_out) {
        for (int x_out = 0; x_out < new_width; ++x_out) {
            
            // Mappatura inversa: Dove cade questo pixel nell'immagine originale?
            // Aggiungiamo 0.5 per centrare il campionamento (convenzione grafica standard)
            float src_x = (x_out + 0.5f) / SCALE_FACTOR - 0.5f;
            float src_y = (y_out + 0.5f) / SCALE_FACTOR - 0.5f;

            // Coordinate intere del pixel centrale nell'input
            int center_x = std::round(src_x);
            int center_y = std::round(src_y);

            float total_weight = 0.0f;
            float weighted_sum = 0.0f;

            // Analisi della Sottomatrice (vicini attorno a center_x, center_y)
            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {
                    
                    // Coordinate del vicino che stiamo analizzando
                    int neighbor_x = center_x + kx;
                    int neighbor_y = center_y + ky;

                    // Recuperiamo il valore (safe handles out of bounds)
                    int pixel_val = input.get_pixel_safe(neighbor_x, neighbor_y);

                    // Calcoliamo la distanza reale tra il punto mappato (float) e il vicino (int)
                    float dx = src_x - neighbor_x;
                    float dy = src_y - neighbor_y;

                    // Calcolo peso
                    float weight = calculate_weight(dx, dy);

                    // Accumulo
                    weighted_sum += pixel_val * weight;
                    total_weight += weight;
                }
            }

            // Normalizzazione e assegnazione
            if (total_weight > 0.0f) {
                output.set_pixel(x_out, y_out, (int)(weighted_sum / total_weight));
            }
        }
    }

    return output;
}

// ==========================================
// MAIN E UTILITY DI STAMPA
// ==========================================

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
    // Esempio: Creiamo una piccola matrice 4x4
    int w = 4, h = 4;
    Image input_img(w, h);

    // Riempiamo con valori di test (es. gradiente o scacchiera)
    // 10  20  10  20
    // 20  80  80  20
    // 20  80  80  20
    // 10  20  10  20
    int values[] = {
        10, 20, 10, 20,
        20, 80, 80, 20,
        20, 80, 80, 20,
        10, 20, 10, 20
    };

    // Copiamo i dati nel vettore
    for(int i=0; i < w*h; i++) input_img.data[i] = values[i];

    // Mostra input
    print_matrix(input_img, "Input Matrix");

    // Elaborazione
    Image output_img = upscale_image(input_img);

    // Mostra output
    print_matrix(output_img, "Upscaled Matrix");

    return 0;
}