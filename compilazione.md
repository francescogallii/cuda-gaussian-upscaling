# Istruzioni di Compilazione ed Esecuzione

Questo file contiene tutti i comandi necessari per scaricare le dipendenze, compilare il codice CPU e GPU, eseguire i programmi e generare i report di profiling.

## 1. Download delle Librerie (Prerequisiti)

Il progetto utilizza le librerie *stb* per la gestione delle immagini (lettura/scrittura). Esegui questi comandi nella cartella del progetto per scaricare gli header necessari.

```bash
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

```

---

## 2. Compilazione ed Esecuzione Codice CPU (Sequenziale)

Utilizziamo `g++` per compilare la versione di riferimento che gira su CPU.

### Compilazione

```bash
g++ -O3 -Wall -Wextra upscaling_sequenziale.cpp -o upscaling_sequenziale -lm

```
Per non mostrare gli Warning di compilazione dovuti alle librerie che usano una sintassi di inizializzazione vecchia


**Spiegazione Flag:**

* `g++`: Compilatore GNU per C++.
* `-O3`: Attiva il livello massimo di ottimizzazione del compilatore (fondamentale per confrontare le performance).
* `-Wall -Wextra`: Attiva tutti i warning e messaggi di avviso extra (utile per intercettare errori e codice non pulito).
* `-o upscaling_sequenziale`: Specifica il nome del file eseguibile in output.
* `-lm`: Linka la libreria matematica standard (`libm`), necessaria per funzioni come `exp`, `pow`, etc.

```bash
g++ -O3 -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter upscaling_sequenziale.cpp -o upscaling_sequenziale -lm

```

### Esecuzione

```bash
./upscaling_sequenziale

```

---

## 3. Compilazione Codice CUDA (GPU)

Utilizziamo `nvcc` (NVIDIA CUDA Compiler). Scegli il comando in base alla versione del kernel che vuoi testare.

**Nota sull'architettura:** Il flag `-arch=sm_75` è specifico per l'architettura **Turing** (NVIDIA GTX 1660 Ti). Se cambiassi scheda, dovresti aggiornare questo numero.

### A. Versione Baseline (Naïve)

```bash
nvcc -arch=sm_75 -lineinfo -o upscale_naive upscaling_Parallelo_Naive.cu

```

### B. Versione Constant Memory

```bash
nvcc -arch=sm_75 -lineinfo -o upscale_constant 5_1_constant_memory.cu

```

### C. Versione Shared Memory

```bash
nvcc -arch=sm_75 -lineinfo -o upscale_shared 5_2_shared_memory.cu

```

### D. Versione Finale (Vectorized - Best Performance)

```bash
nvcc -arch=sm_75 -lineinfo -o upscale_vectorized 5_3_vectorized.cu

```

**Spiegazione Flag:**

* `nvcc`: Il compilatore CUDA.
* `-arch=sm_75`: Compila il codice specificamente per la Compute Capability 7.5 (GTX 1660 Ti), abilitando le istruzioni specifiche di quell'hardware.
* `-lineinfo`: Genera informazioni sulle righe di codice sorgente. **Fondamentale** per Nsight Compute, permette di vedere a quale riga di C++ corrisponde un collo di bottiglia nel profiler.
* `-o nome_file`: Specifica il nome dell'eseguibile.

---

## 4. Esecuzione Codice GPU

Una volta compilati, puoi eseguire i programmi lanciandoli direttamente. Assicurati di avere le immagini di input nella stessa cartella (es. `paesaggio_fhd.jpg`).

```bash
# Esecuzione standard (stampa il tempo a video)
./upscale_vectorized

```

---

## 5. Profiling con Nsight Compute (ncu)

Questi comandi servono per analizzare le prestazioni nel dettaglio. È richiesto `sudo` per accedere ai contatori hardware della GPU.

### A. Report Completo (Full)

Da usare per l'analisi approfondita (Roofline Model, Memory Analysis).

```bash
sudo ncu --set full -o report_full_vectorized -f ./upscale_vectorized

```

**Spiegazione Flag:**

* `sudo`: Necessario per i permessi di profiling hardware.
* `ncu`: Eseguibile di Nsight Compute CLI.
* `--set full`: Raccoglie **tutte** le metriche disponibili. Rende l'esecuzione molto lenta ma fornisce il report completo.
* `-o report_name`: Nome del file di output (verrà salvato come `.ncu-rep`).
* `-f`: (Force) Sovrascrive il file di report se esiste già.

### B. Report Leggero (Light)

Da usare per lo **Studio di Scalabilità** (Punto 7), quando serve misurare velocemente Grid Size, Throughput e Occupancy su molte immagini diverse.

```bash
sudo ncu -o report_scaling_4k -f ./upscale_vectorized

```

*(Senza `--set full`, il profiler è molto più veloce e raccoglie solo i dati essenziali).*