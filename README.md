# CUDA Gaussian Upscaling Optimization

Questo repository contiene il progetto finale sviluppato per l'esame di **Sistemi di Elaborazione Accelerata M** del corso di laurea magistrale in Ingegneria Informatica UNIBO.

Il progetto consiste nell'implementazione e nell'ottimizzazione incrementale di un algoritmo di **Gaussian Upscaling** (interpolazione di immagini) utilizzando **CUDA C++**. L'obiettivo principale è stato massimizzare il throughput e ridurre i tempi di esecuzione rispetto a una baseline sequenziale e a una prima implementazione GPU ingenua (Naïve).

## Obiettivi del Progetto
* Accelerare l'elaborazione di immagini a risoluzione variabile tramite parallelismo massivo su GPU.
* Analizzare e superare i colli di bottiglia architetturali (Compute Bound vs Memory Bound).
* Validare le prestazioni tramite profiling avanzato con **NVIDIA Nsight Compute**.

## Fasi di Ottimizzazione
Il codice è stato evoluto attraverso quattro versioni distinte, ognuna mirata a risolvere specifiche criticità emerse dall'analisi del profiler:

1.  **Baseline (Naïve):** Implementazione diretta dell'algoritmo. Limitata da pesanti calcoli ridondanti (`expf`) e accessi alla memoria non ottimizzati.
2.  **Constant Memory:** Ottimizzazione computazionale tramite pre-calcolo del kernel gaussiano e memorizzazione in *Constant Memory* per ridurre la latenza di accesso ai coefficienti.
3.  **Shared Memory (Tiling):** Implementazione del tiling per sfruttare la *Shared Memory*. Analisi critica del regresso prestazionale dovuto al disallineamento dei dati RGB (3 canali).
4.  **Vectorized Memory (Finale):** Adozione di tipi vettoriali nativi (`uchar4`) per garantire accessi alla memoria *coalesced* e allineati. Questa versione ha raggiunto la saturazione della banda passante (**Memory Bound**).

## Risultati Chiave
* **Hardware Target:** NVIDIA GeForce GTX 1660 Ti (Turing).
* **Speedup:** ~1.65x stabile rispetto alla versione CUDA Naïve su dataset grandi (FHD, 4K, 8K).
* **Efficienza:** L'analisi con il **Roofline Model** conferma che la versione finale tocca il limite fisico della banda di memoria (DRAM), massimizzando l'utilizzo delle risorse hardware disponibili.

---