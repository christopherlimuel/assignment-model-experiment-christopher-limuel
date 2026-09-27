# Sentiment Analysis Model Experiment: Classic Machine Learning (TF-IDF + Logistic Regression) vs. LLM API (Gemini Flash-Lite)

Project ini bertujuan untuk menguji dan membandingkan performa, latensi, serta efisiensi biaya antara pendekatan **Machine Learning Klasik** dan **Large Language Model (LLM) via API** dalam tugas analisis sentimen ulasan pelanggan.

---

## 1. Problem Statement & Dataset

#### Problem Statement
Mengklasifikasikan ulasan pelanggan ke dalam dua kategori sentimen (**Positif** vs **Negatif**) secara otomatis. Sekaligus mengevaluasi pendekatan Classical NLP dan LLM dalam sentiment analysis menggunakan metrik performa klasifikasi, kecepatan implementasi, dan biaya penggunaan.

#### Dataset Overview
* **Total Ukuran Data:** 200 data ulasan pelanggan.
* **Pembagian Data (Data Split):**
  * **Training Set:** 160 sampel (80%)
  * **Test Set:** 40 sampel (20%)
* **Distribusi Test Set:** 18 Negatif, 22 Positif.

---

## 2. Ringkasan Eksperimen

Eksperimen ini membandingkan dua pendekatan dalam melakukan sentiment analysis:

1. **Pendekatan Klasik (TF-IDF + Logistic Regression):**
   * **Feature Extraction:** Ekstraksi fitur statistik menggunakan *Term Frequency-Inverse Document Frequency* (TF-IDF).
   * **Classifier Model:** Logistic Regression dilatih secara lokal pada 160 data training.

2. **Pendekatan LLM (Google Gemini 3.1 Flash-Lite via API):**
   * **Metode:** *Zero-shot Prompting* menggunakan API Request.
   * **Prompt Strategy:** Menginstruksikan model untuk mengembalikan klasifikasi biner (`Positif` / `Negatif`) berdasarkan konteks dominan ulasan.

---

## 3. Hasil Evaluasi & Perbandingan

#### Perbandingan Metrik Kuantitatif (pada 40 Test Data)

| Metrik / Parameter | Pendekatan Klasik (TF-IDF + LogReg) | Pendekatan LLM (Gemini 3.1 Flash-Lite) |
| :--- | :---: | :---: |
| **Accuracy** | **1.00 (100%)** | **1.00 (100%)** | 
| **Precision (Macro Avg)** | **1.00** | **1.00** | 
| **Recall (Macro Avg)** | **1.00** | **1.00** | 
| **F1-Score (Macro Avg)** | **1.00** | **1.00** | 
| **Inference Time (Latensi)** | **~0.0015 detik** | **~1.22 detik** | 
| **Biaya Komputasi (Cost)** | Gratis (CPU Lokal) | Berbayar / Kuota API | 
| **Dependensi Infrastruktur** | Offline / Local Memory | API Request via Internet connection

---

## 4. Analisis Trade-off dan Limitation

#### **Analisis Performa**

Dari test dataset berukuran 40 data, baik pendekatan Klasik (TF-IDF + Logistic Regression) maupun LLM (Gemini 3.1 Flash-Lite) menunjukkan performa kuantitatif yang identik, yaitu mencapai Accuracy dan F1-Score sebesar 1.00 (100%).
Namun hasil ini juga dapat mengindikasikan test dataset saat ini kurang kompleks sehingga belum mampu menunjukkan limitasi dari kedua pendekatan.

#### **Analisis Kecepatan dan Biaya**

Pendekatan klasik unggul secara signifikan dari segi kecepatan dan biaya penggunaan. Metode klasik dapat memproses input dengan jauh lebih cepat (0.0015s), kebutuhan resource yang ringan, dan dapat dijalankan secara offline. Sedangkan metode LLM butuh waktu lebih lama untuk memproses input (1.22s), mengonsumsi token, dan melibatkan pengiriman data ke platform ketiga.

#### **Catatan**

Pada inference test, digunakan sampel kalimat yang mengandung mixed sentiment dan kata hubung pertentangan:
> _"Barangnya oke, tapi pengirimannya sangat lama"_
> Ground Truth: **negatif** (ada penekanan kekecewaan di klausa kedua)

Model LLM mengkategorikan ulasan contoh tersebut sebagai ulasan negatif, sedangkan model klasik melihatnya sebagai ulasan positif. 

Temuan kualitatif ini mengindikasikan bahwa pendekatan klasik belum mampu menangani kalimat ambigu/kompleks yang lazim ditemui pada kondisi real-world.

---

## 5. Rekomendasi Technical Approach

Berdasarkan performa yang terukur dari test dataset yang ada saat ini, **pendekatan klasik (TF-IDF + Logistic Regression)** lebih direkomendasikan dibandingkan pendekatan LLM. 

Alasan:
- Menghasilkan metrik performa yang tinggi dan sepadan dengan metode LLM, pada test dataset
- Memiliki latensi yang jauh lebih rendah
- Menghemat cost (bebas biaya token)

Namun disarankan untuk meningkatkan kualitas dataset agar lebih besar dan variatif, memuat kalimat ambigu, sarkasme, negasi ganda, dll. Sehingga dataset lebih representatif dan model lebih robust untuk digunakan di real-world cases. 

Setelah itu dapat dilakukan pengujian kembali (re-evaluasi) apakah pendekatan klasik masih memiliki performa akurasi yang mumpuni dengan dataset yang baru.

---

## 6. Cara Menjalankan Project

#### Prasyarat System
* Python 3.9+
* Google Gemini API Key (diakses dari system variable)

#### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/christopherlimuel/assignment-model-experiment-christopher-limuel

pip install -r requirements.txt
```


#### 2. Menjalankan Notebook / Script Evaluasi

```bash
jupyter notebook notebook/experiment_notebook.ipynb
```
