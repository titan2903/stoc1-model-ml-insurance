# Machine Learning - Prediksi Kategori Biaya Asuransi Kesehatan

Tugas ini menggunakan **Decision Tree Classifier** untuk memprediksi kategori biaya asuransi kesehatan berdasarkan data demografis dan gaya hidup pasien.

---

## Daftar Isi

- [Deskripsi Tugas](#deskripsi-tugas)
- [Struktur Dataset](#struktur-dataset)
- [Requirements](#requirements)
- [Workflow Machine Learning](#workflow-machine-learning)
- [Penggunaan](#penggunaan)
- [File Output](#file-output)
- [Hasil Model](#hasil-model)
- [Interpretasi Kategori](#interpretasi-kategori)

---

## Deskripsi Tugas

Tugas ini menerapkan workflow machine learning untuk:
1. **Memproses data numerik** dari dataset asuransi kesehatan
2. **Mengidentifikasi features dan label** yang relevan untuk supervised learning
3. **Membangun model** Decision Tree Classifier untuk klasifikasi kategori biaya
4. **Menguji dan mengevaluasi** akurasi model
5. **Memprediksi kategori biaya** untuk data baru

**Tujuan Akhir:** Mengklasifikasikan biaya asuransi menjadi 4 kategori: Low, Medium, High, dan Very High.

---

## Struktur Dataset

### File Input
- **`insurance.csv`** - Dataset asuransi kesehatan dengan 1338 records

### Kolom Dataset
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `age` | integer | Usia pasien (18-64 tahun) |
| `sex` | string | Jenis kelamin (male/female) |
| `bmi` | float | Indeks Massa Tubuh |
| `children` | integer | Jumlah anak/tanggungan |
| `smoker` | string | Status perokok (yes/no) |
| `region` | string | Region geografis (northeast, northwest, southeast, southwest) |
| `charges` | float | Biaya asuransi tahunan (USD) |

### Preprocessing
- **Encoding kategorik**: Kolom `sex`, `smoker`, dan `region` dikonversi ke format numerik
- **Target variable**: Kolom `charges` dikategorisasi menjadi 4 kategori berdasarkan quartile:
  - **Low**: Quartile 1 (terendah)
  - **Medium**: Quartile 2
  - **High**: Quartile 3
  - **Very High**: Quartile 4 (tertinggi)

---

## Requirements

### Library yang Diperlukan
```bash
pip install pandas numpy scikit-learn
```

### Versi Minimum
- Python 3.7+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- numpy >= 1.21.0

---

## Workflow Machine Learning

### Tahap 1: Import Library & Baca Data
```python
# Membaca dataset insurance.csv
insurance = pd.read_csv('insurance.csv')
```
**Output:** Dataset dengan 1338 baris dan 7 kolom

---

### Tahap 2: Eksplorasi & Preprocessing Data
```python
# Membuat kategori biaya berdasarkan quartile
insurance['charges_category'] = pd.qcut(insurance['charges'], 
                                         q=4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
```

---

### Tahap 3: Persiapan Features & Label
```python
# Memisahkan features (X) dan label (y)
X = insurance.drop(['charges', 'charges_category'], axis=1)
y = insurance['charges_category']

# Encoding data kategorik
X['sex'] = LabelEncoder().fit_transform(X['sex'])
X['smoker'] = LabelEncoder().fit_transform(X['smoker'])
X['region'] = LabelEncoder().fit_transform(X['region'])
```

---

### Tahap 4: Split Data Training & Testing
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
```
**Split Ratio:**
- Training: 80% (≈ 1070 samples)
- Testing: 20% (≈ 268 samples)

---

### Tahap 5: Training Model
```python
classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
classifier.fit(X_train, y_train)
```

---

### Tahap 6: Evaluasi Model
```python
# Prediksi pada data testing
y_pred = classifier.predict(X_test)

# Laporan evaluasi
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

### Tahap 7: Prediksi pada Data Baru
```python
# Data baru untuk diprediksi
new_data = pd.DataFrame({
    'age': [25, 45, 55, 30, 65],
    'sex': ['male', 'female', 'male', 'female', 'male'],
    'bmi': [24.5, 28.3, 32.1, 26.8, 35.2],
    # ... kolom lainnya
})

# Prediksi kategori biaya
predictions = classifier.predict(new_data_encoded)
```

---

## Penggunaan

### Menjalankan Notebook

1. **Buka Jupyter Notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Jalankan semua sel (Cells)**
   - Tekan `Ctrl + A` untuk memilih semua sel
   - Tekan `Ctrl + Enter` untuk menjalankan semua sel

3. **Atau jalankan per sel:**
   - Pilih sel yang diinginkan
   - Tekan `Shift + Enter` untuk menjalankan sel saat ini

### Struktur Notebook (`main.ipynb`)

| Sel | Deskripsi |
|-----|-----------|
| 1 | Pengenalan dan penjelasan tugas |
| 2 | Import library dan baca data |
| 3 | Eksplorasi dan kategorisasi data |
| 4 | Encoding data kategorik |
| 5 | Split data training/testing |
| 6 | Training dan evaluasi model |
| 7 | Perbandingan hasil prediksi |
| 8 | Prediksi untuk seluruh dataset |
| 9 | Testing dengan data baru |

---

## File Output

### File yang Dihasilkan

#### 1. **`hasil_prediksi_insurance.csv`**
Dataset lengkap dengan kolom tambahan:
- `charges_category`: Kategori biaya aktual
- `predicted_category`: Prediksi kategori dari model

**Contoh:**
```csv
age,sex,bmi,children,smoker,region,charges,charges_category,predicted_category
25,male,24.5,0,no,northeast,2500,Low,Low
45,female,28.3,2,no,southwest,8000,Medium,Medium
65,male,35.2,0,yes,northeast,15000,Very High,Very High
```

---

## Hasil Model

### Model Performance
- **Type**: Decision Tree Classifier
- **Max Depth**: 5 (untuk mencegah overfitting)
- **Test Size**: 20%
- **Random State**: 42 (untuk reprodusibilitas)

### Metrics Evaluasi
Model dievaluasi menggunakan:
- **Precision**: Keakuratan prediksi positif
- **Recall**: Sensitivitas model
- **F1-Score**: Harmonic mean dari precision dan recall
- **Support**: Jumlah samples per kategori
- **Confusion Matrix**: Untuk menganalisis error tipe I dan II

### Output Evaluasi
```
              precision    recall  f1-score   support

         Low       0.88      0.90      0.89        68
      Medium       0.80      0.75      0.77        60
        High       0.85      0.82      0.83        67
   Very High       0.87      0.91      0.89        73

   accuracy                           0.85       268
  macro avg       0.85      0.85      0.85       268
weighted avg       0.85      0.85      0.85       268
```

---

## Interpretasi Kategori

### Kategori Biaya Asuransi

| Kategori | Range (USD) | Karakteristik Tipik |
|----------|------------|-------------------|
| **Low** | $1,121 - $9,722 | Usia muda, BMI normal, non-smoker |
| **Medium** | $9,723 - $16,639 | Usia menengah, BMI sedang, non-smoker |
| **High** | $16,640 - $27,321 | Usia lebih tua atau perokok, BMI tinggi |
| **Very High** | $27,322 - $63,770 | Usia tua, perokok, BMI tinggi |

---

## Contoh Prediksi Data Baru

### Input Data
| Age | Sex | BMI | Smoker | Region |
|-----|-----|-----|--------|--------|
| 25 | Male | 24.5 | No | Northeast |
| 45 | Female | 28.3 | No | Southwest |
| 55 | Male | 32.1 | Yes | Northwest |
| 30 | Female | 26.8 | No | Southeast |
| 65 | Male | 35.2 | Yes | Northeast |

### Output Prediksi
| Age | Sex | BMI | Smoker | Region | Predicted Category |
|-----|-----|-----|--------|--------|-------------------|
| 25 | Male | 24.5 | No | Northeast | **Low** |
| 45 | Female | 28.3 | No | Southwest | **Medium** |
| 55 | Male | 32.1 | Yes | Northwest | **High** |
| 30 | Female | 26.8 | No | Southeast | **Low** |
| 65 | Male | 35.2 | Yes | Northeast | **Very High** |

---

## Insights & Analisis

### Faktor Penting yang Mempengaruhi Biaya Asuransi

1. **Usia (Age)**: Semakin tua, biaya semakin tinggi
2. **Status Perokok (Smoker)**: Perokok memiliki biaya signifikan lebih tinggi
3. **BMI**: BMI lebih tinggi berkaitan dengan biaya lebih tinggi
4. **Region**: Lokasi geografis berpengaruh pada biaya

### Rekomendasi

- **Untuk Biaya Rendah**: Pertahankan usia muda, BMI sehat, dan tidak merokok
- **Untuk Manajemen Risiko**: Pantau peningkatan BMI dan status merokok
- **Untuk Perencanaan Finansial**: Antisipasi peningkatan biaya seiring bertambahnya usia

---

## Lisensi & Attribution

Data source: Insurance Health Cost Dataset
Model: Decision Tree Classifier (scikit-learn)
Python Libraries: pandas, numpy, scikit-learn