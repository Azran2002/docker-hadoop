import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# Fungsi membaca stopwords dari file
def load_stopwords(folder_path):
    stopwords_file_path = os.path.join(folder_path, "stopwords.txt")
    if not os.path.exists(stopwords_file_path):
        raise FileNotFoundError(f"File stopwords '{stopwords_file_path}' tidak ditemukan!")
    with open(stopwords_file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# Lokasi folder crawling data dan subfolder pemilu_2024
folder_path = "Crawling_Data/pemilu_2024"  # Ganti dengan lokasi folder subfolder pemilu_2024

# Load stopwords dari file
custom_stopwords = load_stopwords("Crawling_Data")  # Stopwords masih bisa diambil dari folder utama

# Fungsi menghitung jarak Euclidean
def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

# Fungsi K-Means manual
def k_means(data, k, max_iters=100):
    # Pastikan jumlah centroid tidak lebih besar dari jumlah data
    if k > len(data):
        print(f"Peringatan: Jumlah data ({len(data)}) lebih sedikit daripada jumlah cluster ({k}). Menyesuaikan jumlah cluster ke {len(data)}.")
        k = len(data)  # Sesuaikan k agar tidak lebih besar dari jumlah data

    # Langkah 1: Inisialisasi centroid awal secara random
    np.random.seed(42)
    centroids = data[np.random.choice(len(data), k, replace=False)]

    for iteration in range(max_iters):
        # Langkah 2: Hitung jarak data ke setiap centroid
        distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data])

        # Langkah 3: Assign data ke cluster terdekat
        clusters = np.argmin(distances, axis=1)

        # Langkah 4: Hitung ulang centroid berdasarkan rata-rata
        new_centroids = np.array([data[clusters == cluster].mean(axis=0) for cluster in range(k)])

        # Cek konvergensi
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return clusters, centroids

# Membaca data teks dari folder pemilu_2024
data = []

if os.path.exists(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt') and file_name != "stopwords.txt":  # Abaikan file stopwords
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                data.append(file.read())
else:
    raise FileNotFoundError(f"Subfolder '{folder_path}' tidak ditemukan!")

if not data:
    raise ValueError("Folder tidak berisi file teks!")

# Vektorisasi teks menggunakan CountVectorizer dengan stopwords dari file
vectorizer = CountVectorizer(stop_words=custom_stopwords)
X = vectorizer.fit_transform(data)

# Ekstraksi topik menggunakan LDA
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda_features = lda.fit_transform(X)

# Menampilkan topik
print("Topik yang dihasilkan LDA:")
terms = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topik {idx + 1}: {', '.join([terms[i] for i in topic.argsort()[-5:]])}")

# Menampilkan distribusi topik dalam dokumen
print("\nDistribusi Topik dalam Setiap Dokumen:")
for idx, doc in enumerate(lda_features):
    print(f"Dokumen {idx + 1}: {', '.join([f'Topik {i+1}: {doc[i]:.2f}' for i in range(len(doc))])}")

# Clustering menggunakan K-Means manual
k = 3  # Jumlah cluster
clusters, centroids = k_means(lda_features, k)

# Menampilkan hasil clustering
result = pd.DataFrame({'Teks': data, 'Cluster': clusters})
print("\nHasil Clustering:")
print(result)

# Visualisasi clustering
plt.figure(figsize=(8, 6))
plt.scatter(lda_features[:, 0], lda_features[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroid')
plt.title("K-Means Clustering Berdasarkan Distribusi Topik LDA")
plt.xlabel("Komponen 1")
plt.ylabel("Komponen 2")
plt.legend()
plt.show()
plt.savefig('/workspaces/docker-hadoop/Big_Data/save/visualisai.png')