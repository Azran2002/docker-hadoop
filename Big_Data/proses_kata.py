import os
from collections import Counter
import string
from nltk.tokenize import word_tokenize  # Pastikan nltk sudah diinstal dan diunduh tokenizersnya

kategori_liputan6 = ["Pemilu 2024"]

# Fungsi untuk menghitung frekuensi kata
def calculate_word_frequencies(text_dir, words_to_remove):
    all_texts = []
    for file in os.listdir(text_dir):
        if file.endswith('.txt'):
            with open(os.path.join(text_dir, file), 'r', encoding='utf-8') as f:
                all_texts.append(f.read())

    combined_text = ' '.join(all_texts)

    # Tokenisasi dan menghapus tanda baca
    words = word_tokenize(combined_text)
    words = [word.lower() for word in words if word not in string.punctuation]  # Menghapus tanda baca dan mengubah menjadi huruf kecil

    # Menghilangkan kata-kata yang tidak diinginkan, kata kosong, atau hanya berisi spasi
    filtered_words = [word for word in words if word not in words_to_remove and word.strip() != '' and word.isalpha()]

    return Counter(filtered_words)

# Daftar kata-kata yang ingin dihilangkan
words_to_remove = {'2024', 'advertisement', 'pemilu_2024', 'untuk', ':', ';', 'diri', 'data', 'com', 'liputan6','di', 'ini', 'dan',
                   'jadi', 'dapat', 'lengkap', 'baca', 'guna', 'pilih', 'ikut', 'kembali', 'dua', 'akan', 'temu', 'yang', 'anda', 'satu'}

# Fungsi untuk mencetak 20 kata teratas
def print_top_words(counter, top_n=20):
    for word, freq in counter.most_common(top_n):
        print(f"{word}: {freq}")

# Proses setiap kategori dan hitung frekuensi kata
for kategori in kategori_liputan6:
    txt_dir = "Crawling_Data/pemilu_2024"  # Mengubah direktori untuk file teks
    word_freqs = calculate_word_frequencies(txt_dir, words_to_remove)
    print(f"\n20 Kata dengan Frekuensi Terbanyak untuk Kategori '{kategori}':")
    print_top_words(word_freqs)
