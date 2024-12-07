import os
from bs4 import BeautifulSoup
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download('popular')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Fungsi untuk mengumpulkan stopwords dari file yang ditentukan
def gather_stopwords(file_path):
    """Gather stopwords from a specified file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

# Fungsi untuk preprocessing teks
def preprocess_text(text, stopwords):
    """Apply stopword removal and stemming to the text."""
    # Ubah menjadi huruf kecil dan hapus karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()  # Pastikan text diubah menjadi huruf kecil
    text = re.sub(r'\W+|\d+', ' ', text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Inisialisasi stemmer Sastrawi
    stemmer = StemmerFactory().create_stemmer()

    # Hapus stopwords dan terapkan stemming
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
    return ' '.join(filtered_tokens)

# Fungsi untuk preprocessing data dari file .txt
def preprocess_text_files(input_text_dir, stopwords):
    """Preprocess all .txt files in a directory."""
    preprocessed_data = []

    # Membaca semua file .txt dalam direktori
    for file_name in os.listdir(input_text_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_text_dir, file_name)

            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                title = ""
                content = ""

                # Mengolah konten setiap file
                for line in lines:
                    if line.startswith("Title: "):
                        title = line.replace("Title: ", "").strip().lower()  # Simpan judul dalam huruf kecil
                    elif line.startswith("News Text:"):
                        content = line.replace("News Text:", "").strip()

                # Preprocessing konten
                preprocessed_content = preprocess_text(content, stopwords)

                # Menyimpan data yang diproses dalam format dictionary
                if title and preprocessed_content:
                    preprocessed_data.append({
                        'title': title,
                        'content': preprocessed_content,  # Simpan konten yang telah diproses
                        'file_name': file_name
                    })

    return preprocessed_data

# Path ke file stopwords di dalam folder Crawling_Data
crawling_data_folder = "Crawling_Data"
pemilu_2024_folder = os.path.join(crawling_data_folder, "pemilu_2024")  # Folder pemilu_2024 di dalam crawling_data
STOPWORD_FILE = os.path.join(crawling_data_folder, 'stopwords.txt')  # File stopwords.txt berada di dalam Crawling_Data
stopwords = gather_stopwords(STOPWORD_FILE)

# Proses preprocessing data dari file .txt yang ada di folder pemilu_2024
preprocessed_data = preprocess_text_files(pemilu_2024_folder, stopwords)

# Menyimpan hasil preprocessing ke file output
output_file = os.path.join(crawling_data_folder, "preprocessed_data.txt")  # Menyimpan output di folder Crawling_Data
with open(output_file, 'w', encoding='utf-8') as file:
    for data in preprocessed_data:
        file.write(f"Title: {data['title']}\n")  # Perbaiki penulisan 'title' dan bukan 'Title'
        file.write(f"Content:\n{data['content']}\n\n")  # Tambahkan konten yang diproses

print(f"Preprocessing completed. Preprocessed data saved to {output_file}.")
