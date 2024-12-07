# Menggunakan base image Hadoop
FROM ubuntu:20.04  

# Menambahkan informasi maintainer
LABEL maintainer="Your Name <azran18112020@gmail.com>"
LABEL version="1.0"
LABEL description="Docker container for big data processing"


# Set non-interactive untuk menghindari prompt saat instalasi
ENV DEBIAN_FRONTEND=noninteractive

# Install Python3 dan pip3
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Set up necessary directories inside the container
RUN mkdir -p /app/data/Crawling_Data

# Mount folder Crawling_Data sebagai volume
VOLUME /app/data/Crawling_Data

# Set working directory di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY ./requirements.txt /app/requirements.txt

# Menyalin folder Crawling_Data ke dalam container
COPY ./Crawling_Data /app/data/Crawling_Data

# Menyalin script utama (main.py) ke dalam container
COPY ./main.py /app/main.py

COPY ./proses_kata.py /app/proses_kata.py

COPY ./Kmeans.py /app/Kmeans.py

# Menyalin script run.sh ke dalam container
ADD run.sh /run.sh

# Memberikan izin eksekusi pada script run.sh
RUN chmod a+x /run.sh

# Install dependensi dari requirements.txt (jika ada)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Mengekspos port yang diperlukan (misalnya jika aplikasi menggunakan port tertentu)
HEALTHCHECK CMD curl -f http://localhost:9090/ || exit 1

EXPOSE 9090

# Perintah yang dijalankan saat container aktif
CMD ["/run.sh"]
