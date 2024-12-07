#!/bin/bash

# Menjalankan aplikasi atau proses, misalnya menjalankan main.py
echo "Starting the application..."
python3 /app/big-data.py

# Menjaga container tetap berjalan
tail -f /dev/null
