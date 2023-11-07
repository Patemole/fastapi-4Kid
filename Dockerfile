# Utilisez une image de base Python
FROM python:3.11-slim-buster

# Insérez cette ligne au début de votre Dockerfile
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential wget

# Remove any existing sqlite3 packages
RUN apt-get purge -y sqlite3 libsqlite3-dev

# Install SQLite from source
RUN cd /tmp && \
    wget https://www.sqlite.org/2023/sqlite-autoconf-3440000.tar.gz && \
    tar xzf sqlite-autoconf-3440000.tar.gz && \
    cd sqlite-autoconf-3440000 && \
    ./configure --prefix=/usr/local && \
    make && make install && \
    ldconfig /usr/local/lib


# Définissez le répertoire de travail
WORKDIR /usr/src/app

# Copiez les fichiers nécessaires
COPY requirements.txt .
COPY LUCY_CODE_COMPLET.py .

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Vous pouvez également installer pysqlite3-binary si nécessaire
# RUN pip install --no-cache-dir pysqlite3-binary

# Commande pour exécuter votre application
CMD ["uvicorn", "LUCY_CODE_COMPLET:app", "--host", "0.0.0.0", "--port", "8000"]








