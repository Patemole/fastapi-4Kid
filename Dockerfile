# Utilisez une image de base Python
FROM python:3.11-slim-buster

# Mettre à jour les paquets et installer les dépendances nécessaires
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential wget

# Supprimer tout paquet sqlite3 existant
RUN apt-get purge -y sqlite3 libsqlite3-dev

# Installer SQLite à partir des sources
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
COPY main.py .

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Commande pour exécuter votre application
# Remarque : Hypercorn utilise un format de commande différent. Assurez-vous que `main.py` contient la création de l'instance app
#CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]
#CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:$PORT"]
#CMD ["sh", "-c", "hypercorn main:app --bind \"0.0.0.0:$PORT\""]





