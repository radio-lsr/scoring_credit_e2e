#!/bin/bash
# Démarrage de l'API en arrière-plan
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Attente que l'API soit prête
sleep 3

# Démarrage du frontend Streamlit
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0