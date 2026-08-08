import os
import urllib.request

MODELS_URL = "https://votre-bucket.s3.amazonaws.com/models.zip"  # ou Google Drive

def download():
    if not os.path.exists("models/model_gradient_boosting.pkl"):
        print("📥 Téléchargement des modèles...")
        urllib.request.urlretrieve(MODELS_URL, "models.zip")
        import zipfile
        with zipfile.ZipFile("models.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("models.zip")
        print("✅ Modèles prêts")

if __name__ == "__main__":
    download()