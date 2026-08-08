"""
Monitoring du data drift avec Evidently AI
A executer periodiquement (cron, Airflow, ou GitHub Actions)
"""
import pandas as pd
import json
from datetime import datetime
from sqlalchemy import create_engine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pathlib import Path

# Configuration
REFERENCE_DATA_PATH = "data/credits_bancaires.csv"
DATABASE_URL = "sqlite:///./scoring_audit.db"
REPORTS_DIR = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement des donnees de reference...")
reference_data = pd.read_csv(REFERENCE_DATA_PATH)

print("Chargement des donnees de production depuis la DB...")
engine = create_engine(DATABASE_URL)
try:
    production_logs = pd.read_sql(
        "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1000",
        engine
    )

    features_records = []
    for _, row in production_logs.iterrows():
        features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
        features['defaut'] = 1 if row['probabilite_defaut'] > 0.45 else 0
        features_records.append(features)

    production_data = pd.DataFrame(features_records)

    print(f"Reference: {len(reference_data)} lignes")
    print(f"Production: {len(production_data)} lignes")

    print("\nGeneration du rapport de drift...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=production_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    drift_report.save_html(str(report_path))

    drift_dict = drift_report.as_dict()
    dataset_drift = drift_dict['metrics'][0]['result']['dataset_drift']
    drifted_features = drift_dict['metrics'][0]['result'].get('number_of_drifted_columns', 0)
    total_features = drift_dict['metrics'][0]['result'].get('number_of_columns', 0)

    print(f"\n{'='*60}")
    print("RESULTATS DU MONITORING")
    print(f"{'='*60}")
    print(f"Drift detecte: {'OUI' if dataset_drift else 'NON'}")
    print(f"Features driftees: {drifted_features}/{total_features}")
    print(f"Rapport sauvegarde: {report_path}")
    print(f"{'='*60}")

    if dataset_drift:
        print("\nALERTE: Data drift detecte !")
        print("   Actions recommandees:")
        print("   1. Analyser les features driftees dans le rapport HTML")
        print("   2. Verifier la qualite des donnees en entree")
        print("   3. Planifier un retraining du modele")
    else:
        print("\nPas de drift detecte. Le modele reste stable.")

except Exception as e:
    print(f"Erreur lors du monitoring: {e}")
    print("   Verifiez que la base de donnees contient des predictions.")