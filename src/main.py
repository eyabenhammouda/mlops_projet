import argparse
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from elasticsearch import Elasticsearch
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
# 🔗 Connexion à Elasticsearch
def connect_to_elasticsearch():
    """Établit une connexion à Elasticsearch"""
    try:
        es = Elasticsearch("http://172.18.0.2:9200", verify_certs=False) 
        if es.ping():
            print("✅ Connexion réussie à Elasticsearch")
            return es
        else:
            print("❌ Impossible de se connecter à Elasticsearch")
            return None
    except Exception as e:   
        print(f"⚠️ Erreur de connexion à Elasticsearch : {e}")
        return None
        

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # URI de suivi MLflow
mlflow.set_experiment("Churn Prediction")  # Nom de l'expérience MLflow
MODEL_NAME = "Churn_Prediction_Model"  # Nom du modèle dans le Model Registry
# 📡 Envoi des logs à Elasticsearch
def log_to_elasticsearch(run_id, stage, metrics, params):
    """Envoie les logs MLflow à Elasticsearch"""
    es = connect_to_elasticsearch()
    if es is None:
        print("🚨 Elasticsearch non disponible, logs non envoyés.")
        return

    log_data = {
        "run_id": run_id,
        "stage": stage,
        "metrics": metrics,
        "params": params
    }

    try:
        es.index(index="mlflow-metriques", body=log_data)  # Correction du Content-Type automatique
        print(f"📡 Logs envoyés à Elasticsearch : {log_data}")
    except Exception as e:
        print(f"⚠️ Échec de l'envoi des logs à Elasticsearch : {e}")


def list_model_versions():
    """Lister toutes les versions du modèle enregistrées dans MLflow."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print("\n📌 *Versions du modèle enregistrées dans MLflow:*")
    for v in versions:
        print(f"🔹 Version: {v.version}, Status: {v.current_stage}, Run ID: {v.run_id}, Date: {v.creation_timestamp}")

def transition_model_stage(model_name, model_version, stage):
    """Changer le stage du modèle (Staging, Production, Archived)"""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage
    )
    logging.info(f"Modèle {model_name} (version {model_version}) est maintenant en '{stage}' !")

def main():
    """
    Pipeline de Machine Learning pour la prédiction du churn :
    - Prétraitement des données
    - Entraînement du modèle
    - Évaluation
    - Enregistrement du modèle dans MLflow
    - Gestion des versions dans le Model Registry
    """
    parser = argparse.ArgumentParser(description="Pipeline de prédiction du churn.")

    # Arguments du script
    parser.add_argument("--prepare", action="store_true", help="Préparer les données.")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle.")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle.")
    parser.add_argument("--save", type=str, help="Sauvegarder le modèle dans un fichier.")
    parser.add_argument("--load", type=str, help="Charger un modèle existant.")
    parser.add_argument("--train_path", type=str, required=True, help="Chemin du fichier CSV d'entraînement.")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin du fichier CSV de test.")
    parser.add_argument("--stage", type=str, choices=["Staging", "Production", "Archived"], help="Stage auquel promouvoir le modèle.")
    
    args = parser.parse_args()

    logging.info("🔄 Préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)

    with mlflow.start_run() as run:
        # Entraînement du modèle
        logging.info("🚀 Entraînement du modèle...")
        model = train_model(X_train, y_train)

        # Enregistrement des paramètres et du modèle dans MLflow
        mlflow.sklearn.log_model(model, "churn_model")

        # Récupération du modèle à partir de l'URI du run
        model_uri = f"runs:/{run.info.run_id}/churn_model"
        logging.info(f"💾 Modèle enregistré sous l'URI : {model_uri}")

        # Ajout du modèle au Model Registry avec une nouvelle version
        logging.info("📥 Enregistrement du modèle dans le Model Registry...")
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        logging.info(f"✅ Modèle '{MODEL_NAME}' enregistré avec la version {registered_model.version}.")

        # Afficher les versions enregistrées
        list_model_versions()

        # Calcul des métriques
        logging.info("📊 Évaluation du modèle...")
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        # Afficher les résultats
        result_message = (
            f"✅ Résultats de l'évaluation :\n"
            f"- Accuracy: {accuracy:.4f}\n"
            f"- Precision: {precision:.4f}\n"
            f"- Recall: {recall:.4f}\n"
            f"- F1-score: {f1:.4f}"
        )
        logging.info(result_message)
        print(result_message)

        # Loguer les métriques dans MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        # 📡 Envoi des logs à Elasticsearch
        log_to_elasticsearch(
            run.info.run_id,
            args.stage,
            {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1},
            {"n_estimators": 100, "max_depth":1 }
        )

        # Promotion du modèle au stage spécifié par l'utilisateur
        if args.stage:
            transition_model_stage(MODEL_NAME, registered_model.version, args.stage)

if __name__ == "__main__":
    main()
