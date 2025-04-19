import os
import mlflow
from mlflow.exceptions import MlflowException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def promote_model():
    try:
        mlflow.set_tracking_uri("http://ec2-13-60-211-129.eu-north-1.compute.amazonaws.com:5000/")
        client = mlflow.MlflowClient()
        model_name = "yt_chrome_plugin_model"

        # Get current "production" model (if exists)
        try:
            current_prod = client.get_model_version_by_alias(model_name, "production")
            logger.info(f"Current Production Model: v{current_prod.version} (Run ID: {current_prod.run_id})")
        except MlflowException:
            current_prod = None

        # Get candidate model (previously "staging")
        candidate = client.get_model_version_by_alias(model_name, "staging")
        logger.info(f"Promoting Candidate Model: v{candidate.version} (Run ID: {candidate.run_id})")

        # Update aliases
        if current_prod:
            # Move current prod to "archived" alias
            client.set_model_version_alias(model_name, current_prod.version, "archived")
            logger.info(f"Archived v{current_prod.version}")

        # Promote candidate to production
        client.set_model_version_alias(model_name, candidate.version, "production")
        logger.info(f"Promoted v{candidate.version} to production alias")

        # Optional: Remove staging alias
        client.delete_model_version_alias(model_name, candidate.version, "staging")

        return True

    except MlflowException as e:
        logger.error(f"MLflow Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = promote_model()
    exit(0 if success else 1)