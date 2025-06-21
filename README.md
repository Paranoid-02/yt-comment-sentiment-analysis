# YouTube Comment Sentiment Analysis

This project provides an end-to-end MLOps solution for analyzing the sentiment of YouTube comments. It includes a machine learning pipeline to train a sentiment analysis model, a Flask API to serve the model, and a CI/CD pipeline for automated building, testing, and deployment.

## Features

- **Sentiment Analysis:** Predicts the sentiment of YouTube comments (Positive, Negative, Neutral).
- **ML Pipeline with DVC:** Uses Data Version Control (DVC) to manage the machine learning pipeline, including data ingestion, preprocessing, model training, and evaluation.
- **REST API:** A Flask-based REST API to serve the trained model and provide sentiment predictions.
- **API Endpoints:**
    - `/predict`: Get sentiment for a list of comments.
    - `/predict_with_timestamps`: Get sentiment for comments with timestamps.
    - `/generate_chart`: Get a pie chart of sentiment distribution.
    - `/generate_wordcloud`: Get a word cloud of comment text.
    - `/generate_trend_graph`: Get a sentiment trend graph over time.
- **Containerization:** The application is containerized using Docker for portability and scalability.
- **CI/CD:** A Jenkins pipeline automates the entire process from code checkout to deployment.
- **Orchestration:** Deployed to Kubernetes using Ansible for automation.
- **Monitoring:** (Future Goal) Basic ELK Stack using Docker Compose.

## Project Structure

```
├── ansible/            # Ansible playbooks for deployment
├── data/               # Project data (managed by DVC)
├── docs/               # Project documentation
├── elk/                # Elasticsearch, Logstash, Kibana setup
├── flask_app/          # Flask application code
│   └── app.py
├── kubernetes/         # Kubernetes manifests
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for exploration
├── reports/            # Generated reports and figures
├── scripts/            # Helper scripts (testing, etc.)
├── src/                # Source code for the ML pipeline
├── .dvc/               # DVC metadata
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Parameters for the ML pipeline
├── jenkinsfile         # Jenkins CI/CD pipeline definition
├── dockerfile          # Dockerfile for building the application image
├── requirements.txt    # Python dependencies
└── README.md
```

## Machine Learning Pipeline

The machine learning pipeline is managed by DVC and defined in `dvc.yaml`. It consists of the following stages:

1.  **data_ingestion:** Ingests the raw data.
2.  **data_preprocessing:** Cleans and preprocesses the text data.
3.  **model_building:** Trains a LightGBM model with TF-IDF vectorization. Hyperparameters are defined in `params.yaml`.
4.  **model_evaluation:** Evaluates the model performance.
5.  **model_registration:** Registers the model in the MLflow model registry.

## Tech Stack

- **Machine Learning:** Scikit-learn, LightGBM, Pandas, Numpy, NLTK
- **MLOps:** DVC, MLflow
- **Web Framework:** Flask
- **Containerization:** Docker
- **CI/CD:** Jenkins
- **Orchestration:** Kubernetes
- **Automation:** Ansible
- **Monitoring:** ELK Stack (Elasticsearch, Logstash, Kibana)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Paranoid-02/yt-comment-sentiment-analysis.git
    cd yt-comment-sentiment-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up DVC:**
    ```bash
    dvc pull
    ```

## Usage

### Run the Flask API locally

```bash
flask run
```

The API will be available at `http://127.0.0.1:5000`.

### Run with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t yt-sentiment-analysis .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 yt-sentiment-analysis
    ```

### API Usage Example (cURL)

**Predict Sentiment:**

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"comments": ["This is a great video!", "I did not like this at all."]}' \
http://localhost:5000/predict
```

## CI/CD Pipeline

The project uses a `jenkinsfile` to define the CI/CD pipeline. The pipeline automates the following steps:

1.  **Code Checkout:** Clones the Git repository.
2.  **Environment Setup:** Sets up the Python environment and installs dependencies.
3.  **DVC Pipeline:** Runs the DVC pipeline to reproduce the model.
4.  **Testing:** Runs unit and integration tests.
5.  **Docker Build & Push:** Builds and pushes the Docker image to a registry.
6.  **Deployment:** Deploys the application to Kubernetes using Ansible.
7.  **Monitoring Setup:** Deploys the ELK stack for logging and monitoring.
