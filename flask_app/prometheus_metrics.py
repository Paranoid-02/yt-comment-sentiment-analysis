from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    'flask_app_request_count',
    'Application Request Count',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'flask_app_request_latency_seconds',
    'Application Request Latency',
    ['method', 'endpoint']
)

MODEL_PREDICTION_TIME = Histogram(
    'model_prediction_latency_seconds',
    'ML Model Prediction Latency'
)

ERROR_COUNT = Counter(
    'flask_app_error_count',
    'Application Error Count',
    ['error_type']
)