import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
# from prometheus_metrics import *
# import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-16-171-20-109.eu-north-1.compute.amazonaws.com:5000/")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "22", "./tfidf_vectorizer.pkl")

# @app.before_request
# def before_request():
#     request.start_time = time.time()

# @app.after_request
# def after_request(response):
#     latency = time.time() - request.start_time
#     REQUEST_LATENCY.labels(
#         method=request.method,
#         endpoint=request.path
#     ).observe(latency)
    
#     REQUEST_COUNT.labels(
#         method=request.method,
#         endpoint=request.path,
#         http_status=response.status_code
#     ).inc()
    
#     if response.status_code >= 400:
#         ERROR_COUNT.labels(error_type=str(response.status_code)).inc()
    
#     return response

# @app.route('/metrics')
# def metrics():
#     from prometheus_client import generate_latest
#     return generate_latest(), 200, {'Content-Type': 'text/plain'}

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():    
    # Ensure model and vectorizer loaded correctly during startup
    if not model or not vectorizer:
        return jsonify({"error": "Model or Vectorizer not loaded. Check server logs."}), 503 # Service Unavailable

    data = request.json
    if not data:
        return jsonify({"error": "Request body is missing or not JSON"}), 400

    comments_data = data.get('comments') # Expects list of {'text': str, 'timestamp': any}

    if not comments_data:
        return jsonify({"error": "No 'comments' key found in JSON payload"}), 400
    if not isinstance(comments_data, list):
        return jsonify({"error": "'comments' should be a list of dictionaries"}), 400
    # Add a basic check for structure - refine if stricter validation is needed
    if not all(isinstance(item, dict) and 'text' in item and 'timestamp' in item for item in comments_data):
         app.logger.warning("Received items in 'comments' list that are not dictionaries or missing 'text'/'timestamp'. Processing valid items.")
         # Filter for valid items only
         original_comments_data = [item for item in comments_data if isinstance(item, dict) and 'text' in item and 'timestamp' in item]
         if not original_comments_data:
             return jsonify({"error": "No valid comment data found in 'comments' list after filtering"}), 400
         app.logger.info(f"Filtered down to {len(original_comments_data)} valid comment data items.")
    else:
         original_comments_data = comments_data # Use original list if all items are valid

    app.logger.info(f"Received {len(original_comments_data)} valid comments with timestamps for prediction.")

    try:
        # Extract original comments and timestamps from the potentially filtered list
        original_comments = [item.get('text', '') for item in original_comments_data]
        timestamps = [item.get('timestamp') for item in original_comments_data]

        # Preprocess each comment before vectorizing
        app.logger.info("Starting preprocessing...")
        preprocessed_comments = [preprocess_comment(comment) for comment in original_comments]
        app.logger.info("Preprocessing finished.")

        # --- Filtering based on non-empty preprocessed comments ---
        # Keep track of indices relative to the 'original_comments_data' list
        valid_indices = [i for i, p_comment in enumerate(preprocessed_comments) if p_comment]

        if not valid_indices:
             app.logger.warning("All valid comments became empty after preprocessing.")
             # Construct response based on original input data before filtering
             response = [{"comment": item.get('text', ''), "sentiment": "N/A - Empty after preprocessing", "timestamp": item.get('timestamp')} for item in comments_data]
             return jsonify(response)

        # Filter the preprocessed comments that will be vectorized
        preprocessed_comments_filtered = [preprocessed_comments[i] for i in valid_indices]

        # Transform the non-empty comments using the vectorizer
        app.logger.info(f"Vectorizing {len(preprocessed_comments_filtered)} non-empty comments...")
        transformed_comments_sparse = vectorizer.transform(preprocessed_comments_filtered)
        app.logger.info(f"Vectorization finished. Sparse matrix shape: {transformed_comments_sparse.shape}")

        # --- Conversion to DataFrame ---
        app.logger.info("Converting sparse matrix to Pandas DataFrame...")
        transformed_comments_dense = transformed_comments_sparse.toarray()
        try:
            # Use get_feature_names_out() for sklearn >= 0.24
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            feature_names = vectorizer.get_feature_names()
        input_df = pd.DataFrame(transformed_comments_dense, columns=feature_names)
        app.logger.info(f"DataFrame created. Shape: {input_df.shape}")
        # --- End Conversion ---

        # Make predictions using the DataFrame
        app.logger.info("Making predictions with the model...")
        predictions_array = model.predict(input_df)
        predictions_list = predictions_array.tolist() # Convert numpy array (if applicable) to list
        app.logger.info("Predictions finished.")

        # Convert predictions to strings for consistency
        predictions_str_list = [str(pred) for pred in predictions_list]

    except mlflow.exceptions.MlflowException as mle:
        app.logger.error(f"MLflow prediction failed: {str(mle)}", exc_info=True)
        # Log input data type and shape again for clarity
        input_type = type(input_df) if 'input_df' in locals() else 'N/A'
        input_shape = input_df.shape if 'input_df' in locals() and isinstance(input_df, pd.DataFrame) else 'N/A'
        app.logger.error(f"Data type passed to predict: {input_type}, shape: {input_shape}")
        return jsonify({"error": f"MLflow prediction failed: {str(mle)}"}), 500
    except Exception as e:
        app.logger.error(f"Generic prediction failed: {str(e)}", exc_info=True) # Log traceback
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # --- Map predictions back to the original full structure ---
    # Create a dictionary mapping valid indices (from original_comments_data) to predictions
    prediction_map = {index: prediction for index, prediction in zip(valid_indices, predictions_str_list)}

    # Build the final response list based on the *original* comments_data received
    response = []
    for i, item_data in enumerate(original_comments_data): # Iterate through the filtered valid items
         original_comment_text = item_data.get('text', '')
         timestamp_val = item_data.get('timestamp')
         # Check if the current index 'i' corresponds to a comment that was processed
         if i in prediction_map:
              sentiment = prediction_map[i]
         else:
              # This case handles comments that were valid input but became empty after preprocessing
              sentiment = "N/A - Empty after preprocessing"
         response.append({"comment": original_comment_text, "sentiment": sentiment, "timestamp": timestamp_val})

    # If you filtered invalid items at the start, you might want to decide how to represent them
    # Option 1 (as done above): Only return results for initially valid items.
    # Option 2: If you need to return *all* original items, including invalid ones:
    # final_response = []
    # valid_items_iter = iter(response)
    # for item in comments_data: # Iterate through the very original input
    #     if isinstance(item, dict) and 'text' in item and 'timestamp' in item:
    #         final_response.append(next(valid_items_iter))
    #     else:
    #         final_response.append({"comment": str(item), "sentiment": "N/A - Invalid input format", "timestamp": None})
    # response = final_response # Uncomment this block and comment out the line below if you need Option 2

    app.logger.info(f"Returning {len(response)} predictions with timestamps.")
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model and vectorizer loaded correctly during startup
    if not model or not vectorizer:
         return jsonify({"error": "Model or Vectorizer not loaded. Check server logs."}), 503 # Service Unavailable

    data = request.json
    if not data:
         return jsonify({"error": "Request body is missing or not JSON"}), 400

    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No 'comments' key found in JSON payload"}), 400
    if not isinstance(comments, list):
         return jsonify({"error": "'comments' should be a list of strings"}), 400
    if not all(isinstance(c, str) for c in comments):
        app.logger.warning("Received non-string elements in comments list. Attempting preprocessing anyway.")
        # Optionally filter out non-strings or return an error
        # comments = [c for c in comments if isinstance(c, str)]

    app.logger.info(f"Received {len(comments)} comments for prediction.")

    try:
        # Preprocess each comment before vectorizing
        app.logger.info("Starting preprocessing...")
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        app.logger.info("Preprocessing finished.")

        # Filter out any comments that became empty after preprocessing
        valid_indices = [i for i, p_comment in enumerate(preprocessed_comments) if p_comment]
        original_comments_filtered = [comments[i] for i in valid_indices]
        preprocessed_comments_filtered = [preprocessed_comments[i] for i in valid_indices]

        if not preprocessed_comments_filtered:
            app.logger.warning("All comments became empty after preprocessing.")
            # Return empty predictions or a specific message
            results = [{"comment": c, "sentiment": "N/A - Empty after preprocessing"} for c in comments]
            return jsonify(results)

        # Transform comments using the vectorizer
        app.logger.info(f"Vectorizing {len(preprocessed_comments_filtered)} non-empty comments...")
        transformed_comments_sparse = vectorizer.transform(preprocessed_comments_filtered)
        app.logger.info(f"Vectorization finished. Sparse matrix shape: {transformed_comments_sparse.shape}")

        # --- Conversion to DataFrame ---
        app.logger.info("Converting sparse matrix to Pandas DataFrame...")
        # Convert sparse matrix to dense numpy array first
        transformed_comments_dense = transformed_comments_sparse.toarray()

        # Get feature names from the vectorizer (handle potential version differences)
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = vectorizer.get_feature_names() # Fallback for older sklearn

        # Create the Pandas DataFrame
        input_df = pd.DataFrame(transformed_comments_dense, columns=feature_names)
        app.logger.info(f"DataFrame created. Shape: {input_df.shape}")
        # Log first few rows/columns for debugging if needed
        # app.logger.debug(f"DataFrame head:\n{input_df.head()}")
        # app.logger.debug(f"DataFrame columns sample: {input_df.columns[:10].tolist()}...")
        # --- End Conversion ---

        # Make predictions using the DataFrame
        app.logger.info("Making predictions with the model...")
        predictions_array = model.predict(input_df)
        predictions = predictions_array.tolist()  # Convert numpy array (if applicable) to list
        app.logger.info("Predictions finished.")

        # Convert predictions to strings for consistency (if they aren't already)
        predictions = [str(pred) for pred in predictions]

    except mlflow.exceptions.MlflowException as mle:
         # Catch MLflow specific exceptions for better debugging
         app.logger.error(f"MLflow prediction failed: {str(mle)}", exc_info=True)
         # Log input data type and shape again for clarity
         app.logger.error(f"Data type passed to predict: {type(input_df)}, shape: {input_df.shape if isinstance(input_df, pd.DataFrame) else 'N/A'}")
         return jsonify({"error": f"MLflow prediction failed: {str(mle)}"}), 500
    except Exception as e:
        app.logger.error(f"Generic prediction failed: {str(e)}", exc_info=True) # Log traceback
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Map predictions back to original comments, handling empty ones
    predictions_iter = iter(predictions)
    response = []
    for i, original_comment in enumerate(comments):
        if i in valid_indices:
            response.append({"comment": original_comment, "sentiment": next(predictions_iter)})
        else:
            response.append({"comment": original_comment, "sentiment": "N/A - Empty after preprocessing"})

    app.logger.info(f"Returning {len(response)} predictions.")
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)