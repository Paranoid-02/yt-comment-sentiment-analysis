import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

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
import logging # Import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO) # Log basic info and errors

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Ensure comment is a string
        if not isinstance(comment, str):
            # Attempt to convert or handle non-string input gracefully
            app.logger.warning(f"Received non-string comment: {type(comment)}. Attempting conversion.")
            comment = str(comment) # Or return an error, or a default empty string

        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation (!?.,)
        # Keep spaces to separate words
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment) # Use a-z since we lowercased

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet', 'against', 'up', 'down', 'very'} # Added a few more potentially useful sentiment words
        words = comment.split()
        comment = ' '.join([word for word in words if word not in stop_words and word.strip() != '']) # Ensure empty strings aren't joined

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = comment.split()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in words])

        return comment
    except Exception as e:
        app.logger.error(f"Error in preprocessing comment '{comment[:50]}...': {e}")
        return "" # Return empty string on error to avoid issues downstream

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    try:
        # Set MLflow tracking URI to your server
        mlflow.set_tracking_uri("http://ec2-13-61-21-255.eu-north-1.compute.amazonaws.com:5000/") # Make sure this is reachable from where the Flask app runs
        app.logger.info(f"Connecting to MLflow at {mlflow.get_tracking_uri()}")
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        app.logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        app.logger.info("Model loaded successfully.")
        app.logger.info(f"Loading vectorizer from path: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
        app.logger.info("Vectorizer loaded successfully.")
        # Check vectorizer type
        app.logger.info(f"Vectorizer type: {type(vectorizer)}")
        # Log number of features expected by vectorizer
        try:
             # Use get_feature_names_out() for sklearn >= 0.24
            feature_names = vectorizer.get_feature_names_out()
            app.logger.info(f"Vectorizer expects {len(feature_names)} features.")
        except AttributeError:
             # Fallback for older sklearn versions
            feature_names = vectorizer.get_feature_names()
            app.logger.info(f"Vectorizer expects {len(feature_names)} features (using older get_feature_names()).")
        except Exception as e:
            app.logger.error(f"Could not get feature names from vectorizer: {e}")

        return model, vectorizer
    except Exception as e:
        app.logger.error(f"Failed to load model or vectorizer: {e}")
        # Potentially raise the exception or exit if loading fails critically
        raise e


# Initialize the model and vectorizer
# Make sure './tfidf_vectorizer.pkl' exists in the same directory as your app.py
# or provide the correct absolute or relative path.
try:
    model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "4", "./tfidf_vectorizer.pkl")
except Exception as e:
    app.logger.error(f"CRITICAL: Application startup failed - could not load model/vectorizer. Error: {e}")
    # Exit or handle this critical failure appropriately in a real application
    model, vectorizer = None, None # Set to None so app doesn't crash immediately if run directly

@app.route('/')
def home():
    return "Welcome to our flask api"

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

# @app.route('/generate_chart', methods=['POST'])
# def generate_chart():
#     try:
#         data = request.get_json()
#         sentiment_counts = data.get('sentiment_counts')
        
#         if not sentiment_counts:
#             return jsonify({"error": "No sentiment counts provided"}), 400

#         # Prepare data for the pie chart
#         labels = ['Positive', 'Neutral', 'Negative']
#         sizes = [
#             int(sentiment_counts.get('1', 0)),
#             int(sentiment_counts.get('0', 0)),
#             int(sentiment_counts.get('-1', 0))
#         ]
#         if sum(sizes) == 0:
#             raise ValueError("Sentiment counts sum to zero")
        
#         colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

#         # Generate the pie chart
#         plt.figure(figsize=(6, 6))
#         plt.pie(
#             sizes,
#             labels=labels,
#             colors=colors,
#             autopct='%1.1f%%',
#             startangle=140,
#             textprops={'color': 'w'}
#         )
#         plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#         # Save the chart to a BytesIO object
#         img_io = io.BytesIO()
#         plt.savefig(img_io, format='PNG', transparent=True)
#         img_io.seek(0)
#         plt.close()

#         # Return the image as a response
#         return send_file(img_io, mimetype='image/png')
#     except Exception as e:
#         app.logger.error(f"Error in /generate_chart: {e}")
#         return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

# @app.route('/generate_wordcloud', methods=['POST'])
# def generate_wordcloud():
#     try:
#         data = request.get_json()
#         comments = data.get('comments')

#         if not comments:
#             return jsonify({"error": "No comments provided"}), 400

#         # Preprocess comments
#         preprocessed_comments = [preprocess_comment(comment) for comment in comments]

#         # Combine all comments into a single string
#         text = ' '.join(preprocessed_comments)

#         # Generate the word cloud
#         wordcloud = WordCloud(
#             width=800,
#             height=400,
#             background_color='black',
#             colormap='Blues',
#             stopwords=set(stopwords.words('english')),
#             collocations=False
#         ).generate(text)

#         # Save the word cloud to a BytesIO object
#         img_io = io.BytesIO()
#         wordcloud.to_image().save(img_io, format='PNG')
#         img_io.seek(0)

#         # Return the image as a response
#         return send_file(img_io, mimetype='image/png')
#     except Exception as e:
#         app.logger.error(f"Error in /generate_wordcloud: {e}")
#         return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

# @app.route('/generate_trend_graph', methods=['POST'])
# def generate_trend_graph():
#     try:
#         data = request.get_json()
#         sentiment_data = data.get('sentiment_data')

#         if not sentiment_data:
#             return jsonify({"error": "No sentiment data provided"}), 400

#         # Convert sentiment_data to DataFrame
#         df = pd.DataFrame(sentiment_data)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])

#         # Set the timestamp as the index
#         df.set_index('timestamp', inplace=True)

#         # Ensure the 'sentiment' column is numeric
#         df['sentiment'] = df['sentiment'].astype(int)

#         # Map sentiment values to labels
#         sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

#         # Resample the data over monthly intervals and count sentiments
#         monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

#         # Calculate total counts per month
#         monthly_totals = monthly_counts.sum(axis=1)

#         # Calculate percentages
#         monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

#         # Ensure all sentiment columns are present
#         for sentiment_value in [-1, 0, 1]:
#             if sentiment_value not in monthly_percentages.columns:
#                 monthly_percentages[sentiment_value] = 0

#         # Sort columns by sentiment value
#         monthly_percentages = monthly_percentages[[-1, 0, 1]]

#         # Plotting
#         plt.figure(figsize=(12, 6))

#         colors = {
#             -1: 'red',     # Negative sentiment
#             0: 'gray',     # Neutral sentiment
#             1: 'green'     # Positive sentiment
#         }

#         for sentiment_value in [-1, 0, 1]:
#             plt.plot(
#                 monthly_percentages.index,
#                 monthly_percentages[sentiment_value],
#                 marker='o',
#                 linestyle='-',
#                 label=sentiment_labels[sentiment_value],
#                 color=colors[sentiment_value]
#             )

#         plt.title('Monthly Sentiment Percentage Over Time')
#         plt.xlabel('Month')
#         plt.ylabel('Percentage of Comments (%)')
#         plt.grid(True)
#         plt.xticks(rotation=45)

#         # Format the x-axis dates
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

#         plt.legend()
#         plt.tight_layout()

#         # Save the trend graph to a BytesIO object
#         img_io = io.BytesIO()
#         plt.savefig(img_io, format='PNG')
#         img_io.seek(0)
#         plt.close()

#         # Return the image as a response
#         return send_file(img_io, mimetype='image/png')
#     except Exception as e:
#         app.logger.error(f"Error in /generate_trend_graph: {e}")
#         return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)