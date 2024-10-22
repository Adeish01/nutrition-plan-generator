import logging
import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import plotly.graph_objects as go
import time
from streamlit_lottie import st_lottie
import requests

# Set up logging to track the program's execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    """
    Load the nutrition dataset and preprocess it for machine learning.
    
    This function performs the following steps:
    1. Load the dataset from Hugging Face
    2. Convert categorical variables to numerical using LabelEncoder
    3. Convert numerical columns to the appropriate data type
    4. Remove rows with missing values
    
    Returns:
    - dataset: Preprocessed pandas DataFrame
    - le_dict: Dictionary of LabelEncoders for categorical variables
    """
    logging.info("Loading dataset...")
    ds = load_dataset("sarthak-wiz01/nutrition_dataset")
    dataset = pd.DataFrame(ds['train'])
    logging.info(f"Dataset loaded. Shape: {dataset.shape}")

    logging.info("Preprocessing data...")
    le_dict = {}
    categorical_columns = ['Gender', 'Activity Level', 'Fitness Goal', 'Dietary Preference']
    for col in categorical_columns:
        le_dict[col] = LabelEncoder()
        dataset[col] = le_dict[col].fit_transform(dataset[col])
        logging.info(f"Encoded {col}: {le_dict[col].classes_}")

    numerical_columns = ['Age', 'Height', 'Weight', 'Daily Calorie Target', 'Protein', 'Carbohydrates', 'Fat']
    for col in numerical_columns:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

    initial_rows = len(dataset)
    dataset = dataset.dropna()
    removed_rows = initial_rows - len(dataset)
    logging.info(f"Removed {removed_rows} rows with NaN values. Remaining rows: {len(dataset)}")

    return dataset, le_dict

def train_model(dataset):
    """
    Train a Random Forest Regressor model on the preprocessed dataset.
    
    This function performs the following steps:
    1. Prepare the features (X) and target variables (y)
    2. Split the data into training and testing sets
    3. Perform hyperparameter tuning using RandomizedSearchCV
    4. Train the model with the best hyperparameters
    5. Evaluate the model using Mean Absolute Error and R-squared score
    6. Perform cross-validation to ensure model robustness
    
    Args:
    - dataset: Preprocessed pandas DataFrame
    
    Returns:
    - best_model: Trained Random Forest Regressor model
    """
    logging.info("Preparing data for model training...")
    X = dataset[['Age', 'Gender', 'Height', 'Weight', 'Activity Level', 'Fitness Goal', 'Dietary Preference']]
    y = dataset[['Daily Calorie Target', 'Protein', 'Carbohydrates', 'Fat']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    logging.info("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                       n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    logging.info(f"Best parameters: {random_search.best_params_}")

    logging.info("Evaluating model performance...")
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Absolute Error: {mae:.2f}")
    logging.info(f"R-squared Score: {r2:.2f}")

    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    logging.info(f"Cross-validation R2 scores: {cv_scores}")
    logging.info(f"Mean CV R2 score: {np.mean(cv_scores):.2f}")

    return best_model

def encode_user_input(user_profile, le_dict):
    """
    Encode the user input using the same LabelEncoders used for training.
    
    Args:
    - user_profile: Dictionary containing user input
    - le_dict: Dictionary of LabelEncoders for categorical variables
    
    Returns:
    - encoded_profile: Dictionary with encoded user input
    """
    encoded_profile = user_profile.copy()
    for col, le in le_dict.items():
        if user_profile[col] in le.classes_:
            encoded_profile[col] = le.transform([user_profile[col]])[0]
        else:
            logging.warning(f"Unseen label in {col}: {user_profile[col]}. Assigning new category.")
            encoded_profile[col] = len(le.classes_)
    return encoded_profile

def generate_nutrition_plan(user_profile, model, le_dict):
    """
    Generate a personalized nutrition plan based on user input.
    
    Args:
    - user_profile: Dictionary containing user input
    - model: Trained Random Forest Regressor model
    - le_dict: Dictionary of LabelEncoders for categorical variables
    
    Returns:
    - plan: Dictionary containing the predicted nutrition plan
    """
    logging.info(f"Generating nutrition plan for user profile: {user_profile}")
    encoded_profile = encode_user_input(user_profile, le_dict)
    prediction = model.predict([list(encoded_profile.values())])
    
    plan = {
        'Daily Calorie Target': round(prediction[0][0]),
        'Protein': round(prediction[0][1]),
        'Carbohydrates': round(prediction[0][2]),
        'Fat': round(prediction[0][3])
    }
    logging.info(f"Generated nutrition plan: {plan}")
    return plan

def plot_feature_importance(model, feature_names):
    """
    Create a bar plot of feature importances.
    
    Args:
    - model: Trained Random Forest Regressor model
    - feature_names: List of feature names
    
    Returns:
    - fig: Plotly Figure object
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure(go.Bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        orientation='v'
    ))
    fig.update_layout(
        title="Feature Importances",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=500
    )
    return fig

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        return r.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"Error loading Lottie animation: {e}")
        return None

def main():
    """
    Main function to run the Streamlit app.
    
    This function sets up the user interface, handles user input,
    generates the nutrition plan, and displays the results.
    """
    st.title("Personalized Nutrition Plan Generator")

    # Load Lottie animation
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_X8aNub.json"
    lottie_json = load_lottieurl(lottie_url)

    # Create a placeholder for the animation or progress bar
    loading_placeholder = st.empty()

    with st.spinner("Loading data and training model..."):
        if lottie_json:
            # Display the Lottie animation if available
            with loading_placeholder:
                st_lottie(lottie_json, speed=1, height=200, key="loading")
        else:
            # Fallback to a progress bar if Lottie animation is not available
            progress_bar = loading_placeholder.progress(0)

        # Load and preprocess data
        dataset, le_dict = load_and_preprocess_data()
        if not lottie_json:
            progress_bar.progress(50)
        
        # Train the model
        model = train_model(dataset)
        if not lottie_json:
            progress_bar.progress(100)

    # Clear the loading placeholder
    loading_placeholder.empty()

    # Create user input form
    st.header("Enter Your Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", options=le_dict['Gender'].classes_)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    activity_level = st.selectbox("Activity Level", options=le_dict['Activity Level'].classes_)
    fitness_goal = st.selectbox("Fitness Goal", options=le_dict['Fitness Goal'].classes_)
    dietary_preference = st.selectbox("Dietary Preference", options=le_dict['Dietary Preference'].classes_)

    user_profile = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'Activity Level': activity_level,
        'Fitness Goal': fitness_goal,
        'Dietary Preference': dietary_preference
    }

    # Generate and display nutrition plan
    if st.button("Generate Nutrition Plan"):
        nutrition_plan = generate_nutrition_plan(user_profile, model, le_dict)
        st.subheader("Your Personalized Nutrition Plan")
        for key, value in nutrition_plan.items():
            st.write(f"{key}: {value}")

    # Display model insights
    st.subheader("Model Insights")
    fig = plot_feature_importance(model, ['Age', 'Gender', 'Height', 'Weight', 'Activity Level', 'Fitness Goal', 'Dietary Preference'])
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
