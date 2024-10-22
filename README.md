# Personalized Nutrition Plan Generator

This project is a Streamlit-based web application that generates personalized nutrition plans using machine learning. It utilizes a Random Forest Regressor model trained on a nutrition dataset to predict daily calorie targets and macronutrient distributions based on user inputs.

## Features

- User-friendly interface for inputting personal information
- Generates personalized nutrition plans including daily calorie target, protein, carbohydrates, and fat recommendations
- Displays model insights through feature importance visualization
- Utilizes machine learning for accurate predictions

## Machine Learning Process

This project uses a Random Forest Regressor model to predict nutrition plans. Here's an overview of the machine learning process:

1. Data Loading and Preprocessing:
   - Load the nutrition dataset using the Hugging Face datasets library
   - Encode categorical variables using LabelEncoder
   - Handle missing values and convert numerical columns to the appropriate data type

2. Model Training:
   - Split the data into training and testing sets
   - Perform hyperparameter tuning using RandomizedSearchCV
   - Train the Random Forest Regressor model with the best hyperparameters
   - Evaluate the model using Mean Absolute Error and R-squared score
   - Perform cross-validation to ensure model robustness

3. Prediction:
   - Encode user input using the same LabelEncoder from training
   - Use the trained model to predict the nutrition plan based on user information

4. Visualization:
   - Display feature importances to show which factors have the most impact on the predictions

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Adeish01/nutrition-plan-generator.git
   cd nutrition-plan-generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv fitness_env
   source fitness_env/bin/activate  # On Windows, use `fitness_env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run main.py
```

## Code Structure

- `main.py`: Contains the main application code, including data preprocessing, model training, and Streamlit interface
- `requirements.txt`: Lists all the required Python packages

For a detailed explanation of the code, please refer to the comments in the `main.py` file.
