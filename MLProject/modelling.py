import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Set MLflow tracking URI
# In CI environment, this might default to local directory if not set
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
# If http, checks connection or falls back? MLflow might error if server not found.
# For CI/CD, we might want to just run locally if server is not available.
# We will trust the environment variable or default to local if connection fails is hard to implement simply.
# But for now, let's keep it but ideally we should point to a local path in CI or mock it.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Titanic Survivability")

def train():
    # Load Titanic dataset from local file
    # Assuming script is run from root or MLProject dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'titanic_preprocessing.csv')
    
    print(f"Loading data from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to verify file existence
        if not os.path.exists(csv_path):
            print(f"File not found at {csv_path}")
        return

    # Preprocessing
    # Ensure necessary columns exist
    required_cols = ['Embarked', 'Age', 'Fare', 'Sex', 'Survived', 'Pclass', 'SibSp', 'Parch']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns. Available: {df.columns}")
        return

    df = df.dropna(subset=['Embarked', 'Age', 'Fare'])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Features and Target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use local file store for CI if server not available (hacky check but useful for robust script)
    # However, strictly if we are using "github actions to build model", likely we just want to ensure it runs without error.
    
    try:
        with mlflow.start_run():
            n_estimators = 100
            max_depth = 5
            
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            print(f"Model trained. Accuracy: {accuracy}")

            # Log model
            mlflow.sklearn.log_model(model, "model", registered_model_name="TitanicModel")
            print("Model logged and registered as 'TitanicModel'.")
    except Exception as e:
        print(f"MLflow run failed, possibly due to connection: {e}")
        # If MLflow fails, at least we trained the model.
        # But failing the CI might be desired if MLflow is critical.
        # For this task, "build model" is key.
        raise e

if __name__ == "__main__":
    train()
