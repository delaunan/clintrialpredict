from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.data.preprocessing import get_preprocessor

def train_logistic_regression(X_train, y_train):
    """
    Trains the Logistic Regression model using pre-split data.
    """
    print(">>> Initializing Logistic Regression Pipeline...")

    # 1. Define the Pipeline
    model = Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', LogisticRegression(
            class_weight='balanced',  # Handle 1:5 imbalance
            max_iter=1000,            # Ensure convergence
            random_state=42
        ))
    ])

    # 2. Fit the model (The heavy lifting)
    print(">>> Fitting model...")
    model.fit(X_train, y_train)

    print(">>> Logistic Regression Trained.")
    return model
