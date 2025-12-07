import os
import joblib

def save_model(model, filename):
    """
    Saves a trained model to the src/models folder.
    """
    # Get the directory where THIS file (model_utils.py) is located
    current_dir = os.path.dirname(__file__)

    # Create the full path
    file_path = os.path.join(current_dir, filename)

    print(f">>> Saving model to: {file_path}")
    joblib.dump(model, file_path)
    return file_path

def load_model(filename):
    """
    Loads a model from the src/models folder.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model not found at {file_path}. Did you train it?")

    print(f">>> Loading model from: {file_path}")
    return joblib.load(file_path)
