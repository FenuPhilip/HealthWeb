import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import sys


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
DATA_PATH = os.path.join(MODEL_DIR, 'DiseaseAndSymptoms.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'disease_predictor_model.joblib')
SYMPTOMS_PATH = os.path.join(MODEL_DIR, 'all_symptoms.txt')

def train_model():
    """
   one-sample-per-class"
    """
    print("Starting model training process...")


    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        sys.exit(1)


    df_melted = df.melt(id_vars=['Disease'], var_name='Symptom_Num', value_name='Symptom')
    df_melted = df_melted.dropna()
    df_melted = df_melted.drop(columns=['Symptom_Num'])
    df_melted['Symptom'] = df_melted['Symptom'].str.strip()


    df_grouped = df_melted.groupby('Disease')['Symptom'].apply(lambda s: ' '.join(s)).reset_index()


    X = df_grouped['Symptom']
    y = df_grouped['Disease']
    
    print(f"Total data: {len(X)} records (diseases).")





    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])


    print(f"Training model on 100% of data for production...")
    pipeline.fit(X, y)


    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Production model saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)


    try:
        all_symptoms = pipeline.named_steps['vectorizer'].get_feature_names_out()
        with open(SYMPTOMS_PATH, 'w') as f:
            for symptom in all_symptoms:
                f.write(f"{symptom}\n")
        print(f"Symptoms list saved successfully to {SYMPTOMS_PATH}")
    except Exception as e:
        print(f"Error saving symptoms list: {e}")
        sys.exit(1)

    print("\nTraining process complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(script_dir)
    train_model()

