import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

def train_ranking_model():
    df = pd.read_csv('data/interactions.csv')
    
    features = [
        'category', 'price', 'rating', 'popularity_score',
        'user_session_duration', 'user_price_sensitivity'
    ]
    target = 'is_clicked'
    
    X = df[features]
    y = df[target]
    
    num_cols = ['price', 'rating', 'popularity_score', 'user_session_duration', 'user_price_sensitivity']
    cat_cols = ['category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    ranker = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        class_weight='balanced'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ranker', ranker)
    ])
    
    print("Training LightGBM Click-Through Rate (CTR) Re-Ranker...")
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print(f"\n--- Model Evaluation ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/lgbm_ranker.joblib')
    print("Model serialized to models/lgbm_ranker.joblib")

if __name__ == '__main__':
    train_ranking_model()