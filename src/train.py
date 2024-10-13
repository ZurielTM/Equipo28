import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Modelos para probar
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC()
    }

    # Hiperparámetros para cada modelo
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10, 100]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }

    best_model = None
    best_score = 0
    best_model_name = None

    # Grid Search para encontrar el mejor modelo y sus hiperparámetros
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train.values.ravel())

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    print(f"\nSelected Best Model: {best_model_name} with accuracy: {best_score}")
    
    return best_model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)

    joblib.dump(model, model_path)