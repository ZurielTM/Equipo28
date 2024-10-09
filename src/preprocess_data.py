import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data_path):
    data = pd.read_csv(data_path)

    # Separar las características (X) y la variable objetivo (y)
    X = data.drop('Response', axis=1)
    y = data['Response']

    # Identificar las columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

    # Crear un pipeline de preprocesamiento para las características categóricas y numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ajustar y transformar los datos de entrenamiento y luego transformar los datos de prueba
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)