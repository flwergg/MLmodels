# Modelos Supervisados y Despliegue ML

## 📋 Descripción
Pipeline completo de machine learning supervisado desde entrenamiento hasta despliegue en producción.

## 🎯 Modelos Incluidos

### Clasificación
- Random Forest Classifier
- SVM
- Logistic Regression
- XGBoost

### Regresión
- Linear Regression
- Random Forest Regressor
- Ridge/Lasso Regression

## 🛠️ Stack Tecnológico
```
scikit-learn
pandas
numpy
xgboost
flask/fastapi
docker
mlflow
streamlit
```

## 📁 Estructura del Proyecto
```
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── models/
│   ├── data/
│   └── deployment/
├── notebooks/
├── config/
├── requirements.txt
└── README.md
```

## 🚀 Instalación
```bash
git clone <repo-url>
cd ml-project
pip install -r requirements.txt
```

## 📈 Uso Rápido

### Entrenar Modelos
```python
python src/models/train_model.py --data data/processed/train.csv
```

### Evaluar Modelos
```python
python src/models/evaluate_model.py --model models/best_model.pkl
```

### Desplegar API
```python
python src/deployment/api.py
```

## 🌐 Despliegue

### Local
```bash
# API Flask
python src/deployment/api.py

# Streamlit
streamlit run src/deployment/app.py
```

### Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

### Predicción
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.4, 5.6]}'
```

## 📊 Métricas
- **Clasificación**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regresión**: MAE, MSE, RMSE, R²

## 🔧 Configuración
Editar `config/model_config.yaml` para ajustar hiperparámetros:

```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
  xgboost:
    learning_rate: 0.1
    n_estimators: 100
```

## 🧪 Testing
```bash
pytest tests/
```

## 📝 Ejemplo de Uso
```python
from src.models import ModelTrainer

# Entrenar
trainer = ModelTrainer()
model = trainer.train('random_forest', X_train, y_train)

# Predecir
predictions = model.predict(X_test)
```
