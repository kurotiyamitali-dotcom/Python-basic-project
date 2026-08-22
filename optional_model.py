from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import joblib

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

param_grid = {
    "svm__C": [1, 10, 20],
    "svm__kernel": ["linear", "rbf"]
}

grid = GridSearchCV(svm_pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

models = {
    "SVM": grid.best_estimator_,
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    results.append([name, accuracy])

comparison = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print(comparison)

best_model_name = comparison.loc[
    comparison["Accuracy"].idxmax(), "Model"
]

best_model = models[best_model_name]

print("Best Model:", best_model_name)

joblib.dump(best_model, "optional_best_model.pkl")

print("Best Model saved as optional_best_model.pkl")