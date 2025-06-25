from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    for name, model in models.items():
        print(f"\n🧪 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} Accuracy: {acc:.4f}")
        print(f"📊 {name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
