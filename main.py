from src.preprocessing import load_and_preprocess
from src.ml_models import train_ml_models
from src.pytorch_model import run_pytorch_model


if __name__ == "__main__":
   X_train, X_test, y_train, y_test = load_and_preprocess("data/sms.tsv")
   print("🔍 Training scikit-learn models...")
   train_ml_models(X_train, X_test, y_train, y_test)
print("🔥 Training PyTorch neural net...")
run_pytorch_model(X_train, X_test, y_train, y_test)
