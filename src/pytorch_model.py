import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class SpamClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

def run_pytorch_model(X_train, X_test, y_train, y_test, epochs=10, lr=0.001):
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = SpamClassifierNN(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ⚖️ Handle class imbalance (spam is minority class)
    class_weights = torch.tensor([1.0, 6.5])  # ham=0, spam=1
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    losses = []

    print("\n🔥 Training PyTorch Neural Network...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "models/spam_model.pth")

    # 📈 Plot loss
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid()
    plt.show()

    # 🎯 Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        y_pred = torch.argmax(preds, axis=1).numpy()
        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ PyTorch Model Accuracy: {acc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

        # 🔍 Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (PyTorch)")
        plt.show()
