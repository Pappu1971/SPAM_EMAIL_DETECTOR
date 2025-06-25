import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Load the vectorizer
# -------------------------
@st.cache_resource
def load_vectorizer():
    with open("models/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

vectorizer = load_vectorizer()

# -------------------------
# Preprocessing
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# Define the model again
# -------------------------
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

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    input_dim = 3000
    model = SpamClassifierNN(input_dim)
    model.load_state_dict(torch.load("models/spam_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("📩 Spam Email Detector (Deep Learning)")

email = st.text_area("✉️ Paste your email content here:", height=200)

if st.button("Check if Spam"):
    if email.strip() == "":
        st.warning("Please enter some email text.")
    else:
        cleaned = clean_text(email)
        X = vectorizer.transform([cleaned])
        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
        with torch.no_grad():
            output = model(X_tensor)
            prediction = torch.argmax(output, axis=1).item()

        if prediction == 1:
            st.error("🚨 This email is likely SPAM.")
        else:
            st.success("✅ This email is likely NOT spam.")
