
import re
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# Example complaint dataset
data = {
    "Billing": [
        "I was charged twice for my order",
        "My invoice contains the wrong amount",
        "There is an extra charge on my bill",
        "I do not understand this payment on my statement",
        "My monthly bill is incorrect",
        "The company charged me an additional fee",
        "I paid the bill but the payment is still showing due",
        "My credit card has an unauthorized charge",
        "Please correct the amount on my invoice",
        "I received a duplicate billing notification",
        "The subscription fee is higher than expected",
        "I was billed for a service I did not use"
    ],
    "Technical Issue": [
        "The application crashes when I open it",
        "The website is not loading properly",
        "I receive an error when placing an order",
        "The mobile app keeps freezing",
        "The payment page is not working",
        "I cannot upload my documents",
        "The website shows a server error",
        "The application is very slow",
        "A technical error prevents me from continuing",
        "The screen goes blank after login",
        "The app closes unexpectedly",
        "The website buttons are not responding"
    ],
    "Account Access": [
        "I cannot log in to my account",
        "I forgot my password and cannot reset it",
        "My account has been locked",
        "I am not receiving the verification code",
        "My account login keeps failing",
        "I cannot access my profile",
        "The password reset link is not working",
        "Someone changed my account password",
        "My account has been suspended unexpectedly",
        "I cannot verify my email address",
        "The login page rejects my correct password",
        "I need help recovering my account"
    ],
    "Delivery": [
        "My order has not arrived yet",
        "The package was delivered to the wrong address",
        "My shipment is delayed",
        "I received the wrong product",
        "The tracking information has not updated",
        "My parcel arrived damaged",
        "The delivery person did not contact me",
        "My order is still showing in transit",
        "Part of my order is missing",
        "The package was marked delivered but I did not receive it",
        "I want to know the expected delivery date",
        "My order has been delayed for several days"
    ],
    "Refund": [
        "I want a refund for my cancelled order",
        "My refund has not arrived",
        "Please return the money to my account",
        "I returned the product but received no refund",
        "My refund request was rejected",
        "The refund amount is incorrect",
        "I am waiting for my money back",
        "The company has not processed my refund",
        "Please cancel my purchase and refund me",
        "I have not received the promised refund",
        "My returned item is eligible for a refund",
        "The refund is taking too long"
    ]
}

texts = []
labels = []

for category, complaints in data.items():
    for complaint in complaints:
        texts.append(complaint)
        labels.append(category)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# NLP preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

x_train_clean = [clean_text(text) for text in x_train]
x_test_clean = [clean_text(text) for text in x_test]

# 1. NLP model: TF-IDF + Logistic Regression
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

baseline.fit(x_train_clean, y_train)
baseline_predictions = baseline.predict(x_test_clean)

print("\n1. TF-IDF + Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
print(classification_report(
    y_test,
    baseline_predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

with open("complaint_model.pkl", "wb") as file:
    pickle.dump(baseline, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

# 2. Deep Learning model: ANN
ann_vectorizer = TfidfVectorizer(max_features=3000)
x_train_ann = ann_vectorizer.fit_transform(x_train_clean).toarray()
x_test_ann = ann_vectorizer.transform(x_test_clean).toarray()

ann = Sequential([
    Input(shape=(x_train_ann.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(encoder.classes_), activation="softmax")
])

ann.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining ANN...")
ann.fit(
    x_train_ann,
    y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=1
)

ann_probabilities = ann.predict(x_test_ann, verbose=0)
ann_predictions = np.argmax(ann_probabilities, axis=1)

print("\n2. Artificial Neural Network (ANN)")
print("Accuracy:", round(accuracy_score(y_test, ann_predictions), 4))
print(classification_report(
    y_test,
    ann_predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

ann.save("ann_model.keras")

with open("ann_vectorizer.pkl", "wb") as file:
    pickle.dump(ann_vectorizer, file)

# 3. Deep Learning model: LSTM
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_clean)

sequence_length = 30

x_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(x_train_clean),
    maxlen=sequence_length,
    padding="post",
    truncating="post"
)

x_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(x_test_clean),
    maxlen=sequence_length,
    padding="post",
    truncating="post"
)

lstm = Sequential([
    Input(shape=(sequence_length,)),
    Embedding(input_dim=5000, output_dim=64, mask_zero=True),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

lstm.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining LSTM...")
lstm.fit(
    x_train_seq,
    y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=1
)

lstm_probabilities = lstm.predict(x_test_seq, verbose=0)
lstm_predictions = np.argmax(lstm_probabilities, axis=1)

print("\n3. Long Short-Term Memory (LSTM)")
print("Accuracy:", round(accuracy_score(y_test, lstm_predictions), 4))
print(classification_report(
    y_test,
    lstm_predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

lstm.save("lstm_model.keras")

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open("sequence_length.pkl", "wb") as file:
    pickle.dump(sequence_length, file)

print("\nAll three models trained and saved successfully.")
print("Categories:", list(encoder.classes_))
