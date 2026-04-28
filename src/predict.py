import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

text = input("Enter text: ")

vec = vectorizer.transform([text])
prediction = model.predict(vec)

print("Sentiment:", prediction[0])