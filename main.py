import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Mehr Trainingsdaten
data = {
    "text": [
        "Your account has been suspended, click here to verify",
        "Update your banking information immediately",
        "Win a free iPhone now",
        "Reset your password immediately using this link",
        "Confirm your login details now",
        "Your PayPal account is locked",
        "Click here to secure your bank account",
        "Verify your identity immediately",
        "Urgent action required to avoid suspension",
        "Your mailbox has exceeded storage, click here",

        "Meeting at 10am tomorrow",
        "Can we reschedule our appointment",
        "Please review the attached document",
        "Let's have lunch next week",
        "Here is the report from yesterday",
        "The class starts at 9am",
        "Please find the invoice attached",
        "Thank you for your message",
        "I will call you later",
        "See you in the office tomorrow"
    ],
    "label": [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

user_email = input("Gib eine E-Mail ein: ")
user_vector = vectorizer.transform([user_email])

prediction = model.predict(user_vector)
probability = model.predict_proba(user_vector)[0]

if prediction[0] == 1:
    print("⚠️ Ergebnis: Phishing")
else:
    print("✅ Ergebnis: Legitimate E-Mail")

print(f"Wahrscheinlichkeit legitim: {probability[0]:.2f}")
print(f"Wahrscheinlichkeit phishing: {probability[1]:.2f}")