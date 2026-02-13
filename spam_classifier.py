from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
texts = [
    "Win money now",
    "Hello how are you",
    "Free prize claim now",
    "Let's meet tomorrow",
    "Earn cash fast",
    "Are you coming today"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test input
test_message = ["Free money offer"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")
