from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from pathlib import Path
import json
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Load data using TIRA API
tira = Client()
train_texts, train_labels = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
), tira.pd.truths(
    "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
)

val_texts, val_labels = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
), tira.pd.truths(
    "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
)

test_texts = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "authorship-verification-test-20240408-testing"
)

# Preprocess data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train = tfidf_vectorizer.fit_transform(train_texts['text'])
y_train = train_labels['generated']

X_val = tfidf_vectorizer.transform(val_texts['text'])
y_val = val_labels['generated']

X_test = tfidf_vectorizer.transform(test_texts['text'])

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate model
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation accuracy: {val_accuracy}")

# Test model and output predictions
test_predictions = model.predict(X_test)

# Save predictions in the required JSONL format
output_directory = get_output_directory(str(Path(__file__).parent))
output_path = Path(output_directory) / "predictions.jsonl"

with open(output_path, 'w') as f:
    for i, prediction in enumerate(test_predictions):
        prediction_entry = {'id': test_texts['id'][i], 'generated': int(prediction)}
        json.dump(prediction_entry, f)
        f.write("\n")

# Save the trained model for future use
dump(model, 'model.joblib')
