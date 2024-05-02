from pathlib import Path
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Training
if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    # Train the model
    model = Pipeline(
        [("vectorizer", TfidfVectorizer()), ("classifier", SVC(probability=True))]
    )
    model.fit(df["text"], df["generated"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")

# Prediction
if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"])
    df["generated"] = predictions
    df = df[["id", "generated"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
