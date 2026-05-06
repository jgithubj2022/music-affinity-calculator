"""
Music Affinity Calculator - CPSC 481 starter project
Approach: Machine Learning classifier + optional Gemini explanation.

Run:
  pip install -r requirements.txt
  python music_affinity.py

Public dataset setup:
  python music_affinity.py --download-public-data
  python music_affinity.py --train

Gemini setup:
  setx GEMINI_API_KEY "your-gemini-key-here"
  python music_affinity.py --gemini-key-check
  python music_affinity.py --test-gemini

This version does NOT require Spotify audio features, because Spotify deprecated/restricted
those endpoints for many new apps. It uses a local starter dataset or a public CSV dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List
from urllib import error, request

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "music_training_data.csv"
MODEL_FILE = "music_affinity_model.joblib"
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PUBLIC_DATASET_URL = (
    "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/"
    "resolve/main/dataset.csv"
)

STARTER_ROWS = [
    ["Blinding Lights", "The Weeknd", 171, 0.80, 0.51, 0.00, 0.00, "energetic", "pop vocals", "synth", "synthpop"],
    ["Midnight City", "M83", 105, 0.72, 0.52, 0.02, 0.35, "dreamy", "light vocals", "synth", "synthpop"],
    ["Take On Me", "a-ha", 169, 0.85, 0.57, 0.02, 0.00, "bright", "pop vocals", "synth", "synthpop"],
    ["Levitating", "Dua Lipa", 103, 0.82, 0.70, 0.01, 0.00, "fun", "pop vocals", "electronic", "dance pop"],
    ["One More Time", "Daft Punk", 123, 0.70, 0.61, 0.02, 0.12, "party", "sampled vocals", "electronic", "dance pop"],
    ["Titanium", "David Guetta", 126, 0.78, 0.60, 0.01, 0.00, "powerful", "pop vocals", "electronic", "edm"],
    ["Levels", "Avicii", 126, 0.82, 0.58, 0.01, 0.30, "uplifting", "sampled vocals", "electronic", "edm"],
    ["Clair de Lune", "Debussy", 70, 0.18, 0.20, 0.98, 0.90, "calm", "instrumental", "piano", "classical"],
    ["Gymnopedie No.1", "Erik Satie", 72, 0.12, 0.18, 0.99, 0.91, "peaceful", "instrumental", "piano", "classical"],
    ["Nocturne Op.9 No.2", "Chopin", 85, 0.22, 0.19, 0.95, 0.88, "romantic", "instrumental", "piano", "classical"],
    ["Bad Guy", "Billie Eilish", 135, 0.42, 0.70, 0.33, 0.13, "dark", "soft vocals", "bass", "alt pop"],
    ["Sweater Weather", "The Neighbourhood", 124, 0.55, 0.61, 0.06, 0.00, "moody", "indie vocals", "guitar", "alt pop"],
    ["Do I Wanna Know?", "Arctic Monkeys", 85, 0.53, 0.55, 0.19, 0.00, "dark", "rock vocals", "guitar", "indie rock"],
    ["Electric Feel", "MGMT", 103, 0.74, 0.65, 0.05, 0.05, "psychedelic", "indie vocals", "synth", "indie pop"],
    ["Space Song", "Beach House", 147, 0.38, 0.39, 0.19, 0.13, "dreamy", "soft vocals", "synth", "dream pop"],
    ["Myth", "Beach House", 155, 0.45, 0.37, 0.10, 0.12, "dreamy", "soft vocals", "synth", "dream pop"],
    ["SICKO MODE", "Travis Scott", 155, 0.73, 0.83, 0.01, 0.00, "hype", "rap vocals", "bass", "hip hop"],
    ["HUMBLE.", "Kendrick Lamar", 150, 0.62, 0.91, 0.00, 0.00, "confident", "rap vocals", "bass", "hip hop"],
    ["Sunflower", "Post Malone", 90, 0.52, 0.76, 0.54, 0.00, "chill", "rap vocals", "bass", "hip hop"],
    ["Redbone", "Childish Gambino", 160, 0.35, 0.74, 0.17, 0.00, "smooth", "soul vocals", "bass", "rnb"],
    ["Get You", "Daniel Caesar", 74, 0.31, 0.58, 0.42, 0.00, "romantic", "soul vocals", "guitar", "rnb"],
    ["Pink + White", "Frank Ocean", 160, 0.50, 0.55, 0.67, 0.00, "warm", "soul vocals", "piano", "rnb"],
    ["Everlong", "Foo Fighters", 158, 0.88, 0.41, 0.00, 0.00, "intense", "rock vocals", "guitar", "rock"],
    ["Smells Like Teen Spirit", "Nirvana", 117, 0.91, 0.50, 0.00, 0.00, "angsty", "rock vocals", "guitar", "rock"],
    ["Mr. Brightside", "The Killers", 148, 0.91, 0.36, 0.00, 0.00, "energetic", "rock vocals", "guitar", "indie rock"],
]

COLUMNS = [
    "title", "artist", "bpm", "energy", "danceability", "acousticness",
    "instrumentalness", "mood", "vocals", "instrumentation", "genre"
]


@dataclass
class UserPreference:
    bpm: int
    energy: float
    danceability: float
    acousticness: float
    instrumentalness: float
    mood: str
    vocals: str
    instrumentation: str


def create_starter_dataset(path: str = DATA_FILE) -> None:
    if not os.path.exists(path):
        df = pd.DataFrame(STARTER_ROWS, columns=COLUMNS)
        df.to_csv(path, index=False)
        print(f"Created starter dataset: {path}")


def estimate_mood(row: pd.Series) -> str:
    valence = float(row.get("valence", 0.5))
    energy = float(row.get("energy", 0.5))

    if valence >= 0.65 and energy >= 0.65:
        return "energetic"
    if valence >= 0.65:
        return "bright"
    if valence <= 0.35 and energy >= 0.65:
        return "intense"
    if valence <= 0.35:
        return "dark"
    if energy <= 0.35:
        return "calm"
    return "balanced"


def estimate_vocals(row: pd.Series) -> str:
    instrumentalness = float(row.get("instrumentalness", 0.0))
    speechiness = float(row.get("speechiness", 0.0))
    energy = float(row.get("energy", 0.5))

    if instrumentalness >= 0.65:
        return "instrumental"
    if speechiness >= 0.33:
        return "rap vocals"
    if energy <= 0.4:
        return "soft vocals"
    return "pop vocals"


def estimate_instrumentation(row: pd.Series) -> str:
    acousticness = float(row.get("acousticness", 0.0))
    instrumentalness = float(row.get("instrumentalness", 0.0))
    energy = float(row.get("energy", 0.5))
    danceability = float(row.get("danceability", 0.5))

    if acousticness >= 0.75 and instrumentalness >= 0.5:
        return "piano"
    if acousticness >= 0.55:
        return "acoustic"
    if danceability >= 0.7 and energy >= 0.55:
        return "electronic"
    if energy >= 0.75:
        return "guitar"
    return "bass"


def convert_public_dataset(raw_df: pd.DataFrame, max_rows_per_genre: int = 200) -> pd.DataFrame:
    required = [
        "track_name", "artists", "tempo", "energy", "danceability",
        "acousticness", "instrumentalness", "track_genre"
    ]
    missing = [column for column in required if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Public dataset is missing required columns: {missing}")

    df = raw_df.dropna(subset=required).copy()
    df = df.rename(
        columns={
            "track_name": "title",
            "artists": "artist",
            "tempo": "bpm",
            "track_genre": "genre",
        }
    )

    df["bpm"] = df["bpm"].round().astype(int)
    df["mood"] = df.apply(estimate_mood, axis=1)
    df["vocals"] = df.apply(estimate_vocals, axis=1)
    df["instrumentation"] = df.apply(estimate_instrumentation, axis=1)

    converted = df[COLUMNS]
    converted = converted.groupby("genre", group_keys=False).head(max_rows_per_genre)
    return converted.sample(frac=1.0, random_state=42).reset_index(drop=True)


def download_public_dataset(path: str = DATA_FILE) -> pd.DataFrame:
    print("Downloading public Spotify Tracks Dataset from Hugging Face...")
    raw_df = pd.read_csv(PUBLIC_DATASET_URL)
    converted = convert_public_dataset(raw_df)
    converted.to_csv(path, index=False)
    print(f"Saved converted dataset: {path} ({len(converted)} rows)")
    print("Note: mood, vocals, and instrumentation are estimated from available audio features.")
    return converted


def train_model(path: str = DATA_FILE) -> Pipeline:
    df = pd.read_csv(path)

    features = [
        "bpm", "energy", "danceability", "acousticness", "instrumentalness",
        "mood", "vocals", "instrumentation"
    ]
    target = "genre"

    X = df[features]
    y = df[target]

    numeric_features = ["bpm", "energy", "danceability", "acousticness", "instrumentalness"]
    categorical_features = ["mood", "vocals", "instrumentation"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=None
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel sanity-check report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model: {MODEL_FILE}")
    return model


def load_or_train_model() -> Pipeline:
    create_starter_dataset()
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_model()


def predict_genres(model: Pipeline, pref: UserPreference, top_n: int = 5) -> List[Dict[str, float]]:
    user_df = pd.DataFrame([pref.__dict__])
    classifier = model.named_steps["classifier"]

    probabilities = model.predict_proba(user_df)[0]
    classes = classifier.classes_

    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )[:top_n]

    return [{"genre": genre, "score": round(float(prob) * 100, 2)} for genre, prob in ranked]


def get_gemini_api_key() -> tuple[str, str]:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if gemini_key:
        return gemini_key, "GEMINI_API_KEY"
    if google_key:
        return google_key, "GOOGLE_API_KEY"
    return "", ""


def describe_gemini_key() -> str:
    api_key, source = get_gemini_api_key()
    if not api_key:
        return (
            "No Gemini API key found. Set GEMINI_API_KEY, then close and reopen "
            "PowerShell or your editor terminal."
        )

    if len(api_key) <= 8:
        masked = "*" * len(api_key)
    else:
        masked = f"{api_key[:4]}...{api_key[-4:]}"
    return f"Using {source}: {masked} ({len(api_key)} characters)"


def build_explanation_prompt(pref: UserPreference, predictions: List[Dict[str, float]]) -> str:
    top = predictions[0]
    return f"""
In exactly 5 words, explain this music match.
Genre: {top['genre']}
Traits: {pref.mood}, {pref.instrumentation}
""".strip()


def extract_gemini_text(data: Dict, minimum_words: int = 12) -> str:
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Gemini did not return a usable response.") from exc

    text = " ".join(part.get("text", "").strip() for part in parts if part.get("text"))
    if not text:
        finish_reason = candidate.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no text. Finish reason: {finish_reason}")

    if len(text.split()) < minimum_words:
        finish_reason = candidate.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned too little text. Finish reason: {finish_reason}")

    return text


def generate_gemini_text(
    prompt: str,
    model_name: str = GEMINI_MODEL,
    minimum_words: int = 3,
) -> str:
    api_key, _ = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it as an environment variable, then close "
            "and reopen PowerShell or your editor terminal."
        )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 24,
            "temperature": 0.3,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{GEMINI_API_URL.format(model=model_name)}?key={api_key}"
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {exc.code} {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc

    return extract_gemini_text(data, minimum_words)


def explain_recommendation(
    pref: UserPreference,
    predictions: List[Dict[str, float]],
    use_gemini: bool = False,
    gemini_model: str = GEMINI_MODEL,
) -> str:
    if use_gemini:
        prompt = build_explanation_prompt(pref, predictions)
        return generate_gemini_text(prompt, gemini_model)

    top = predictions[0]
    return (
        f"Your highest match is {top['genre']} with an affinity score of {top['score']}%. "
        f"This was selected because you requested music with {pref.mood} mood, "
        f"{pref.instrumentation} instrumentation, BPM around {pref.bpm}, energy {pref.energy}, "
        f"and danceability {pref.danceability}. The model compares those traits with songs in the training dataset "
        f"and ranks genres by similarity."
    )


def collect_user_input() -> UserPreference:
    print("\nMusic Affinity Calculator")
    print("Answer each question, then press Enter.")
    print("For decimal ratings, type a number from 0.0 to 1.0.")
    print("Example full input set: 120, 0.8, 0.7, 0.1, 0.0, energetic, pop vocals, electronic\n")

    return UserPreference(
        bpm=int(input("1. Preferred tempo/BPM as a whole number, example 120: ").strip()),
        energy=float(input("2. Energy as a decimal from 0.0 calm to 1.0 intense, example 0.8: ").strip()),
        danceability=float(input("3. Danceability as a decimal from 0.0 low to 1.0 high, example 0.7: ").strip()),
        acousticness=float(input("4. Acousticness as a decimal from 0.0 electronic to 1.0 acoustic, example 0.1: ").strip()),
        instrumentalness=float(input("5. Instrumentalness as a decimal from 0.0 vocals to 1.0 instrumental, example 0.0: ").strip()),
        mood=input("6. Mood/vibe as words, examples: energetic, dreamy, dark, calm, bright: ").strip().lower(),
        vocals=input("7. Vocal style as words, examples: pop vocals, rap vocals, instrumental, soft vocals: ").strip().lower(),
        instrumentation=input("8. Main instrumentation as words, examples: synth, guitar, bass, piano, electronic: ").strip().lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Affinity Calculator")
    parser.add_argument(
        "--download-public-data",
        action="store_true",
        help="Download and convert a public Spotify Tracks CSV dataset into the local training format.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model from the current CSV dataset.",
    )
    parser.add_argument(
        "--data",
        default=DATA_FILE,
        help=f"Path to the training CSV. Default: {DATA_FILE}",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini to generate the explanation from the model predictions.",
    )
    parser.add_argument(
        "--gemini-model",
        default=GEMINI_MODEL,
        help=f"Gemini model name to use for explanations. Default: {GEMINI_MODEL}",
    )
    parser.add_argument(
        "--test-gemini",
        action="store_true",
        help="Send a small test request to Gemini to confirm the environment variable works.",
    )
    parser.add_argument(
        "--gemini-key-check",
        action="store_true",
        help="Show which Gemini environment variable is loaded without printing the full key.",
    )
    args = parser.parse_args()

    if args.gemini_key_check:
        print(describe_gemini_key())
        return

    if args.test_gemini:
        print(describe_gemini_key())
        try:
            print(generate_gemini_text("Reply with: Gemini API key is working.", args.gemini_model, minimum_words=3))
        except RuntimeError as exc:
            print(f"Gemini test failed: {exc}")
            print(
                "\nIf this says 403 Forbidden, create a key from a different Google AI Studio "
                "account/project, then reset GEMINI_API_KEY and reopen PowerShell."
            )
        return

    if args.download_public_data:
        download_public_dataset(args.data)
        if not args.train:
            print("Next step: run `python music_affinity.py --train` to train on the new dataset.")
            return

    if args.train:
        train_model(args.data)
        return

    model = load_or_train_model()
    pref = collect_user_input()
    predictions = predict_genres(model, pref)

    print("\nRanked genre recommendations:")
    for index, item in enumerate(predictions, start=1):
        print(f"{index}. {item['genre']} - {item['score']}% affinity")

    print("\nExplanation:")
    try:
        print(explain_recommendation(pref, predictions, args.use_gemini, args.gemini_model))
    except RuntimeError as exc:
        print(f"API explanation failed: {exc}")
        print("\nFallback explanation:")
        print(explain_recommendation(pref, predictions))


if __name__ == "__main__":
    main()
