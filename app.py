from __future__ import annotations

from flask import Flask, render_template, request

from music_affinity import (
    UserPreference,
    explain_recommendation,
    load_or_train_model,
    predict_genres,
)

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        model = load_or_train_model()
    return model


def parse_preference(form_data) -> UserPreference:
    return UserPreference(
        bpm=int(form_data["bpm"]),
        energy=float(form_data["energy"]),
        danceability=float(form_data["danceability"]),
        acousticness=float(form_data["acousticness"]),
        instrumentalness=float(form_data["instrumentalness"]),
        mood=form_data["mood"].strip().lower(),
        vocals=form_data["vocals"].strip().lower(),
        instrumentation=form_data["instrumentation"].strip().lower(),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    explanation = None
    error = None
    values = {
        "bpm": "120",
        "energy": "0.8",
        "danceability": "0.7",
        "acousticness": "0.1",
        "instrumentalness": "0.0",
        "mood": "energetic",
        "vocals": "pop vocals",
        "instrumentation": "electronic",
        "use_gemini": False,
    }

    if request.method == "POST":
        values.update(request.form.to_dict())
        values["use_gemini"] = request.form.get("use_gemini") == "on"

        try:
            pref = parse_preference(request.form)
            predictions = predict_genres(get_model(), pref)
            try:
                explanation = explain_recommendation(pref, predictions, values["use_gemini"])
            except RuntimeError as exc:
                error = f"Gemini explanation unavailable, showing local explanation instead. ({exc})"
                explanation = explain_recommendation(pref, predictions)
        except Exception as exc:
            error = str(exc)
            if predictions:
                explanation = explain_recommendation(pref, predictions)

    return render_template(
        "index.html",
        predictions=predictions,
        explanation=explanation,
        error=error,
        values=values,
    )


@app.route("/health")
def health():
    return "Music Affinity web app is running."


if __name__ == "__main__":
    print("Starting Music Affinity web app...")
    print("Open this URL in your browser: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
