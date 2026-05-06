# Music Affinity Calculator - ML Version

This is a buildable version of the proposed Bayesian-network project. Instead of hand-writing conditional probability tables, it uses a supervised machine learning classifier that learns relationships between user-given music attributes and likely genres.

## Why this approach
A Bayesian network is possible, but it is hard because you would need to manually define probability tables like `P(Genre | tempo, mood, instrumentation, vocals)`. That gets complicated quickly. A machine learning classifier is easier to explain, easier to implement, and still fits the AI/music-affinity goal.

## How to run
```bash
pip install -r requirements.txt
python music_affinity.py
```

## How to run the web interface
```bash
pip install -r requirements.txt
python app.py
```

Then open:
```text
http://127.0.0.1:5050
```

## How to use a public dataset
This project can use the Hugging Face `maharshipandya/spotify-tracks-dataset`, a public CSV dataset with about 114,000 tracks, audio features, and genre labels.

Download and convert it into this project's format:
```bash
python music_affinity.py --download-public-data
```

Train a new model from the downloaded data:
```bash
python music_affinity.py --train
```

Then run the app:
```bash
python music_affinity.py
```

## Gemini API key setup
Do not paste your Gemini API key directly into the code. Store it as an environment variable instead:

```powershell
setx GEMINI_API_KEY "your-gemini-key-here"
```

Close and reopen PowerShell, then test:
```bash
python music_affinity.py --gemini-key-check
python music_affinity.py --test-gemini
```

Run the app with Gemini-generated explanations:
```bash
python music_affinity.py --use-gemini
```

If `--test-gemini` returns `403 PERMISSION_DENIED`, create a new Gemini API key using a different personal Google account or a new Google AI Studio project, then set `GEMINI_API_KEY` again and reopen PowerShell.

The public dataset includes real audio-feature columns such as `tempo`, `energy`, `danceability`, `acousticness`, `instrumentalness`, and `track_genre`. It does not directly include your custom labels for `mood`, `vocals`, or `instrumentation`, so the importer estimates those labels from the available features. That is acceptable for a class prototype, but you should mention it in your report.

## What data to collect
Add rows to `music_training_data.csv` with:

- title
- artist
- bpm
- energy from 0.0 to 1.0
- danceability from 0.0 to 1.0
- acousticness from 0.0 to 1.0
- instrumentalness from 0.0 to 1.0
- mood/vibe
- vocal style
- instrumentation
- genre

Aim for at least 150 songs total and at least 15-20 songs per genre.

## Project demo idea
User enters: energetic, fast, synth, pop vocals, high danceability.
System returns: synthpop, dance pop, EDM with affinity scores and explanation.
