# Crime Category Prediction (Deployed)

Live demo: https://crime-prediction-deployed.onrender.com/

## What this is
A simple Flask web app that predicts a broad crime category from a small set of incident features using a pre-trained scikit-learn / LightGBM model. It's intended for demonstration and prototyping — data scientists or developers can run it locally, review the preprocessing, and swap in a different model.

### Stack
- **Language(s):** Python (backend), HTML (frontend)
- **Framework / runtime:** Flask 3
- **Notable libraries:** scikit-learn, lightgbm, pandas, numpy

## How it's organized
```
README.md                 Project overview and usage (this file)
app.py                    Flask app + input processing and prediction endpoint
model.pkl                 Pre-trained model (binary pickle) used by the app
requirements.txt          Python dependencies
Templates/                HTML templates (index.html)
  index.html              UI form for sending inputs to /predict
.gitignore
```

How it fits together: app.py loads model.pkl at startup, serves the HTML form from Templates/index.html, accepts POSTs to /predict, builds a single-row DataFrame from form inputs, applies simple imputation (via sklearn ColumnTransformer), and passes the cleaned features into the model to produce a category prediction.

## Inputs (fields the web form accepts)
The form (Templates/index.html) and app.py expect these fields:
- Victim_Age (integer) — optional in the form but converted to numeric
- Reporting_District_no (integer) — required
- hour (0-23 integer) — required
- Date_Occurred_month (1-12 integer) — required
- Date_Occurred_dayofweek (0-6 integer; Monday=0) — required
- Date_Occurred_dayofmonth (1-31 integer) — required
- Premise_Code (integer) — required
- Area_Name (string) — required
- Victim_Sex (categorical) — default 'M' if missing
- Victim_Descent (categorical) — default 'H' if missing
- Status_Description (categorical) — default 'IC' if missing
- Location (string; latitude, longitude suggested) — required

app.py also fills these defaults when not provided:
- Modus_Operandi: '0416 1241 1243 1813 1821 2000'
- Weapon_Used_Code: 400.0
- Part 1-2: 2.0

The app constructs Date_Reported fields (month, dayofweek, dayofmonth) from the current server date/time.

## How to run it (local development)
1. Clone the repo:

```bash
git clone https://github.com/mudgalma/Crime_prediction_deployed.git
cd Crime_prediction_deployed
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app for development:

```bash
python app.py
```

Then open http://127.0.0.1:5000/ in your browser and use the form.

Production (example with gunicorn):

```bash
gunicorn app:app --bind 0.0.0.0:8000
```

Notes:
- model.pkl is bundled in the repository. If you replace the model, make sure it accepts the same feature ordering and preprocessing.
- Flask debug mode is enabled in app.py (app.run(debug=True)) — disable this in production.

## API (direct POST)
You can POST form-encoded data to /predict. Example using curl (adjust field values):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "Victim_Age=30" \
  -d "Reporting_District_no=200" \
  -d "hour=14" \
  -d "Date_Occurred_month=6" \
  -d "Date_Occurred_dayofweek=2" \
  -d "Date_Occurred_dayofmonth=15" \
  -d "Premise_Code=10" \
  -d "Area_Name=Central" \
  -d "Location=34.05,-118.25"
```

## Predicted categories
The model predicts one of these categories (mapped in app.py):
- crimes against person
- crimes against public order
- fraud and white-collar crimes
- other crimes
- property crimes
- violent crimes

## Troubleshooting
- ValueError rendering: if a form value can't be converted to int/float the page will show an error. Ensure required numeric fields are valid numbers.
- If you replace model.pkl, make sure the preprocessing in app.py matches the model's expected features and ordering.
- If template rendering fails, check that Flask's template_folder path in app.py points to `./Templates` (it does in this repo).

## Deployment
This project is already deployed to Render: https://crime-prediction-deployed.onrender.com/

To deploy elsewhere, bundle `model.pkl`, ensure all packages are installed, and run via a WSGI server (gunicorn, uWSGI) behind a reverse proxy.

## Security & data
- model.pkl is a pickle file. Treat it as untrusted input if you did not create it — loading arbitrary pickles can execute code. Replace with a safer serialization or ensure provenance for production.
- The app does no authentication and should not be used with sensitive or personal data in a production setting.

## Contributing
Pull requests are welcome. Open an issue with a short description before large changes (e.g., changing model schema or feature names).

## License
Add a LICENSE file if you want to specify usage terms.

## Acknowledgements
This repository demonstrates a minimal example of serving a scikit-learn/LightGBM model with Flask.
