# Livestock AI Monitor

A hackathon-style web application prototype that combines a livestock dashboard, simulated animal health indicators, anomaly scoring, optional YOLO-based photo detection, and a Twilio voice-call integration.

## Current prototype capabilities

- Flask backend with a browser-based dashboard
- Dashboard for 12 demo animal records
- Demo risk scoring based on temperature deviation, movement, and feeding activity
- Isolation Forest anomaly detection over generated demo data
- Photo upload endpoint for livestock detection using an Ultralytics YOLO model
- Optional Twilio voice-call workflow, configured through environment variables
- Live camera preview in the browser; the interface explicitly treats this as a preview rather than a live detection feed

## Important project limitations

**This is a prototype, not a veterinary diagnostic system.** The animal measurements are simulated demo values, and the risk scores are illustrative. They must not be used to diagnose illness or make animal-care decisions.

The default general-purpose vision model only recognises the classes it was trained on. The application code maps supported class names, but that does not mean the default model can identify every livestock species. A suitable livestock-trained model may be needed for broader coverage.

## Requirements

- Python 3
- Flask
- NumPy
- scikit-learn
- python-dotenv
- Twilio Python helper library
- Ultralytics
- OpenCV and the other packages imported by the application

Install the dependencies used by the backend:

```bash
python -m pip install Flask numpy scikit-learn python-dotenv twilio ultralytics opencv-python
```

A compatible YOLO model may need to be downloaded on first use. Model availability and package compatibility depend on your Python environment.

## Configuration and security

The repository currently contains a tracked `.env` file. Do not publish real Twilio credentials or phone numbers. Rotate any real credentials that have been committed, move secrets into a local environment file, and exclude that file from version control.

The application reads configuration such as the Twilio account SID, auth token, caller number, farmer number, public base URL, and optional YOLO model name from environment variables.

## Run locally

From the repository directory:

```bash
python app.py
```

Open the local Flask address printed by the application in your browser. Twilio voice calls and externally reachable callbacks require valid account configuration and a publicly reachable application URL.

## Repository layout

```text
.
├── app.py
├── index.html
├── style.css
└── README.md
```

The HTML currently links to `/static/style.css`; verify that the stylesheet is served from the matching Flask static directory when running the app.

## Safety note

Use this project for software demonstration and learning only. Any real animal-health monitoring deployment would require validated sensors, representative data, veterinary review, and testing under real farm conditions.
