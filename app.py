import os, ssl, urllib.request, threading
from flask import Flask, render_template, jsonify, request
import model

app = Flask(__name__)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(SCRIPT_DIR, "results.csv")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def refresh_data():
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(RESULTS_URL, context=ctx) as r:
            with open(CSV_PATH, "wb") as f:
                f.write(r.read())
    except Exception:
        pass
    model.load_and_train(CSV_PATH)

# Load on startup
refresh_data()

@app.route("/")
def index():
    days = int(request.args.get("days", 7))
    upcoming = model.get_upcoming_games(days_ahead=days)
    # extend window if nothing in range
    if not upcoming:
        upcoming = model.get_upcoming_games(days_ahead=30)
    updated = model.last_updated.strftime("%b %d %Y %H:%M") if model.last_updated else "—"
    return render_template("index.html", upcoming=upcoming, last_updated=updated, days=days)

@app.route("/bracket")
def bracket_page():
    b = model.simulate_bracket()
    updated = model.last_updated.strftime("%b %d %Y %H:%M") if model.last_updated else "—"
    return render_template("index.html", bracket=b, last_updated=updated,
                           upcoming=[], days=7, show_bracket=True)

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    threading.Thread(target=refresh_data, daemon=True).start()
    return jsonify({"status": "refreshing"})

@app.route("/api/upcoming")
def api_upcoming():
    days = int(request.args.get("days", 7))
    return jsonify(model.get_upcoming_games(days_ahead=days))

@app.route("/api/bracket")
def api_bracket():
    return jsonify(model.simulate_bracket())

@app.route("/api/predict")
def api_predict():
    home = request.args.get("home", "")
    away = request.args.get("away", "")
    if not home or not away:
        return jsonify({"error": "provide home= and away= params"}), 400
    return jsonify(model.predict_match(home, away))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  ⚽  WC 2026 Predictor running at  http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
