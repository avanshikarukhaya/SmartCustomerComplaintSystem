
import os
import re
import csv
import pickle
import sqlite3
import secrets
from io import StringIO
from datetime import datetime
from functools import wraps

import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, make_response
)
from werkzeug.security import check_password_hash

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    tf = None
    pad_sequences = None

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-local-development-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "complaints.db")

# Set these environment variables before using this application beyond local testing.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@123")

CATEGORIES = [
    "Billing", "Technical Issue", "Account Access",
    "Delivery", "Refund"
]

STATUSES = ["Pending", "In Progress", "Resolved", "Rejected"]
PRIORITIES = ["High", "Medium", "Low"]

DEPARTMENTS = {
    "Billing": "Billing Department",
    "Technical Issue": "Technical Support",
    "Account Access": "Account Support",
    "Delivery": "Delivery Department",
    "Refund": "Refund Department"
}

# Load trained model artifacts.
def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as file:
        return pickle.load(file)


baseline_model = load_pickle("complaint_model.pkl")
label_encoder = load_pickle("label_encoder.pkl")
ann_vectorizer = load_pickle("ann_vectorizer.pkl")
tokenizer = load_pickle("tokenizer.pkl")
sequence_length = load_pickle("sequence_length.pkl")

ann_model = None
lstm_model = None

if tf is not None:
    ann_path = os.path.join(BASE_DIR, "ann_model.keras")
    lstm_path = os.path.join(BASE_DIR, "lstm_model.keras")

    if os.path.exists(ann_path):
        ann_model = tf.keras.models.load_model(ann_path)

    if os.path.exists(lstm_path):
        lstm_model = tf.keras.models.load_model(lstm_path)


def get_db():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    connection = get_db()
    cursor = connection.cursor()

    # Preserve the legacy columns used by your earlier version.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL DEFAULT '',
            email TEXT DEFAULT '',
            complaint TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT 'Technical Issue',
            priority TEXT NOT NULL DEFAULT 'Low',
            department TEXT NOT NULL DEFAULT 'Technical Support',
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT DEFAULT '',
            customer_name TEXT DEFAULT '',
            nlp_category TEXT DEFAULT '',
            ann_category TEXT DEFAULT '',
            lstm_category TEXT DEFAULT '',
            ticket_id TEXT,
            sentiment TEXT DEFAULT 'Neutral',
            sentiment_score REAL DEFAULT 0,
            confidence REAL DEFAULT 0,
            updated_at TEXT DEFAULT ''
        )
    """)

    # Migrate databases created by previous versions.
    existing_columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(complaints)").fetchall()
    }

    new_columns = {
        "name": "TEXT NOT NULL DEFAULT ''",
        "email": "TEXT DEFAULT ''",
        "complaint": "TEXT NOT NULL DEFAULT ''",
        "category": "TEXT NOT NULL DEFAULT 'Technical Issue'",
        "priority": "TEXT NOT NULL DEFAULT 'Low'",
        "department": "TEXT NOT NULL DEFAULT 'Technical Support'",
        "status": "TEXT NOT NULL DEFAULT 'Pending'",
        "created_at": "TEXT DEFAULT ''",
        "customer_name": "TEXT DEFAULT ''",
        "nlp_category": "TEXT DEFAULT ''",
        "ann_category": "TEXT DEFAULT ''",
        "lstm_category": "TEXT DEFAULT ''",
        "ticket_id": "TEXT",
        "sentiment": "TEXT DEFAULT 'Neutral'",
        "sentiment_score": "REAL DEFAULT 0",
        "confidence": "REAL DEFAULT 0",
        "updated_at": "TEXT DEFAULT ''"
    }

    for column, definition in new_columns.items():
        if column not in existing_columns:
            cursor.execute(
                f"ALTER TABLE complaints ADD COLUMN {column} {definition}"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaint_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL,
            old_status TEXT,
            new_status TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            FOREIGN KEY (complaint_id) REFERENCES complaints(id)
        )
    """)

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_complaints_ticket
        ON complaints(ticket_id)
        WHERE ticket_id IS NOT NULL
    """)

    # Give older records ticket IDs and synchronize legacy fields.
    rows = cursor.execute(
        "SELECT id, name, customer_name, category, nlp_category, "
        "created_at, ticket_id, updated_at FROM complaints"
    ).fetchall()

    for row in rows:
        ticket = row["ticket_id"] or f"CC-{row['id']:06d}"
        customer = row["customer_name"] or row["name"] or "Customer"
        category = row["nlp_category"] or row["category"] or "Technical Issue"
        created = row["created_at"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            UPDATE complaints
            SET ticket_id = ?,
                name = ?,
                customer_name = ?,
                category = ?,
                nlp_category = ?,
                created_at = ?,
                updated_at = COALESCE(NULLIF(updated_at, ''), created_at)
            WHERE id = ?
        """, (ticket, customer, customer, category, category, created, row["id"]))

    connection.commit()
    connection.close()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment(text):
    positive_words = {
        "good", "great", "excellent", "helpful", "satisfied",
        "thank", "thanks", "resolved", "happy", "love", "quick"
    }
    negative_words = {
        "bad", "poor", "terrible", "angry", "upset", "failed",
        "failure", "wrong", "delay", "delayed", "broken", "fraud",
        "stolen", "unauthorized", "worst", "disappointed", "problem"
    }

    words = set(clean_text(text).split())
    score = len(words & positive_words) - len(words & negative_words)

    if score < 0:
        return "Negative", float(score)
    if score > 0:
        return "Positive", float(score)
    return "Neutral", 0.0


def get_priority(text, sentiment):
    text = clean_text(text)

    high_words = [
        "urgent", "emergency", "fraud", "stolen", "unauthorized",
        "hacked", "security", "immediately", "account compromised",
        "money missing", "money stolen"
    ]
    medium_words = [
        "delay", "delayed", "not working", "failed", "failure",
        "wrong", "missing", "unable", "problem", "error"
    ]

    if any(word in text for word in high_words):
        return "High"
    if sentiment == "Negative" or any(word in text for word in medium_words):
        return "Medium"
    return "Low"


def safe_confidence(probabilities):
    values = np.asarray(probabilities).reshape(-1)
    if values.size == 0:
        return 0.0
    return round(float(np.max(values)) * 100, 2)


def predict_categories(text):
    cleaned = clean_text(text)

    # Baseline NLP model.
    baseline_encoded = int(baseline_model.predict([cleaned])[0])
    nlp_category = str(label_encoder.inverse_transform([baseline_encoded])[0])

    ann_category = nlp_category
    lstm_category = nlp_category
    confidence = 0.0

    # Obtain probabilities when the saved models support them.
    if hasattr(baseline_model, "predict_proba"):
        try:
            confidence = safe_confidence(
                baseline_model.predict_proba([cleaned])[0]
            )
        except Exception:
            confidence = 0.0

    # ANN model.
    if ann_model is not None:
        try:
            vector = ann_vectorizer.transform([cleaned]).toarray()
            prediction = np.asarray(ann_model.predict(vector, verbose=0))
            predicted_index = int(np.argmax(prediction[0]))
            ann_category = str(
                label_encoder.inverse_transform([predicted_index])[0]
            )
        except Exception as error:
            app.logger.warning("ANN prediction failed: %s", error)

    # LSTM model.
    if lstm_model is not None and pad_sequences is not None:
        try:
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(
                sequence,
                maxlen=int(sequence_length),
                padding="pre",
                truncating="pre"
            )
            prediction = np.asarray(lstm_model.predict(padded, verbose=0))
            predicted_index = int(np.argmax(prediction[0]))
            lstm_category = str(
                label_encoder.inverse_transform([predicted_index])[0]
            )
        except Exception as error:
            app.logger.warning("LSTM prediction failed: %s", error)

    # The NLP baseline is the primary category for routing.
    return {
        "nlp_category": nlp_category,
        "ann_category": ann_category,
        "lstm_category": lstm_category,
        "confidence": confidence
    }


def admin_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please log in as administrator.", "warning")
            return redirect(url_for("admin_login"))
        return function(*args, **kwargs)
    return decorated_function


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        complaint = request.form.get("complaint", "").strip()

        if not name or not email or not complaint:
            flash("Please complete all fields.", "danger")
            return redirect(url_for("index"))

        if len(complaint) < 10:
            flash("Please describe the complaint in at least 10 characters.", "warning")
            return redirect(url_for("index"))

        if len(complaint) > 5000:
            flash("The complaint must be 5,000 characters or fewer.", "warning")
            return redirect(url_for("index"))

        predictions = predict_categories(complaint)
        sentiment, sentiment_score = get_sentiment(complaint)
        priority = get_priority(complaint, sentiment)
        category = predictions["nlp_category"]
        department = DEPARTMENTS.get(category, "Customer Support")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        connection = get_db()
        cursor = connection.cursor()

        try:
            cursor.execute("""
                INSERT INTO complaints (
                    name, customer_name, email, complaint,
                    category, nlp_category, ann_category, lstm_category,
                    priority, department, status, created_at, updated_at,
                    sentiment, sentiment_score, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, name, email, complaint,
                category, predictions["nlp_category"],
                predictions["ann_category"], predictions["lstm_category"],
                priority, department, "Pending", now, now,
                sentiment, sentiment_score, predictions["confidence"]
            ))

            complaint_id = cursor.lastrowid
            ticket_id = f"CC-{complaint_id:06d}"

            cursor.execute(
                "UPDATE complaints SET ticket_id = ? WHERE id = ?",
                (ticket_id, complaint_id)
            )
            cursor.execute("""
                INSERT INTO complaint_history
                (complaint_id, old_status, new_status, changed_at)
                VALUES (?, ?, ?, ?)
            """, (complaint_id, None, "Pending", now))

            connection.commit()

        except sqlite3.Error:
            connection.rollback()
            app.logger.exception("Could not save complaint")
            flash("The complaint could not be saved. Please check the server terminal.", "danger")
            return redirect(url_for("index"))
        finally:
            connection.close()

        return render_template(
            "index.html",
            submitted=True,
            ticket_id=ticket_id,
            predictions=predictions,
            sentiment=sentiment,
            priority=priority,
            department=department
        )

    return render_template("index.html", submitted=False)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        # Supports a Werkzeug password hash if ADMIN_PASSWORD_HASH is configured.
        password_hash = os.environ.get("ADMIN_PASSWORD_HASH", "")
        password_ok = (
            check_password_hash(password_hash, password)
            if password_hash
            else secrets.compare_digest(password, ADMIN_PASSWORD)
        )

        if secrets.compare_digest(username, ADMIN_USERNAME) and password_ok:
            session.clear()
            session["admin_logged_in"] = True
            session["admin_username"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid administrator credentials.", "danger")

    return render_template("login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard")
@admin_required
def dashboard():
    search = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip()
    priority = request.args.get("priority", "").strip()
    status = request.args.get("status", "").strip()

    query = "SELECT * FROM complaints WHERE 1=1"
    parameters = []

    if search:
        query += """ AND (
            ticket_id LIKE ? OR name LIKE ? OR email LIKE ? OR complaint LIKE ?
        )"""
        term = f"%{search}%"
        parameters.extend([term, term, term, term])

    if category in CATEGORIES:
        query += " AND nlp_category = ?"
        parameters.append(category)

    if priority in PRIORITIES:
        query += " AND priority = ?"
        parameters.append(priority)

    if status in STATUSES:
        query += " AND status = ?"
        parameters.append(status)

    query += " ORDER BY id DESC"

    connection = get_db()
    complaints = connection.execute(query, parameters).fetchall()

    total = connection.execute(
        "SELECT COUNT(*) FROM complaints"
    ).fetchone()[0]

    high_priority = connection.execute(
        "SELECT COUNT(*) FROM complaints WHERE priority = 'High'"
    ).fetchone()[0]

    pending = connection.execute(
        "SELECT COUNT(*) FROM complaints WHERE status = 'Pending'"
    ).fetchone()[0]

    resolved = connection.execute(
        "SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'"
    ).fetchone()[0]

    category_rows = connection.execute("""
        SELECT nlp_category AS category, COUNT(*) AS total
        FROM complaints
        GROUP BY nlp_category
        ORDER BY total DESC
    """).fetchall()

    priority_rows = connection.execute("""
        SELECT priority, COUNT(*) AS total
        FROM complaints
        GROUP BY priority
    """).fetchall()

    status_rows = connection.execute("""
        SELECT status, COUNT(*) AS total
        FROM complaints
        GROUP BY status
    """).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        complaints=complaints,
        total=total,
        high_priority=high_priority,
        pending=pending,
        resolved=resolved,
        category_rows=category_rows,
        priority_rows=priority_rows,
        status_rows=status_rows,
        categories=CATEGORIES,
        priorities=PRIORITIES,
        statuses=STATUSES,
        filters={
            "search": search,
            "category": category,
            "priority": priority,
            "status": status
        }
    )


@app.route("/admin/update-status/<int:complaint_id>", methods=["POST"])
@admin_required
def update_status(complaint_id):
    new_status = request.form.get("status", "")
    if new_status not in STATUSES:
        flash("Invalid status selected.", "danger")
        return redirect(url_for("dashboard"))

    connection = get_db()
    current = connection.execute(
        "SELECT status FROM complaints WHERE id = ?",
        (complaint_id,)
    ).fetchone()

    if current is None:
        connection.close()
        flash("Complaint not found.", "danger")
        return redirect(url_for("dashboard"))

    old_status = current["status"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection.execute("""
        UPDATE complaints
        SET status = ?, updated_at = ?
        WHERE id = ?
    """, (new_status, now, complaint_id))

    if old_status != new_status:
        connection.execute("""
            INSERT INTO complaint_history
            (complaint_id, old_status, new_status, changed_at)
            VALUES (?, ?, ?, ?)
        """, (complaint_id, old_status, new_status, now))

    connection.commit()
    connection.close()

    flash("Complaint status updated.", "success")
    return redirect(url_for("dashboard"))


@app.route("/admin/history/<int:complaint_id>")
@admin_required
def complaint_history(complaint_id):
    connection = get_db()
    complaint = connection.execute(
        "SELECT * FROM complaints WHERE id = ?",
        (complaint_id,)
    ).fetchone()
    history = connection.execute("""
        SELECT * FROM complaint_history
        WHERE complaint_id = ?
        ORDER BY id DESC
    """, (complaint_id,)).fetchall()
    connection.close()

    if complaint is None:
        flash("Complaint not found.", "danger")
        return redirect(url_for("dashboard"))

    return render_template(
        "dashboard.html",
        history_only=True,
        history_complaint=complaint,
        history=history,
        complaints=[],
        total=0,
        high_priority=0,
        pending=0,
        resolved=0,
        category_rows=[],
        priority_rows=[],
        status_rows=[],
        categories=CATEGORIES,
        priorities=PRIORITIES,
        statuses=STATUSES,
        filters={"search": "", "category": "", "priority": "", "status": ""}
    )


@app.route("/admin/export")
@admin_required
def export_csv():
    connection = get_db()
    rows = connection.execute("""
        SELECT ticket_id, name, email, complaint, nlp_category,
               ann_category, lstm_category, confidence, sentiment,
               priority, department, status, created_at, updated_at
        FROM complaints
        ORDER BY id DESC
    """).fetchall()
    connection.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Ticket ID", "Customer Name", "Email", "Complaint",
        "NLP Category", "ANN Category", "LSTM Category",
        "NLP Confidence (%)", "Sentiment", "Priority",
        "Department", "Status", "Created At", "Updated At"
    ])

    for row in rows:
        writer.writerow(list(row))

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=complaints_report.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


@app.errorhandler(404)
def page_not_found(error):
    return "Page not found.", 404


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
