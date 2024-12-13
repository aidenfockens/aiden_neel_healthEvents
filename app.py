from flask import Flask, jsonify, request, render_template, send_file
from dotenv import load_dotenv
import os
from sqlalchemy import Column, Integer, String, DateTime
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
from flask_cors import CORS

# Set non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
load_dotenv()
DATABASE_URI = os.getenv('DATABASE_URI')
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Allow cookies across domains
app.config['SESSION_COOKIE_SECURE'] = True

db = SQLAlchemy(app)

# SQLAlchemy Event model
class Event(db.Model):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    Location = Column(String(150), nullable=False)
    Severity = Column(String(150), nullable=False)
    EventType = Column(String(150), nullable=False)
    Details = Column(String(150), nullable=False)
    Timestamp = Column(DateTime, nullable=False)

# Constants for form options
LOCATIONS = ["London", "New York", "Paris", "Boston", "Los Angeles", "Berlin"]
SEVERITIES = ["high", "medium", "low"]
EVENTTYPES = ["hospital_admission", "general_health_report", "health_mention", "emergency_incident", "routine_checkup", "vaccination"]

# Routes
@app.route("/")
def home():
    return render_template("index.html", locations=LOCATIONS)

@app.route("/events", methods=["POST"])
def get_events():
    location = request.form.get("location")
    time_frame = request.form.get("time_frame")

    if location not in LOCATIONS:
        return jsonify({"error": "Invalid location"}), 400

    now = datetime.now()
    if time_frame == "week":
        start_time = now - timedelta(weeks=1)
    elif time_frame == "month":
        start_time = now - timedelta(days=30)
    elif time_frame == "year":
        start_time = now - timedelta(days=365)
    elif time_frame == "decade":
        start_time = now - timedelta(days=365 * 10)
    else:
        return jsonify({"error": "Invalid time frame"}), 400

    # Query database
    print("starting to query database")
    events = Event.query.filter(Event.Location == location, Event.Timestamp >= start_time).all()
    print("success!")
    if not events:
        return jsonify({"error": "No events found for the given criteria"}), 404

    # Aggregate and generate graph
    data = aggregate_data(events, time_frame)
    graph = generate_graph(data, time_frame)
    return graph

def aggregate_data(events, time_frame):
    data = {"severity": {}, "event_type": {}}

    for severity in SEVERITIES:
        data["severity"][severity] = 0
    for event_type in EVENTTYPES:
        data["event_type"][event_type] = 0 
        
    for event in events:
        data["severity"][event.Severity] += 1
        data["event_type"][event.EventType] += 1

    if time_frame == "week":
        bucket_key = lambda event: event.Timestamp.date()
    elif time_frame == "month":
        bucket_key = lambda event: event.Timestamp.isocalendar()[1]
    elif time_frame == "year":
        bucket_key = lambda event: event.Timestamp.month
    elif time_frame == "decade":
        bucket_key = lambda event: event.Timestamp.year

    buckets = {}
    for event in events:
        key = bucket_key(event)
        if key not in buckets:
            buckets[key] = {"severity": {}, "event_type": {}}
            for severity in SEVERITIES:
                buckets[key]["severity"][severity] = 0
            for event_type in EVENTTYPES:
                buckets[key]["event_type"][event_type] = 0

        buckets[key]["severity"][event.Severity] += 1
        buckets[key]["event_type"][event.EventType] += 1
    print(buckets)
    return buckets

def generate_graph(data, time_frame):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from flask import send_file

    plt.figure(figsize=(12, 8))

    if time_frame == "week":
        x_labels = [key.strftime('%Y-%m-%d') for key in data.keys()]
    elif time_frame == "month":
        x_labels = [f"Week {key}" for key in data.keys()]
    elif time_frame == "year":
        x_labels = [f"Month {key}" for key in data.keys()]
    elif time_frame == "decade":
        x_labels = [f"Year {key}" for key in data.keys()]

    x_indices = range(len(x_labels))
    severity_data = {severity: [bucket["severity"].get(severity, 0) for bucket in data.values()] for severity in SEVERITIES}
    event_type_data = {event_type: [bucket["event_type"].get(event_type, 0) for bucket in data.values()] for event_type in EVENTTYPES}

    # Colors for severity levels
    severity_colors = {
        "high": "red",
        "medium": "orange",
        "low": "green"
    }

    # Colors for event types (using a colormap for distinct colors and applying alpha for faded effect)
    event_type_colors = plt.cm.tab10(range(len(EVENTTYPES)))
    faded_alpha = 0.5

    # Offsets for severity and event type bars
    severity_offset = -0.2  # Shift severity bars slightly left
    event_type_offset = 0.2  # Shift event type bars slightly right
    bar_width = 0.4

    # Plot severity bars
    severity_bottom = [0] * len(x_indices)
    for severity, counts in severity_data.items():
        color = severity_colors.get(severity, "gray")  # Default to gray if not specified
        plt.bar(
            [x + severity_offset for x in x_indices],
            counts,
            width=bar_width,
            bottom=severity_bottom,
            color=color,
            label=f"{severity.capitalize()} (Severity)"
        )
        # Update bottom for stacking
        severity_bottom = [sum(x) for x in zip(severity_bottom, counts)]

    # Plot event type bars
    event_type_bottom = [0] * len(x_indices)
    for i, (event_type, counts) in enumerate(event_type_data.items()):
        color = event_type_colors[i]
        plt.bar(
            [x + event_type_offset for x in x_indices],
            counts,
            width=bar_width,
            bottom=event_type_bottom,
            color=color,
            label=f"{event_type.capitalize()} (Event Type)",
            alpha=faded_alpha  # Apply faded effect
        )
        # Update bottom for stacking
        event_type_bottom = [sum(x) for x in zip(event_type_bottom, counts)]

    # Custom Legend
    severity_legend = [
        plt.Line2D([0], [0], color="red", lw=4, label="High (Severity)"),
        plt.Line2D([0], [0], color="orange", lw=4, label="Medium (Severity)"),
        plt.Line2D([0], [0], color="green", lw=4, label="Low (Severity)")
    ]
    event_type_legend = [
        plt.Line2D([0], [0], color=event_type_colors[i], lw=4, alpha=faded_alpha, label=f"{event_type.capitalize()} (Event Type)")
        for i, event_type in enumerate(EVENTTYPES)
    ]

    # Add legend with a short title
    plt.legend(
        handles=severity_legend + event_type_legend,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Legend"
    )

    plt.xticks(x_indices, x_labels, rotation=45, ha="right")
    plt.xlabel("Time Buckets")
    plt.ylabel("Counts")
    plt.title(f"Events in {request.form.get('location')} ({time_frame.capitalize()})")

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close()
    return send_file(buffer, mimetype='image/png')





# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
