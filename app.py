from flask import Flask, render_template, request, redirect, session
import pandas as pd
import pickle
import sqlite3
import random
import os
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- FIX PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)

model_path = os.path.join(BASE_DIR, "model.pkl")
data_path = os.path.join(BASE_DIR, "sales.csv")
print("Data path:", data_path)
db_path = os.path.join(BASE_DIR, "users.db")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open(model_path, "rb"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv(data_path)
print("Loaded df shape:", df.shape)
print("DF head:", df.head())
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("After to_datetime, Date sample:", df['Date'].head())
df = df.dropna().sort_values('Date')
print("After processing df shape:", df.shape)

# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    conn.commit()
    conn.close()

init_db()

# ---------------- HOME ----------------
@app.route("/")
def home():
    # if 'user' not in session:
    #     return redirect("/login")

    recent = df.tail(10)
    labels = recent['Date'].dt.strftime('%d-%m').tolist()
    values = recent['Sales'].tolist()

    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    wk = df.groupby('week')['Sales'].sum().tail(5)

    print("In home, df shape:", df.shape)
    print("Labels length:", len(labels))
    print("Labels sample:", labels[:3] if labels else "Empty")

    return render_template("index.html",
        prediction_text="Enter values to predict",
        labels=labels,
        values=values,
        week_labels=wk.index.astype(str).tolist(),
        week_values=wk.values.tolist(),
        compare_labels=["ML","ARIMA"],
        compare_values=[0,0]
    )

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get('username')
        p = request.form.get('password')

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        r = c.fetchone()
        conn.close()

        if r:
            session['user'] = u
            return redirect("/")
        return render_template("login.html", error="Invalid ❌")

    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        u = request.form.get('username')
        p = request.form.get('password')

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?)",(u,p))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------------- FORGOT ----------------
@app.route("/forgot", methods=["GET","POST"])
def forgot():
    if request.method == "POST":
        otp = str(random.randint(1000,9999))
        session['otp'] = otp
        session['user_reset'] = request.form.get('username')
        return render_template("verify.html", otp_display=otp)

    return render_template("forgot.html")

# ---------------- VERIFY ----------------
@app.route("/verify", methods=["POST"])
def verify():
    if request.form.get('otp') == session.get('otp'):
        return redirect("/reset")
    return render_template("verify.html", error="Wrong OTP ❌")

# ---------------- RESET ----------------
@app.route("/reset", methods=["GET","POST"])
def reset():
    if request.method == "POST":
        new = request.form.get('password')
        u = session.get('user_reset')

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE username=?", (new,u))
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("reset.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    # Base data
    recent = df.tail(10)
    labels = recent['Date'].dt.strftime('%d-%m').tolist()
    values = recent['Sales'].tolist()

    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    wk = df.groupby('week')['Sales'].sum().tail(5)

    week_labels = wk.index.astype(str).tolist()
    week_values = wk.values.tolist()
    compare_labels = ["ML","ARIMA"]
    compare_values = [0,0]

    try:
        data = {
            'year':[int(request.form.get('year',0))],
            'month':[int(request.form.get('month',0))],
            'day':[int(request.form.get('day',0))],
            'week':[int(request.form.get('week',0))],
            'lag_1':[float(request.form.get('lag_1',0))],
            'lag_7':[float(request.form.get('lag_7',0))]
        }

        X = pd.DataFrame(data)
        pred = model.predict(X)[0]

        # Update for prediction
        labels.append("Pred")
        values.append(round(pred,2))

        # ARIMA
        ts = df.set_index('Date')['Sales']
        arima = ARIMA(ts, order=(5,1,0)).fit()
        ar_pred = arima.forecast(1)[0]

        compare_values = [round(pred,2),round(ar_pred,2)]

        prediction_text = f"Prediction: {round(pred,2)}"

    except Exception as e:
        prediction_text = f"Error: {str(e)}"

    return render_template("index.html",
        prediction_text=prediction_text,
        labels=labels,
        values=values,
        week_labels=week_labels,
        week_values=week_values,
        compare_labels=compare_labels,
        compare_values=compare_values
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)