"""
=====================================================================
 SMART HOTEL / HOSTEL ROOM BOOKING SYSTEM
 with Booking-Cancellation Prediction and Dynamic Price Suggestion
 (DBMS + Machine Learning project, single file)
=====================================================================
HOW TO RUN
    1) pip install streamlit pandas numpy scikit-learn
    2) streamlit run hotel_booking_system.py

DBMS PART  : SQLite database with 8 tables, primary/foreign keys, CHECK
             constraints, 3 triggers, 2 views, joins, group by, subqueries,
             and a bill-calculation function (stored-procedure equivalent).
ML PART    : (a) Cancellation prediction  - classification
                 (Logistic Regression vs Random Forest)
             (b) Dynamic price suggestion  - regression
                 (Linear Regression vs Random Forest)
             Optional real data: put the Kaggle file "hotel_bookings.csv"
             ("Hotel Booking Demand") in the same folder. If it is not
             there, realistic synthetic data is generated automatically.
=====================================================================
"""
import os
import random
import sqlite3
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_FILE = "hotel.db"
GST = 0.12
DEPOSIT = {0: "No deposit", 1: "Refundable", 2: "Non-refundable"}

# =====================================================================
# 1. DATABASE LAYER  (DBMS part)
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS Customer(
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT UNIQUE NOT NULL,
    email TEXT,
    id_proof TEXT,
    prev_cancellations INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS RoomType(
    type_id INTEGER PRIMARY KEY,
    type_name TEXT NOT NULL,
    base_price REAL NOT NULL CHECK(base_price > 0),
    capacity INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS Room(
    room_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_no TEXT UNIQUE NOT NULL,
    type_id INTEGER NOT NULL REFERENCES RoomType(type_id),
    floor INTEGER,
    status TEXT DEFAULT 'Available' CHECK(status IN ('Available','Occupied','Maintenance'))
);
CREATE TABLE IF NOT EXISTS Staff(
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT
);
CREATE TABLE IF NOT EXISTS Booking(
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES Customer(customer_id),
    room_id INTEGER NOT NULL REFERENCES Room(room_id),
    booking_date TEXT,
    check_in TEXT NOT NULL,
    check_out TEXT NOT NULL,
    guests INTEGER NOT NULL,
    deposit_type INTEGER DEFAULT 0,
    price_per_night REAL NOT NULL,
    status TEXT DEFAULT 'Booked'
        CHECK(status IN ('Booked','Checked-In','Checked-Out','Cancelled')),
    CHECK(check_out > check_in)
);
CREATE TABLE IF NOT EXISTS Payment(
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id INTEGER NOT NULL REFERENCES Booking(booking_id),
    amount REAL NOT NULL CHECK(amount > 0),
    mode TEXT,
    pay_date TEXT
);
CREATE TABLE IF NOT EXISTS Feedback(
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id INTEGER NOT NULL REFERENCES Booking(booking_id),
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comments TEXT
);
CREATE TABLE IF NOT EXISTS Prediction_Log(
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id INTEGER NOT NULL REFERENCES Booking(booking_id),
    cancel_probability REAL,
    suggested_price REAL,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- TRIGGERS ---------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_checkin
AFTER UPDATE OF status ON Booking WHEN NEW.status = 'Checked-In'
BEGIN
    UPDATE Room SET status = 'Occupied' WHERE room_id = NEW.room_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_release_room
AFTER UPDATE OF status ON Booking WHEN NEW.status IN ('Checked-Out','Cancelled')
BEGIN
    UPDATE Room SET status = 'Available' WHERE room_id = NEW.room_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_cancel_count
AFTER UPDATE OF status ON Booking WHEN NEW.status = 'Cancelled'
BEGIN
    UPDATE Customer SET prev_cancellations = prev_cancellations + 1
    WHERE customer_id = NEW.customer_id;
END;

-- VIEWS ------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_occupancy AS
SELECT rt.type_name AS room_type,
       COUNT(r.room_id) AS total_rooms,
       SUM(r.status = 'Occupied') AS occupied_now
FROM Room r JOIN RoomType rt ON r.type_id = rt.type_id
GROUP BY rt.type_name;

CREATE VIEW IF NOT EXISTS v_booking_details AS
SELECT b.booking_id, c.name AS customer, r.room_no, rt.type_name AS room_type,
       b.check_in, b.check_out, b.guests, b.price_per_night, b.status
FROM Booking b
JOIN Customer c  ON b.customer_id = c.customer_id
JOIN Room r      ON b.room_id = r.room_id
JOIN RoomType rt ON r.type_id = rt.type_id;
"""


def calc_bill(nights, price_per_night, gst=GST):
    """Bill = nights x price + GST (acts as the 'stored procedure')."""
    return round(nights * price_per_night * (1 + gst), 2)


def seed(conn):
    """Insert sample data the first time the app runs."""
    if conn.execute("SELECT COUNT(*) FROM RoomType").fetchone()[0]:
        return
    rnd = random.Random(1)
    prices = {1: 1500, 2: 2500, 3: 4000, 4: 6500}
    conn.executemany("INSERT INTO RoomType VALUES(?,?,?,?)",
                     [(1, "Single", 1500, 1), (2, "Double", 2500, 2),
                      (3, "Deluxe", 4000, 3), (4, "Suite", 6500, 4)])
    for t in range(1, 5):
        for i in range(1, 5):
            conn.execute("INSERT INTO Room(room_no,type_id,floor) VALUES(?,?,?)",
                         (f"{t}0{i}", t, t))
    conn.executemany("INSERT INTO Staff(name,role) VALUES(?,?)",
                     [("Ramesh Patil", "Manager"), ("Sneha Joshi", "Receptionist"),
                      ("Amit Kulkarni", "Housekeeping")])
    names = ["Aarav Sharma", "Priya Nair", "Rohan Mehta", "Ananya Iyer",
             "Karan Singh", "Neha Gupta", "Vikram Rao", "Isha Kapoor",
             "Arjun Desai", "Meera Joshi", "Rahul Verma", "Sana Khan"]
    for i, n in enumerate(names):
        conn.execute("INSERT INTO Customer(name,phone,email,id_proof) VALUES(?,?,?,?)",
                     (n, f"98765{10000 + i}", f"{n.split()[0].lower()}@mail.com",
                      f"ID-{1000 + i}"))
    today = date.today()

    def add(cid, rid, cin, nights, status, deposit):
        tid = conn.execute("SELECT type_id FROM Room WHERE room_id=?", (rid,)).fetchone()[0]
        cout = cin + timedelta(days=nights)
        clash = conn.execute(
            "SELECT 1 FROM Booking WHERE room_id=? AND status<>'Cancelled' "
            "AND check_in<? AND check_out>?",
            (rid, cout.isoformat(), cin.isoformat())).fetchone()
        if clash:
            return
        price = prices[tid]
        cur = conn.execute(
            "INSERT INTO Booking(customer_id,room_id,booking_date,check_in,check_out,"
            "guests,deposit_type,price_per_night,status) VALUES(?,?,?,?,?,?,?,?,?)",
            (cid, rid, (cin - timedelta(days=rnd.randint(1, 40))).isoformat(),
             cin.isoformat(), cout.isoformat(), rnd.randint(1, tid), deposit, price, status))
        bid = cur.lastrowid
        if status == "Checked-Out":
            conn.execute("INSERT INTO Payment(booking_id,amount,mode,pay_date) VALUES(?,?,?,?)",
                         (bid, calc_bill(nights, price), rnd.choice(["UPI", "Card", "Cash"]),
                          cout.isoformat()))
            if rnd.random() < 0.7:
                conn.execute("INSERT INTO Feedback(booking_id,rating,comments) VALUES(?,?,?)",
                             (bid, rnd.randint(3, 5), "Good stay"))

    for _ in range(45):  # past bookings
        add(rnd.randint(1, 12), rnd.randint(1, 16),
            today - timedelta(days=rnd.randint(8, 100)), rnd.randint(1, 5),
            "Cancelled" if rnd.random() < 0.25 else "Checked-Out", rnd.choice([0, 0, 1, 2]))
    add(1, 5, today - timedelta(days=1), 3, "Checked-In", 0)          # staying now
    conn.execute("UPDATE Room SET status='Occupied' WHERE room_id=5")
    for _ in range(4):  # upcoming bookings
        add(rnd.randint(1, 12), rnd.randint(6, 16),
            today + timedelta(days=rnd.randint(3, 25)), rnd.randint(1, 4), "Booked", 1)
    conn.execute("UPDATE Customer SET prev_cancellations = (SELECT COUNT(*) FROM Booking "
                 "WHERE Booking.customer_id = Customer.customer_id AND status='Cancelled')")
    conn.commit()


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    seed(conn)
    return conn


conn = get_conn()


def q(sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


def run(sql, params=()):
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def available_rooms(type_id, cin, cout):
    """Rooms of a type with no overlapping active booking in [cin, cout)."""
    return q("""SELECT room_id, room_no FROM Room
                WHERE type_id = ? AND status <> 'Maintenance' AND room_id NOT IN (
                    SELECT room_id FROM Booking
                    WHERE status IN ('Booked','Checked-In')
                      AND check_in < ? AND check_out > ?)""",
             (int(type_id), cout.isoformat(), cin.isoformat()))


# =====================================================================
# 2. MACHINE LEARNING LAYER
# =====================================================================
FEATURES = ["lead_time", "stay_nights", "guests", "room_type",
            "deposit_type", "prev_cancellations", "price_per_night", "month"]
PRICE_FEATURES = ["room_type", "month", "guests", "lead_time", "stay_nights"]
BASE = np.array([0, 1500, 2500, 4000, 6500])


def make_synthetic(n=6000, seed_=42):
    """Realistic synthetic booking data (used when Kaggle CSV is absent)."""
    rng = np.random.default_rng(seed_)
    room_type = rng.integers(1, 5, n)
    month = rng.integers(1, 13, n)
    lead = rng.integers(0, 300, n)
    stay = rng.integers(1, 10, n)
    guests = np.clip(rng.integers(1, 5, n), 1, room_type)
    deposit = rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])
    prev = rng.choice([0, 1, 2, 3], n, p=[0.8, 0.12, 0.05, 0.03])
    season = np.where(np.isin(month, [4, 5, 10, 11, 12]), 0.25, 0.0)
    early_discount = 1 - np.minimum(lead, 120) / 120 * 0.10
    price = BASE[room_type] * (1 + season) * early_discount + rng.normal(0, 100, n)
    price = np.clip(price, 500, None)
    logit = (-2.4 + 0.011 * lead + 0.55 * prev - 2.5 * (deposit == 2)
             - 0.4 * (deposit == 1) - 0.08 * stay + 0.00005 * price)
    canceled = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(dict(lead_time=lead, stay_nights=stay, guests=guests,
                             room_type=room_type, deposit_type=deposit,
                             prev_cancellations=prev, price_per_night=price.round(0),
                             month=month, canceled=canceled))


def load_kaggle(path):
    """Convert the Kaggle 'Hotel Booking Demand' CSV to our feature format."""
    d = pd.read_csv(path)
    d["stay_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["guests"] = (d["adults"] + d["children"].fillna(0)).astype(int)
    d["deposit_type"] = d["deposit_type"].map(
        {"No Deposit": 0, "Refundable": 1, "Non Refund": 2}).fillna(0).astype(int)
    d["month"] = d["arrival_date_month"].map({m: i for i, m in enumerate(calendar.month_name) if m})
    d["room_type"] = d["reserved_room_type"].map(lambda x: min(max(ord(str(x)[0]) - 64, 1), 4))
    d["price_per_night"] = d["adr"] * 50          # rough EUR -> INR-like scale
    d = d.rename(columns={"previous_cancellations": "prev_cancellations",
                          "is_canceled": "canceled"})
    d = d[(d.stay_nights > 0) & (d.guests > 0) & (d.adr > 0) & (d.adr < 1000)]
    return d[FEATURES + ["canceled"]].dropna()


@st.cache_resource(show_spinner="Training ML models...")
def train_models():
    if os.path.exists("hotel_bookings.csv"):
        df = load_kaggle("hotel_bookings.csv")
        df = df.sample(min(len(df), 20000), random_state=1)
        source = f"Kaggle 'Hotel Booking Demand' dataset ({len(df)} bookings used)"
    else:
        df = make_synthetic()
        source = f"Synthetic data ({len(df)} bookings) - add hotel_bookings.csv for real data"

    # ---- (a) cancellation classifier ---------------------------------
    X, y = df[FEATURES], df["canceled"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12,
                                                random_state=42, n_jobs=-1),
    }
    rows, cms, best, best_f1, best_name = [], {}, None, -1, ""
    for name, m in candidates.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        f1 = f1_score(yte, p)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, zero_division=0),
                     "Recall": recall_score(yte, p), "F1-score": f1})
        cms[name] = pd.DataFrame(confusion_matrix(yte, p),
                                 index=["Actual: Stayed", "Actual: Cancelled"],
                                 columns=["Pred: Stayed", "Pred: Cancelled"])
        if f1 > best_f1:
            best, best_f1, best_name = m, f1, name
    importance = pd.Series(candidates["Random Forest"].feature_importances_,
                           index=FEATURES).sort_values(ascending=False)

    # ---- (b) price regressor ----------------------------------------
    pdf = make_synthetic(4000, seed_=7)
    Xp, yp = pdf[PRICE_FEATURES], pdf["price_per_night"]
    Xptr, Xpte, yptr, ypte = train_test_split(Xp, yp, test_size=0.2, random_state=42)
    regs = {"Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                                   random_state=42, n_jobs=-1)}
    prow, best_reg, best_r2, best_reg_name = [], None, -9, ""
    for name, m in regs.items():
        m.fit(Xptr, yptr)
        p = m.predict(Xpte)
        r2 = r2_score(ypte, p)
        prow.append({"Model": name, "MAE (Rs)": mean_absolute_error(ypte, p), "R2 score": r2})
        if r2 > best_r2:
            best_reg, best_r2, best_reg_name = m, r2, name

    return dict(source=source, clf=best, clf_name=best_name,
                clf_table=pd.DataFrame(rows).round(3), cms=cms, importance=importance,
                reg=best_reg, reg_name=best_reg_name, reg_table=pd.DataFrame(prow).round(3))


def cancel_probability(models, row):
    return float(models["clf"].predict_proba(pd.DataFrame([row])[FEATURES])[0][1])


def suggest_price(models, type_id, month, guests, lead, nights):
    X = pd.DataFrame([dict(room_type=type_id, month=month, guests=guests,
                           lead_time=lead, stay_nights=nights)])[PRICE_FEATURES]
    p = float(models["reg"].predict(X)[0])
    base = float(BASE[type_id])
    return round(min(max(p, 0.8 * base), 1.5 * base), -1)   # keep within sane range


# =====================================================================
# 3. USER INTERFACE (Streamlit)
# =====================================================================
st.set_page_config(page_title="Smart Hotel Booking System", page_icon="🏨", layout="wide")
models = train_models()

st.sidebar.title("🏨 Smart Hotel System")
page = st.sidebar.radio("Menu", ["Dashboard", "New Booking", "Manage Bookings",
                                 "SQL Query Lab", "ML Insights"])
st.sidebar.caption("DBMS")

# ---------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("📊 Dashboard")
    total_rooms = q("SELECT COUNT(*) n FROM Room").n[0]
    occupied = q("SELECT COUNT(*) n FROM Room WHERE status='Occupied'").n[0]
    bookings = q("SELECT COUNT(*) n FROM Booking").n[0]
    revenue = q("SELECT COALESCE(SUM(amount),0) s FROM Payment").s[0]
    rating = q("SELECT AVG(rating) a FROM Feedback").a[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rooms", int(total_rooms))
    c2.metric("Occupied now", int(occupied))
    c3.metric("Total bookings", int(bookings))
    c4.metric("Revenue (Rs)", f"{revenue:,.0f}")
    c5.metric("Avg rating", f"{rating:.1f} / 5" if rating == rating else "-")

    left, right = st.columns(2)
    with left:
        st.subheader("Occupancy by room type (view)")
        st.dataframe(q("SELECT * FROM v_occupancy"), hide_index=True)
        st.subheader("Bookings by status")
        st.bar_chart(q("SELECT status, COUNT(*) AS bookings FROM Booking GROUP BY status")
                     .set_index("status"))
    with right:
        st.subheader("Monthly revenue")
        st.bar_chart(q("SELECT strftime('%Y-%m', pay_date) AS month, SUM(amount) AS revenue "
                       "FROM Payment GROUP BY month ORDER BY month").set_index("month"))
        st.subheader("Upcoming / current stays")
        st.dataframe(q("SELECT * FROM v_booking_details WHERE status IN ('Booked','Checked-In') "
                       "ORDER BY check_in"), hide_index=True)

# -------------------------------------------------------------- New Booking
elif page == "New Booking":
    st.title("🛏️ New Booking (with AI assistance)")
    types = q("SELECT * FROM RoomType")
    type_ids = types.type_id.tolist()
    tinfo = types.set_index("type_id")

    st.subheader("1. Customer")
    mode = st.radio("Customer type", ["Existing customer", "New customer"], horizontal=True)
    prev_cancel, cid = 0, None
    if mode == "Existing customer":
        cust = q("SELECT customer_id, name, phone, prev_cancellations FROM Customer ORDER BY name")
        cid = st.selectbox("Select customer", cust.customer_id.tolist(),
                           format_func=lambda i: "{} ({})".format(
                               cust.set_index("customer_id").loc[i, "name"],
                               cust.set_index("customer_id").loc[i, "phone"]))
        prev_cancel = int(cust.set_index("customer_id").loc[cid, "prev_cancellations"])
    else:
        a, b = st.columns(2)
        new_name = a.text_input("Full name")
        new_phone = b.text_input("Phone")
        new_email = a.text_input("Email")
        new_id = b.text_input("ID proof number")

    st.subheader("2. Stay details")
    a, b, c, d = st.columns(4)
    cin = a.date_input("Check-in", date.today() + timedelta(days=7), min_value=date.today())
    cout = b.date_input("Check-out", date.today() + timedelta(days=9), min_value=date.today())
    tid = c.selectbox("Room type", type_ids, format_func=lambda i: tinfo.loc[i, "type_name"])
    guests = d.number_input("Guests", 1, 6, 1)
    dep = st.selectbox("Deposit type", list(DEPOSIT.keys()), format_func=lambda k: DEPOSIT[k])

    nights = max((cout - cin).days, 1)
    lead = max((cin - date.today()).days, 0)
    ai_price = suggest_price(models, int(tid), cin.month, int(guests), lead, nights)
    base_price = float(tinfo.loc[tid, "base_price"])

    st.subheader("3. Pricing and AI insights")
    choice = st.radio("Price option", [f"Standard price (Rs {base_price:,.0f})",
                                       f"AI suggested price (Rs {ai_price:,.0f})"], horizontal=True)
    price = base_price if choice.startswith("Standard") else ai_price
    risk = cancel_probability(models, dict(lead_time=lead, stay_nights=nights, guests=int(guests),
                                           room_type=int(tid), deposit_type=int(dep),
                                           prev_cancellations=prev_cancel,
                                           price_per_night=price, month=cin.month))
    avail = available_rooms(tid, cin, cout) if cout > cin else pd.DataFrame()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nights", nights)
    m2.metric("Estimated bill (incl. GST)", f"Rs {calc_bill(nights, price):,.0f}")
    m3.metric("Cancellation risk", f"{risk * 100:.0f}%")
    m4.metric("Rooms available", len(avail))
    st.progress(min(max(risk, 0.0), 1.0))
    if risk > 0.6:
        st.warning("High cancellation risk. Suggest a non-refundable or refundable deposit.")
    elif risk > 0.3:
        st.info("Medium cancellation risk. Send a reminder before check-in.")
    else:
        st.success("Low cancellation risk.")

    st.subheader("4. Payment (optional advance)")
    a, b = st.columns(2)
    advance = a.number_input("Advance paid (Rs)", 0.0, 1000000.0, 0.0, step=500.0)
    pay_mode = b.selectbox("Payment mode", ["UPI", "Card", "Cash"])

    if st.button("✅ Confirm Booking", type="primary"):
        if cout <= cin:
            st.error("Check-out date must be after check-in date.")
        elif guests > tinfo.loc[tid, "capacity"]:
            st.error(f"This room type allows at most {tinfo.loc[tid, 'capacity']} guest(s).")
        elif avail.empty:
            st.error("No room of this type is available for the selected dates.")
        else:
            try:
                if mode == "New customer":
                    if not new_name.strip() or not new_phone.strip():
                        st.error("Name and phone are required.")
                        st.stop()
                    cid = run("INSERT INTO Customer(name,phone,email,id_proof) VALUES(?,?,?,?)",
                              (new_name.strip(), new_phone.strip(), new_email, new_id)).lastrowid
                rid = int(avail.room_id.iloc[0])
                bid = run("INSERT INTO Booking(customer_id,room_id,booking_date,check_in,check_out,"
                          "guests,deposit_type,price_per_night,status) VALUES(?,?,?,?,?,?,?,?,'Booked')",
                          (int(cid), rid, date.today().isoformat(), cin.isoformat(),
                           cout.isoformat(), int(guests), int(dep), price)).lastrowid
                if advance > 0:
                    run("INSERT INTO Payment(booking_id,amount,mode,pay_date) VALUES(?,?,?,?)",
                        (bid, float(advance), pay_mode, date.today().isoformat()))
                run("INSERT INTO Prediction_Log(booking_id,cancel_probability,suggested_price) "
                    "VALUES(?,?,?)", (bid, risk, ai_price))
                st.success(f"Booking #{bid} confirmed. Room {avail.room_no.iloc[0]} allotted.")
                st.balloons()
            except sqlite3.IntegrityError as e:
                st.error(f"Could not save booking: {e} (phone number may already exist).")

# --------------------------------------------------------- Manage Bookings
elif page == "Manage Bookings":
    st.title("🧾 Manage Bookings")
    active = q("SELECT * FROM v_booking_details WHERE status IN ('Booked','Checked-In') "
               "ORDER BY check_in")
    if active.empty:
        st.info("No active bookings.")
    else:
        bid = st.selectbox("Select booking", active.booking_id.tolist(),
                           format_func=lambda i: "#{} - {} (Room {}, {})".format(
                               i, active.set_index("booking_id").loc[i, "customer"],
                               active.set_index("booking_id").loc[i, "room_no"],
                               active.set_index("booking_id").loc[i, "status"]))
        row = active.set_index("booking_id").loc[bid]
        nights = (date.fromisoformat(row.check_out) - date.fromisoformat(row.check_in)).days
        bill = calc_bill(nights, float(row.price_per_night))
        paid = float(q("SELECT COALESCE(SUM(amount),0) s FROM Payment WHERE booking_id=?",
                       (int(bid),)).s[0])
        a, b, c = st.columns(3)
        a.metric("Total bill", f"Rs {bill:,.0f}")
        b.metric("Paid", f"Rs {paid:,.0f}")
        c.metric("Balance", f"Rs {max(bill - paid, 0):,.0f}")

        if row.status == "Booked":
            x, y = st.columns(2)
            if x.button("Check-In guest"):
                run("UPDATE Booking SET status='Checked-In' WHERE booking_id=?", (int(bid),))
                st.success("Checked in. Room marked Occupied by trigger.")
            if y.button("Cancel booking"):
                run("UPDATE Booking SET status='Cancelled' WHERE booking_id=?", (int(bid),))
                st.warning("Booking cancelled. Customer's cancellation count updated by trigger.")
        else:
            mode = st.selectbox("Payment mode for balance", ["UPI", "Card", "Cash"])
            rating = st.slider("Guest rating", 1, 5, 4)
            comment = st.text_input("Feedback comment")
            if st.button("Collect balance & Check-Out"):
                if bill - paid > 0:
                    run("INSERT INTO Payment(booking_id,amount,mode,pay_date) VALUES(?,?,?,?)",
                        (int(bid), round(bill - paid, 2), mode, date.today().isoformat()))
                run("INSERT INTO Feedback(booking_id,rating,comments) VALUES(?,?,?)",
                    (int(bid), rating, comment))
                run("UPDATE Booking SET status='Checked-Out' WHERE booking_id=?", (int(bid),))
                st.success("Checked out. Room released by trigger.")

    st.subheader("All bookings")
    st.dataframe(q("SELECT * FROM v_booking_details ORDER BY booking_id DESC"), hide_index=True)
    st.subheader("Customers")
    st.dataframe(q("SELECT * FROM Customer"), hide_index=True)

# ---------------------------------------------------------------- SQL Lab
elif page == "SQL Query Lab":
    st.title("🧪 SQL Query Lab (for viva demo)")
    QUERIES = {
        "Rooms available today (subquery)":
            "SELECT r.room_no, rt.type_name, rt.base_price\nFROM Room r JOIN RoomType rt ON r.type_id = rt.type_id\n"
            "WHERE r.room_id NOT IN (SELECT room_id FROM Booking\n"
            "    WHERE status IN ('Booked','Checked-In')\n"
            "    AND check_in <= date('now') AND check_out > date('now'))",
        "Monthly revenue (group by)":
            "SELECT strftime('%Y-%m', pay_date) AS month, SUM(amount) AS revenue\n"
            "FROM Payment GROUP BY month ORDER BY month",
        "Top customers (joins + group by)":
            "SELECT c.name, COUNT(DISTINCT b.booking_id) AS bookings, SUM(p.amount) AS total_spent\n"
            "FROM Customer c JOIN Booking b ON c.customer_id = b.customer_id\n"
            "JOIN Payment p ON b.booking_id = p.booking_id\n"
            "GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5",
        "Customers spending above average (nested subquery)":
            "SELECT name FROM Customer WHERE customer_id IN (\n"
            "  SELECT b.customer_id FROM Booking b JOIN Payment p ON b.booking_id = p.booking_id\n"
            "  GROUP BY b.customer_id\n"
            "  HAVING SUM(p.amount) > (SELECT AVG(t) FROM (\n"
            "      SELECT SUM(p2.amount) AS t FROM Payment p2\n"
            "      JOIN Booking b2 ON p2.booking_id = b2.booking_id GROUP BY b2.customer_id)))",
        "Cancellation rate by room type":
            "SELECT rt.type_name, COUNT(*) AS total, SUM(b.status='Cancelled') AS cancelled,\n"
            "  ROUND(100.0 * SUM(b.status='Cancelled') / COUNT(*), 1) AS cancel_percent\n"
            "FROM Booking b JOIN Room r ON b.room_id = r.room_id\n"
            "JOIN RoomType rt ON r.type_id = rt.type_id GROUP BY rt.type_name",
        "Average rating per room type":
            "SELECT rt.type_name, ROUND(AVG(f.rating), 2) AS avg_rating, COUNT(*) AS reviews\n"
            "FROM Feedback f JOIN Booking b ON f.booking_id = b.booking_id\n"
            "JOIN Room r ON b.room_id = r.room_id JOIN RoomType rt ON r.type_id = rt.type_id\n"
            "GROUP BY rt.type_name",
        "View: current occupancy": "SELECT * FROM v_occupancy",
        "ML prediction log (joins)":
            "SELECT pl.booking_id, c.name, ROUND(pl.cancel_probability*100,1) AS cancel_risk_percent,\n"
            "       pl.suggested_price, b.status\n"
            "FROM Prediction_Log pl JOIN Booking b ON pl.booking_id = b.booking_id\n"
            "JOIN Customer c ON b.customer_id = c.customer_id ORDER BY pl.log_id DESC",
    }
    name = st.selectbox("Choose a query", list(QUERIES.keys()))
    sql = st.text_area("SQL (you can edit it; only SELECT is allowed)", QUERIES[name], height=170)
    if st.button("Run query"):
        if not sql.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed here.")
        else:
            try:
                st.dataframe(q(sql), hide_index=True)
            except Exception as e:
                st.error(f"SQL error: {e}")
    with st.expander("Show table structure (for ER diagram)"):
        for t in ["Customer", "RoomType", "Room", "Staff", "Booking", "Payment", "Feedback",
                  "Prediction_Log"]:
            st.markdown(f"**{t}**")
            st.dataframe(q(f"PRAGMA table_info({t})")[["name", "type", "notnull", "pk"]],
                         hide_index=True)

# ------------------------------------------------------------ ML Insights
else:
    st.title("🤖 ML Insights")
    st.caption(f"Data source: {models['source']}")
    st.subheader("A. Cancellation prediction (classification)")
    st.dataframe(models["clf_table"], hide_index=True)
    st.success(f"Best model used in the app: {models['clf_name']}")
    c1, c2 = st.columns(2)
    for col, (nm, cm) in zip([c1, c2], models["cms"].items()):
        col.markdown(f"**Confusion matrix - {nm}**")
        col.dataframe(cm)
    st.subheader("Feature importance (Random Forest)")
    st.bar_chart(models["importance"])

    st.subheader("B. Dynamic price suggestion (regression)")
    st.dataframe(models["reg_table"], hide_index=True)
    st.success(f"Best model used in the app: {models['reg_name']}")

    st.subheader("Try it yourself")
    a, b, c, d = st.columns(4)
    t = a.selectbox("Room type", [1, 2, 3, 4],
                    format_func=lambda i: ["", "Single", "Double", "Deluxe", "Suite"][i])
    mth = b.selectbox("Month", list(range(1, 13)), format_func=lambda i: calendar.month_name[i])
    ld = c.slider("Lead time (days)", 0, 300, 60)
    stay = d.slider("Nights", 1, 10, 2)
    e, f, g = st.columns(3)
    gst_n = e.slider("Guests", 1, 4, min(2, t))
    dp = f.selectbox("Deposit", list(DEPOSIT.keys()), format_func=lambda k: DEPOSIT[k])
    pc = g.slider("Previous cancellations", 0, 3, 0)
    sp = suggest_price(models, t, mth, gst_n, ld, stay)
    pr = cancel_probability(models, dict(lead_time=ld, stay_nights=stay, guests=gst_n,
                                         room_type=t, deposit_type=dp, prev_cancellations=pc,
                                         price_per_night=sp, month=mth))
    x, y = st.columns(2)
    x.metric("Suggested price / night", f"Rs {sp:,.0f}")
    y.metric("Cancellation probability", f"{pr * 100:.0f}%")
