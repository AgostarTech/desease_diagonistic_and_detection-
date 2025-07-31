# =========================================
# Imports & App Setup
# =========================================
import csv
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import spacy
from flask import Flask, render_template, request, session, redirect, url_for, flash
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize  # kept to preserve original imports/usage
from nltk.wsd import lesk
from spacy.lang.en.stop_words import STOP_WORDS
from flask import render_template, request, redirect, url_for, session, flash, abort
from forms import LoginForm, SignupForm


# Create the Flask app (single instance)
app = Flask(__name__)

# Load spaCy English model (used by text preprocessing)
nlp = spacy.load('en_core_web_sm')

# -------------------------------------------------
# Persisted JSON store (kept same behavior)
# This overwrites DATA.json on each startup, as in original.
# -------------------------------------------------
data = {"users": []}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)


# =========================================
# Utility & NLP / ML Helper Functions
# =========================================

def write_json(new_data, filename='DATA.json'):
    """
    Append a new user entry to DATA.json.
    - Loads the existing file,
    - Appends the `new_data` dict under "users",
    - Writes back to disk.
    """
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["users"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)


# -----------------------------------------
# Load datasets (Training/Testing)
# -----------------------------------------
df_tr = pd.read_csv('Medical_dataset/Training.csv')
df_tt = pd.read_csv('Medical_dataset/Testing.csv')

# Build symptom vectors and disease labels from training data
symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

# Full list of raw symptom column names (excluding prognosis)
all_symp_col = list(df_tr.columns[:-1])


def clean_symp(sym):
    """
    Normalize symptom string for display (replace underscores, variants, etc.).
    """
    return (
        sym.replace('_', ' ')
           .replace('.1', '')
           .replace('(typhos)', '')
           .replace('yellowish', 'yellow')
           .replace('yellowing', 'yellow')
    )


# Human-friendly symptom names (for display) built from columns
all_symp = [clean_symp(sym) for sym in all_symp_col]


def preprocess(doc):
    """
    Minimal text preprocessing:
    - tokenization via spaCy
    - remove stopwords & non-alphabetics
    - lemmatize
    Returns a space-joined string of lemmas.
    """
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


# Preprocessed symptoms (lemma-based) to use for matching
all_symp_pr = [preprocess(sym) for sym in all_symp]

# Map preprocessed symptom string -> original column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# =========================================
# I. Syntactic Similarity Helpers
# =========================================

def powerset(seq):
    """
    Yield all subsets of a list (powerset).
    E.g., [1,2,3] -> [[1,2,3],[1,2],[1,3],[2,3],[1],[2],[3],[]]
    (Ordering preserved to mirror original behavior.)
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


def sort(a):
    """
    Sort a list of lists by descending length (in-place),
    then pop the last element. Mirrors original behavior.
    """
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a


def permutations(s):
    """
    Return all permutations of list `s` as space-joined strings.
    """
    permutations_list = list(itertools.permutations(s))
    return [' '.join(permutation) for permutation in permutations_list]


def DoesExist(txt):
    """
    Check if any permutation of any subset of the input `txt` (string)
    exists in the preprocessed symptoms list `all_symp_pr`.
    Returns the matched preprocessed symptom string if found; otherwise False.
    """
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symp_pr:
                return sym
    return False


def jaccard_set(str1, str2):
    """
    Compute Jaccard similarity between two whitespace-tokenized strings.
    """
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def syntactic_similarity(symp_t, corpus):
    """
    Apply Jaccard similarity between `symp_t` and each item in `corpus`.
    - If an exact-like existence via DoesExist is found, return (1, [best_match]).
    - Else return (1, list_of_nonzero_similarity_candidates) if any,
      or (0, None) if none.
    """
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


def check_pattern(inp, dis_list):
    """
    Check if regex pattern `inp` (as raw) matches any item in `dis_list`.
    Returns (1, matches) if found else (0, None).
    """
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"  # kept, mirrors original (unused variable)
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


# =========================================
# II. Semantic Similarity Helpers
# =========================================

def WSD(word, context):
    """
    Word Sense Disambiguation using NLTK's Lesk algorithm for `word` in `context`.
    Returns a Synset or None.
    """
    sens = lesk(context, word)
    return sens


def semanticD(doc1, doc2):
    """
    Compute a semantic similarity score between two documents by:
    - preprocessing each,
    - checking pairwise synset similarity via Wu-Palmer,
    - summing scores > 0.25 threshold,
    - normalizing by token count product.
    """
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                # Alternative path similarity commented out in original
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


def semantic_similarity(symp_t, corpus):
    """
    Find the most semantically similar item in `corpus` to `symp_t`
    using `semanticD`. Returns (max_score, best_match or None).
    """
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


def suggest_syn(sym):
    """
    Suggest possible preprocessed symptom candidates for a given `sym`
    by:
    - collecting WordNet lemmas,
    - finding the most semantically similar preprocessed symptoms,
    - deduplicating results.
    """
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


# =========================================
# III. Vectorization / Data Helpers
# =========================================

def OHV(cl_sym, all_sym):
    """
    Build a One-Hot Vector (DataFrame) for the set of selected symptoms `cl_sym`
    against the full list `all_sym`.
    Column names are the human-friendly `all_symp` (kept as original).
    """
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    """
    True if every element in list `small` exists in list `big`.
    """
    a = True
    for i in small:
        if i not in big:
            a = False
    return a


def possible_diseases(l):
    """
    Given a list of symptom column names `l`, find diseases for which
    all those symptoms are present in at least one training row.
    """
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis


def symVONdisease(df, disease):
    """
    Return the list of symptom column names that occur for a given `disease`
    (any row where the symptom is 1) in dataframe `df`.
    """
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# =========================================
# IV. Model & Dictionaries
# =========================================

# Load KNN classifier model (kept as original path)
knn_clf = joblib.load('model/knn.pkl')

# Global dictionaries for severity, description, and precautions
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    """
    Load disease description text from CSV into `description_list`.
    """
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    """
    Load symptom severity scores from CSV into `severityDictionary`.
    """
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            # Keep original silent pass on any parsing issues
            pass


def getprecautionDict():
    """
    Load disease precaution steps (list of 4 items) from CSV into `precautionDictionary`.
    """
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


# Initialize dictionaries at import time (same as original)
getSeverityDict()
getprecautionDict()
getDescription()


def calc_condition(exp, days):
    """
    Compute patient condition severity by summing symptom severities in `exp`
    and scaling by number of days.
    - Returns 1 if threshold exceeded (advise doctor consultation),
      else 0 (advise precautions).
    """
    sum_val = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum_val = sum_val + severityDictionary[item]
    if ((sum_val * days) / (len(exp)) > 13):
        return 1
        # original code printed messages after return; preserved structure
    else:
        return 0


def related_sym(psym1):
    """
    Build a clarifying question listing candidate symptoms from `psym1`
    with numeric options for disambiguation. Returns 0 if none.
    """
    s = "could you be more specific, <br>"
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0


# =========================================
# Flask Routes (UI shell + conversational endpoint)
# =========================================

# app.py
from flask import Flask, render_template, redirect, url_for, flash, request
from forms import LoginForm, SignupForm  # <-- import forms hapa

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wekapasswordisiri-ngumu-na-ndefu-hapa'  # inahitajika na Flask-WTF (CSRF)



# NOTE: Secret key is set below for flash/session use (kept same net result).
app.secret_key = 'your-secret-key'  # needed for forms and flash messages (will be overwritten in __main__)


######################################################################





# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-very-simple')  # badilisha kwa production

# -------- SQLite helpers --------
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



def init_db():
    conn = db_connect()
    cur = conn.cursor()

    # 1) Base table (no email / role yet)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
            -- we'll add email / role below if missing
        )
    """)

    # 2) Find existing columns
    cur.execute("PRAGMA table_info(users)")
    existing_cols = {row[1] for row in cur.fetchall()}  # row[1] = column name

    # 3) Add email column if missing (NO UNIQUE here)
    if 'email' not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
        # refresh column list
        existing_cols.add('email')

    # 4) Add role column if missing (then backfill)
    if 'role' not in existing_cols:
        cur.execute("ALTER TABLE users ADD COLUMN role TEXT")
        cur.execute("UPDATE users SET role='patient' WHERE role IS NULL")
        existing_cols.add('role')

    # 5) Seed built-in admin (set email/role if columns exist)
    cur.execute("SELECT 1 FROM users WHERE username = ?", ("admin",))
    if cur.fetchone() is None:
        # If 'email' and 'role' exist, include them in insert
        if 'email' in existing_cols and 'role' in existing_cols:
            cur.execute(
                "INSERT INTO users(username, password, email, role) VALUES(?,?,?,?)",
                ("admin", "admin123", "admin@example.com", "admin")
            )
        elif 'role' in existing_cols:
            cur.execute(
                "INSERT INTO users(username, password, role) VALUES(?,?,?)",
                ("admin", "admin123", "admin")
            )
        else:
            cur.execute(
                "INSERT INTO users(username, password) VALUES(?,?)",
                ("admin", "admin123")
            )
            # If role column exists now, set admin role explicitly
            if 'role' in existing_cols:
                cur.execute("UPDATE users SET role='admin' WHERE username='admin'")
        # if email exists but we inserted without it, set later:
        if 'email' in existing_cols:
            cur.execute("UPDATE users SET email='admin@example.com' WHERE username='admin' AND (email IS NULL OR email='')")

    # 6) Create UNIQUE index on email (enforces uniqueness going forward)
    if 'email' in existing_cols:
        try:
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users(email)")
        except sqlite3.OperationalError as e:
            # Likely caused by duplicate existing emails
            print("[init_db] Could not create unique index on email:", e)
            print("         Check for duplicate email values in the users table.")

    conn.commit()
    conn.close()


init_db()

def get_user_by_username(username: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

def create_user(username: str, password: str):
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users(username, password) VALUES(?,?)", (username, password))
        conn.commit()
    finally:
        conn.close()

def verify_password(stored: str, given: str) -> bool:
    # Plaintext for simplicity (for production, use hashing)
    return stored == given

def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return wrapper


#################################################






@app.route('/', endpoint='home')
def home():
    return render_template('index.html')


#
from werkzeug.security import generate_password_hash



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = (form.username.data or '').strip()
        email    = (form.email.data or '').strip().lower()
        password = form.password.data
        role     = 'patient'

        conn = db_connect()
        cur  = conn.cursor()

        # Check duplicates for username or email
        cur.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (username, email))
        if cur.fetchone():
            conn.close()
            flash('Username or email already exists.', 'danger')
            return render_template('signup.html', form=form)

        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?,?,?,?)",
            (username, email, password, role)
        )
        conn.commit()
        conn.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)



# ---------- LOGIN ----------
from werkzeug.security import check_password_hash

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()  # or remove this line if you're using a plain HTML form

    # If you use Flask-WTF on the template, keep `form.validate_on_submit()`.
    # If your template is a plain HTML form (no {{ form.hidden_tag() }}), use: if request.method == 'POST':
    if request.method == 'POST' and (not hasattr(form, 'validate_on_submit') or form.validate_on_submit()):
        # Read fields from form or request.form safely
        username_or_email = (getattr(form, 'username', None).data if hasattr(form, 'username') else request.form.get('username') or '').strip()
        password = (getattr(form, 'password', None).data if hasattr(form, 'password') else request.form.get('password') or '')

        # --- Lookup user by username OR email (case-insensitive) ---
        conn = db_connect()
        cur  = conn.cursor()
        # COLLATE NOCASE makes it case-insensitive
        cur.execute("""
            SELECT username, email, password, role
            FROM users
            WHERE username = ? COLLATE NOCASE
               OR email = ? COLLATE NOCASE
            LIMIT 1
        """, (username_or_email, username_or_email))
        user = cur.fetchone()
        conn.close()

        if not user:
            # User not found -> send to signup
            flash('Account not found. Please sign up.', 'info')
            return redirect(url_for('signup'))

        # Plaintext comparison (since that’s what you used at signup)
        # If you switch to hashing, replace with check_password_hash(...)
        if user['password'] != password:
            flash('Wrong password. Try again or sign up.', 'danger')
            # You can redirect back to login, or to signup as you prefer:
            return redirect(url_for('login'))

        # Success: set session and redirect by role
        session['user'] = user['username']
        session['role'] = user['role'] or 'patient'

        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))

    # GET, or invalid form submit (e.g., missing CSRF when using Flask-WTF)
    return render_template('login.html', form=form)



@app.route('/trained_bot')
@login_required
def trained_bot():
    return render_template('home.html', username=session.get('user'))




@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))


from flask import abort


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if session.get('role') != 'admin':
        abort(403)

    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT username, email, role, id FROM users")
    users = cur.fetchall()

    cur.execute("SELECT COUNT(*) AS total_users FROM users")
    total_users = cur.fetchone()['total_users']

    cur.execute("SELECT COUNT(*) AS total_predictions FROM predictions")
    total_preds = cur.fetchone()['total_predictions']

    cur.execute("SELECT * FROM faq")  # assuming table: faq (id, question, answer)
    faq = cur.fetchall()

    cur.execute("SELECT * FROM users WHERE username=?", (session['user'],))
    admin_user = cur.fetchone()

    conn.close()

    return render_template('admin_dashboard.html',
                           user=admin_user,
                           user_stats={'total_users': total_users, 'total_predictions': total_preds},
                           users=users,
                           faq=faq)



@app.route('/admin/user/<username>')
@login_required
def view_user_profile(username):
    if session.get('role') != 'admin':
        abort(403)
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cur.fetchone()
    conn.close()
    return render_template('view_user.html', user=user)


@app.route('/admin/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    if session.get('role') != 'admin':
        abort(403)
    conn = db_connect(); cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit(); conn.close()
    flash("User deleted successfully")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/update_profile', methods=['POST'])
@login_required
def update_admin_profile():
    if session.get('role') != 'admin':
        abort(403)
    email = request.form.get('email')
    password = request.form.get('password')
    conn = db_connect(); cur = conn.cursor()
    if password:
        hashed = generate_password_hash(password)
        cur.execute("UPDATE users SET email=?, password=? WHERE username=?",
                    (email, hashed, session['user']))
    else:
        cur.execute("UPDATE users SET email=? WHERE username=?",
                    (email, session['user']))
    conn.commit(); conn.close()
    flash("Profile updated")
    return redirect(url_for('admin_dashboard'))



@app.route('/admin/promote/<int:user_id>', methods=['POST'])
@login_required
def promote_user(user_id):
    if session.get('role') != 'admin':
        abort(403)

    conn = db_connect(); cur = conn.cursor()
    cur.execute("UPDATE users SET role='admin' WHERE id=?", (user_id,))
    conn.commit(); conn.close()

    flash("User promoted to admin.", "success")
    return redirect(url_for('admin_dashboard'))



# Patient dashboard view (populate stats, history, email)
@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    if session.get('role') != 'patient':
        abort(403)
    username = session.get('user')
    email = None
    predictions = []
    stats = dict(total_preds=0, last_pred_date=None)

    conn = db_connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT email FROM users WHERE username=?", (username,))
        r = cur.fetchone()
        if r: email = r['email']
    except Exception: pass

    try:
        cur.execute("""
          SELECT symptoms, predicted_disease, confidence, created_at
          FROM predictions
          WHERE user_id = (SELECT id FROM users WHERE username=?)
          ORDER BY created_at DESC LIMIT 25
        """, (username,))
        predictions = cur.fetchall()
        stats['total_preds'] = len(predictions)
        stats['last_pred_date'] = predictions[0]['created_at'] if predictions else None
    except Exception:
        predictions = []
    conn.close()

    return render_template('patient_dashboard.html',
                           username=username,
                           email=email,
                           predictions=predictions,
                           stats=stats)

# Update email
@app.route('/patient/profile/update', methods=['POST'])


@login_required
def patient_update_profile():
    if session.get('role') != 'patient':
        abort(403)
    email = (request.form.get('email') or '').strip().lower()
    if not email:
        flash('Email is required.', 'danger')
        return redirect(url_for('patient_dashboard'))
    conn = db_connect(); cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET email=? WHERE username=?", (email, session.get('user')))
        conn.commit()
        flash('Email updated.', 'success')
    except Exception:
        flash('Could not update email.', 'danger')
    finally:
        conn.close()
    return redirect(url_for('patient_dashboard'))

# Change password (plaintext version — for dev only)
@app.route('/patient/profile/password', methods=['POST'])
@login_required
def patient_change_password():
    if session.get('role') != 'patient':
        abort(403)
    curr = request.form.get('current_password') or ''
    newp = request.form.get('new_password') or ''
    if not newp:
        flash('New password required.', 'danger')
        return redirect(url_for('patient_dashboard'))

    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username=?", (session.get('user'),))
    row = cur.fetchone()
    if not row or row['password'] != curr:
        conn.close()
        flash('Current password incorrect.', 'danger')
        return redirect(url_for('patient_dashboard'))

    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET password=? WHERE username=?", (newp, session.get('user')))
        conn.commit()
        flash('Password changed.', 'success')
    except Exception:
        flash('Could not change password.', 'danger')
    finally:
        conn.close()
    return redirect(url_for('patient_dashboard'))



# ---------- STATIC PAGES ----------
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if session.get('role') != 'patient':
        abort(403)
    
    username = session.get('user')
    email = request.form.get('email')
    password = request.form.get('password')

    conn = db_connect(); cur = conn.cursor()
    try:
        if password:
            hashed = generate_password_hash(password)
            cur.execute("UPDATE users SET email=?, password=? WHERE username=?", (email, hashed, username))
        else:
            cur.execute("UPDATE users SET email=? WHERE username=?", (email, username))
        conn.commit()
    except Exception as e:
        flash("Error updating profile.")
    finally:
        conn.close()

    return redirect(url_for('patient_dashboard'))



#####################################################


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # If JSON body:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            symptoms = (data.get('symptoms') or '').strip()
        else:
            symptoms = (request.form.get('symptoms') or '').strip()

        # TODO: run your real model here
        prediction = "Flu" if "fever" in symptoms.lower() else "Common Cold"
        confidence = 92

        # (optional) log to predictions table tied to current user
        try:
            if 'user' in session:
                conn = db_connect(); cur = conn.cursor()
                cur.execute("""
                  INSERT INTO predictions(user_id, symptoms, predicted_disease, confidence, created_at)
                  VALUES((SELECT id FROM users WHERE username=?), ?, ?, ?, datetime('now'))
                """, (session.get('user'), symptoms, prediction, confidence))
                conn.commit(); conn.close()
        except Exception:
            pass

        return {"prediction": prediction, "confidence": confidence}

    # GET -> render a plain page or redirect to dashboard/chatbot tab
    return render_template('predict.html')


####################################################################
@app.route("/get")
def get_bot_response():
    """
    Conversational endpoint for chatbot:
    - Drives a multi-step dialog via the Flask session,
    - Performs syntactic/semantic matching of user-described symptoms,
    - Proposes clarifying questions,
    - Builds a symptom set and predicts disease with a KNN model,
    - Provides description and precautions based on CSV dictionaries.
    The flow mirrors the original logic exactly.
    """
    s = request.args.get('msg')

    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s == "q":
                "Thank you for using ower web site Mr/Ms " + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender

    if s and s.upper() == "OK":
        return "What is your name ?"

    if 'name' not in session and 'step' not in session:
        session['name'] = s
        session['step'] = "age"
        return "How old are you? "

    if session["step"] == "age":
        session["age"] = int(s)
        session["step"] = "gender"
        return "Can you specify your gender ?"

    if session["step"] == "gender":
        session["gender"] = s
        session["step"] = "Depart"

    if session['step'] == "Depart":
        session['step'] = "BFS"
        return ("Well, Hello again Mr/Ms " + session["name"] +
                ", now I will be asking some few questions about your symptoms to see what you should do. "
                "Tap S to start diagnostic!")

    if session['step'] == "BFS":
        session['step'] = "FS"  # first symptom
        return "Can you precise your main symptom Mr/Ms " + session["name"] + " ?"

    if session['step'] == "FS":
        sym1 = s
        sym1 = preprocess(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        temp = [sym1, sim1, psym1]
        session['FSY'] = temp  # first symptom info
        session['step'] = "SS"  # second symptom
        if sim1 == 1:
            session['step'] = "RS1"  # related_sym1 disambiguation
            s = related_sym(psym1)
            if s != 0:
                return s
        else:
            return "You are probably facing another symptom, if so, can you specify it?"

    if session['step'] == "RS1":
        temp = session['FSY']
        psym1 = temp[2]
        psym1 = psym1[int(s)]
        temp[2] = psym1
        session['FSY'] = temp
        session['step'] = 'SS'
        return "You are probably facing another symptom, if so, can you specify it?"

    if session['step'] == "SS":
        sym2 = s
        sym2 = preprocess(sym2)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2]
        session['SSY'] = temp  # second symptom info
        session['step'] = "semantic"  # evaluate semantics
        if sim2 == 1:
            session['step'] = "RS2"  # related_sym2 disambiguation
            s = related_sym(psym2)
            if s != 0:
                return s

    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"

    if session['step'] == "semantic":
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'  # move to possible diseases

    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = [sym1, sim1, psym1]
            session['FSY'] = temp
            session['step'] = "sim1=0"  # suggest synonyms for symptom 1
        else:
            session['step'] = "BFsim2=0"

    if session['step'] == "sim1=0":  # semantic no => suggestions
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s == "yes":
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"

    if session['step'] == "BFsim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = [sym2, sim2, psym2]
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"

    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s == "yes":
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"  # test fallback when both fail

    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            # both unknown -> go to END without prediction
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'  # proceed to possible diseases

    if session['step'] == 'PD':
        # Build patient symptom list and compute candidate diseases
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
        session["diseases"] = possible_diseases(session["all"])
        all_sym = session["all"]
        diseases = session["diseases"]
        dis = diseases[0]
        session["dis"] = dis
        session['step'] = "for_dis"

    if session['step'] == "DIS":
        # Ask about symptoms belonging to the current candidate disease
        if "symv" in session:
            if len(s) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if s == "yes":
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])
        if len(session["symv"]) > 0:
            # NOTE: `symts` is used as in original structure
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                symts = session["symv"]
                msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                s = ""
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session["diseases"]
            if diseases[0] in PD:
                session["testpred"] = diseases[0]
                PD.remove(diseases[0])
            session["diseases"] = PD
            session['step'] = "for_dis"

    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:
            session['step'] = 'PREDICT'
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()  # loop through symptoms of disease

    if session['step'] == "PREDICT":
        result = knn_clf.predict(OHV(session["all"], all_symp_col))
        session['step'] = "END"

    if session['step'] == "END":
        if result is not None:
            if result[0] != session["testpred"]:
                session['step'] = "Q_C"
                return ("as you provide me with few symptoms, I am sorry to announce that I cannot predict your "
                        "disease for the moment!!! <br> Can you specify more about what you are feeling or Tap q to "
                        "stop the conversation ")
            session['step'] = "Description"
            session["disease"] = result[0]
            return ("Well Mr/Ms " + session["name"] + ", you may have " + result[0] +
                    ". Tap D to get a description of the disease .")
        else:
            session['step'] = "Q_C"
            return "can you specify more what you feel or Tap q to stop the conversation"

    if session['step'] == "Description":
        y = {
            "Name": session["name"],
            "Age": session["age"],
            "Gender": session["gender"],
            "Disease": session["disease"],
            "Sympts": session["all"]
        }
        write_json(y)
        session['step'] = "Severity"
        if session["disease"] in description_list.keys():
            return description_list[session["disease"]] + " \n <br>  How many days have you had symptoms?"
        else:
            if " " in session["disease"]:
                session["disease"] = session["disease"].replace(" ", "_")
            return ("please visit <a href='" +
                    "https://en.wikipedia.org/wiki/" + session["disease"] + "'>  here  </a>")

    if session['step'] == "Severity":
        session['step'] = 'FINAL'
        if calc_condition(session["all"], int(s)) == 1:
            return "you should take the consultation from doctor <br> Tap q to exit"
        else:
            msg = 'Nothing to worry about, but you should take the following precautions :<br> '
            i = 1
            for e in precautionDictionary[session["disease"]]:
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += ' Tap q to end'
            return msg

    if session['step'] == "FINAL":
        session['step'] = "BYE"
        return "Your diagnosis was perfectly completed. Do you need another medical consultation (yes or no)? "

    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if s and s.lower() == "yes":
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again Mr/Ms " + session["name"] + " Please tell me your main symptom. "
        else:
            return ("THANKS Mr/Ms " + name +
                    " for using me for more information please contact <b> +21266666666</b>")


# =========================================
# Entrypoint
# =========================================
if __name__ == "__main__":
    # Preserve original behavior: overwrite earlier secret key with random key
    import random
    import string

    S = 10  # number of characters in the string.
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    app.secret_key = str(ran)

    # Run the Flask app (debug disabled to mirror the last run block behavior)
    # Your original file had two run blocks; we keep a single final one.
    app.run()