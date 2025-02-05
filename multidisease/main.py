from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash 
from flask_sqlalchemy import SQLAlchemy 
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///admindata.db"
app.config["SQLALCHEMY_BINDS"] = {
    "patients_db": "sqlite:///patientdata.db",  # Second database for PatientUsers
    "doctors_db": "sqlite:///doctordata.db"  # Third database for DoctorUsers
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
db = SQLAlchemy(app)

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Database Model for admin
class AdminUser(db.Model):
    admin_id = db.Column(db.Integer, primary_key=True)
    admin_username = db.Column(db.String(25), unique=True, nullable=False)
    admin_password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.admin_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.admin_password_hash, password)

# Database Model for patient
class PatientUser(db.Model):
    __bind_key__ = "patients_db"
    patient_id = db.Column(db.Integer, primary_key=True)
    patient_username = db.Column(db.String(25), unique=True, nullable=False)
    patient_email = db.Column(db.String(25), unique=True, nullable=False) 
    patient_password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.patient_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.patient_password_hash, password)

    def set_email(self, email):
        self.patient_email = email



# Database Model for doctor
class DoctorUser(db.Model):
    __bind_key__ = "doctors_db"
    doctor_id = db.Column(db.Integer, primary_key=True)
    doctor_username = db.Column(db.String(25), unique=True, nullable=False)
    doctor_password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.doctor_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.doctor_password_hash, password)


# Global variable to store the trained model
model = None

# Routes
# Routes
@app.route("/")
def home():
    # Check if the user is already logged in
    if "username" in session:
        return redirect(url_for('dashboard'))  # Redirect to dashboard if logged in
    return render_template("landingpage.html")  # Show landing page if not logged in

#adminloginpage
@app.route("/adminloginpage")
def admin_loginpage():
    return render_template("adminpage.html")

#patientloginpage
@app.route("/patientloginpage")
def patient_loginpage():
    return render_template("patientspage.html")

#patientregistrationpage
@app.route("/patientregistrationpage")
def patient_registration_page():
    return render_template("patient_registrationpage.html")


#doctorsloginpage
@app.route("/doctorloginpage")
def doctor_loginpage():
    return render_template("doctorspage.html")

@app.route("/heartdisease")
def heart_disease():
    return render_template("heartdisease.html")

# Login for admin
@app.route("/adminlogin", methods=["POST", "GET"])
def admin_login():
    username = request.form['username']
    password = request.form['password']
    user = AdminUser.query.filter_by(admin_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('admin_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("home"))

# Login for patient
@app.route("/patientlogin", methods=["POST", "GET"])
def patient_login():
    username = request.form['username']
    password = request.form['password']
    user = PatientUser.query.filter_by(patient_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('patient_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("home"))

# Login for doctor
@app.route("/doctorlogin", methods=["POST", "GET"])
def doctor_login():
    username = request.form['username']
    password = request.form['password']
    user = DoctorUser.query.filter_by(doctor_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('doctor_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("home"))


# Register for admin
@app.route("/adminregister", methods=["POST", "GET"])
def admin_register():
    username = request.form['username']
    password = request.form["password"]
    user = AdminUser.query.filter_by(admin_username=username).first()
    
    if user:
        flash("User already exists!", "warning")
        return redirect(url_for("home"))
    else:
        new_user = AdminUser(admin_username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('admin_dashboard'))
    

# Register for patient
@app.route("/patientregister", methods=["POST", "GET"])
def patient_register():
    username = request.form['username']
    password = request.form["password"]
    email = request.form["email"]
    user = PatientUser.query.filter_by(patient_username=username).first()
    
    if user:
        flash("User already exists!", "warning")
        return redirect(url_for("home"))
    else:
        new_user = PatientUser(patient_username=username)
        new_user.set_password(password)
        new_user.set_email(email)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('patient_dashboard'))


# Register
@app.route("/doctorregister", methods=["POST", "GET"])
def doctor_register():
    username = request.form['username']
    password = request.form["password"]
    user = DoctorUser.query.filter_by(doctor_username=username).first()
    
    if user:
        flash("User already exists!", "warning")
        return redirect(url_for("home"))
    else:
        new_user = DoctorUser(doctor_username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('doctor_dashboard'))

# Dashboard
@app.route("/admin_dashboard")
def admin_dashboard():
    if "username" in session:
        return render_template("admin_dashboard.html", username=session['username'])
    return redirect(url_for('home'))

# Dashboard
@app.route("/doctor_dashboard")
def doctor_dashboard():
    if "username" in session:
        return render_template("doctor_dashboard.html", username=session['username'])
    return redirect(url_for('home'))

# Dashboard
@app.route("/patient_dashboard")
def patient_dashboard():
    if "username" in session:
        return render_template("patient_dashboard.html", username=session['username'])
    return redirect(url_for('home'))

# Upload dataset and train the model
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    global model
    if "username" not in session:
        return redirect(url_for("home"))
    
    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("admin_dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("admin_dashboard"))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # Load dataset and train model
        try:
            heart_data = pd.read_csv(filepath)
            
            # Check if 'target' column exists
            if 'target' not in heart_data.columns:
                flash("Invalid dataset format. 'target' column is missing.", "danger")
                return redirect(url_for("dashboard"))
            
            X = heart_data.drop(columns='target', axis=1)
            Y = heart_data['target']
            
            # Split data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, Y_train)

            # Calculate accuracy
            train_accuracy = accuracy_score(model.predict(X_train), Y_train)
            test_accuracy = accuracy_score(model.predict(X_test), Y_test)

            flash(f"Model trained successfully! Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}", "success")
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")

    return redirect(url_for("admin_dashboard"))

# Predict heart disease
@app.route("/predict", methods=["POST"])
def predict():
    global model
    if "username" not in session:
        return redirect(url_for("home"))

    if model is None:
        flash("No trained model available. Upload a dataset first!", "warning")
        return redirect(url_for("admin_dashboard"))

    try:
        # Extract input features from form
        input_data = [float(request.form[f"feature{i}"]) for i in range(1, 14)]
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)

        result = "The person has heart disease." if prediction[0] == 1 else "The person does not have heart disease."

        return render_template("dashboard.html", username=session['username'], result=result)

    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

# Logout
@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

# Run the app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
