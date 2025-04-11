from flask import Blueprint, render_template, request, redirect, session, url_for, flash, current_app
from database_models.admin import AdminUser
from database_models.database import db
from database_models.doctor import DoctorUser
from database_models.patient import PatientUser
from database_models.modelaccuracy import AccuracyScore
import re 

admin_routes = Blueprint('admin_routes', __name__)


@admin_routes.route("/adminloginpage",endpoint="admin_loginpage")
def admin_loginpage():
    return render_template("adminpage.html")

@admin_routes.route('/upload_datasets_page', endpoint="upload_dataset_page")
def upload_dataset_page():
     return render_template("uploaddatasets.html")

@admin_routes.route('/viewresults', endpoint="trainingresults")
def trainingresults():
     return render_template("trainingresults.html")

@admin_routes.route('/pagemodaccuracy',methods=["POST","GET"])
def pagemodaccuracy():
     return render_template("modelacc.html")

@admin_routes.route("/adminlogin", methods=["POST", "GET"])
def admin_login():
    username = request.form['username']
    password = request.form['password']
    user = AdminUser.query.filter_by(admin_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('admin_routes.admin_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("admin_routes.admin_loginpage"))

# Register for admin
@admin_routes.route("/adminregister", methods=["POST", "GET"])
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
        return redirect(url_for('admin_routes.admin_dashboard'))

# Dashboard
@admin_routes.route("/admin_dashboard")
def admin_dashboard():
    if "username" in session:
        return render_template("admin_dashboard.html", username=session['username'])
    return redirect(url_for('home'))

# add doctors
@admin_routes.route("/admin_addDoctors", methods=["POST", "GET"])
def add_doctors():
        return render_template("add_doctors.html")

# add patients
@admin_routes.route("/admin_addPatients", methods=["POST", "GET"])
def add_patients():
        return render_template("add_patients.html")

#view the model accuracy scores page 
# add doctors
@admin_routes.route("/modelacc", methods=["POST", "GET"])
def page_model_accuracy():
        return render_template("add_doctors.html")


@admin_routes.route("/add_doctor", methods=["POST", "GET"])
def add_doctor():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    speciality = request.form['speciality']
    username = request.form['username']
    password = request.form["password"]

    # Username validation: 3+ letters, 1 underscore, 1 number
    username_pattern = r'^(?=.*[a-zA-Z]{3,})(?=.*\d)(?=.*_)[a-zA-Z0-9_]+$'
    
    # Password validation: 8+ characters, at least one number
    password_pattern = r'^(?=.*\d).{8,}$'

    if not re.match(username_pattern, username):
        flash("Invalid username! It must contain at least 3 letters, an underscore, and a number.", "danger")
        return redirect(url_for("doctor_routes.add_doctorpage"))

    if not re.match(password_pattern, password):
        flash("Password must be at least 8 characters long and contain at least one number.", "danger")
        return redirect(url_for("doctor_routes.add_doctorpage"))

    # Check if user already exists
    user = DoctorUser.query.filter_by(doctor_username=username).first()
    
    if user:
        flash("User already exists!", "warning")
        return redirect(url_for("doctor_routes.doctor_loginpage"))

    # Create new user
    new_user = DoctorUser(doctor_username=username)
    new_user.set_password(password)
    new_user.set_firstname(firstname)
    new_user.set_lastname(lastname)
    new_user.set_speciality(speciality)

    db.session.add(new_user)
    db.session.commit()
    session['username'] = username

    return redirect(url_for('admin_routes.add_doctors'))


@admin_routes.route("/add_patient", methods=["POST", "GET"])
def add_patient():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    age = request.form['age']
    username = request.form['username']
    password = request.form["password"]
    email = request.form["email"]
    gender = request.form["gender"]
    user = PatientUser.query.filter_by(patient_username=username).first()

    # Username validation: 3+ letters, 1 underscore, 1 number
    username_pattern = r'^(?=.*[a-zA-Z]{3,})(?=.*\d)(?=.*_)[a-zA-Z0-9_]+$'
    
    # Password validation: 8+ characters, at least one number
    password_pattern = r'^(?=.*\d).{8,}$'

    if not re.match(username_pattern, username):
        flash("Invalid username! It must contain at least 3 letters, an underscore, and a number.", "danger")
        return redirect(url_for("patient_routes.patientregistration_page"))

    if not re.match(password_pattern, password):
        flash("Password must be at least 8 characters long and contain at least one number.", "danger")
        return redirect(url_for("patient_routes.patientregistration_page"))

    # Check if user already exists
    user = PatientUser.query.filter_by(patient_username=username).first()
    
    if user:
        flash("User already exists!", "warning")
        return redirect(url_for("patient_routes.patientregistration_page"))

    # Create new user
    new_user = PatientUser(patient_username=username)
    new_user.set_firstname(firstname)
    new_user.set_lastname(lastname)
    new_user.set_password(password)
    new_user.set_email(email)
    new_user.set_age(age)
    new_user.set_gender(gender)

    db.session.add(new_user)
    db.session.commit()
    session['username'] = username

    return redirect(url_for('admin_routes.add_patients'))

@admin_routes.route('/heart_accuracy')
def heart_accuracy():
    all_scores = AccuracyScore.query.filter_by(disease_type="Heart Disease").all()
    return render_template('model_accuracy.html', scores=all_scores)

@admin_routes.route('/diabetes_accuracy')
def diabetes_accuracy():
    all_scores = AccuracyScore.query.filter_by(disease_type="Diabetes").all()
    return render_template('model_accuracy.html', scores=all_scores)

@admin_routes.route('/parkinsons_accuracy')
def parkinsons_accuracy():
    all_scores = AccuracyScore.query.filter_by(disease_type="Parkinsons").all()
    return render_template('model_accuracy.html', scores=all_scores)