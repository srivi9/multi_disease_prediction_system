from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from database_models.doctor import DoctorUser
from database_models.database import db
from database_models.heartprediction import HeartPrediction 
from database_models.diabetes_system_pred import DiabetesPrediction
from database_models.parkinsons_system_pred import ParkinsonsPrediction
from database_models.patient import PatientUser
import re

doctor_routes = Blueprint('doctor_routes', __name__)

@doctor_routes.route("/doctorloginpage")
def doctor_loginpage():
    return render_template("doctorspage.html")

@doctor_routes.route("/add_doctors")
def add_doctorpage():
    return render_template("add_doctors.html")

@doctor_routes.route("/doctorlogin", methods=["POST", "GET"])
def doctor_login():
    username = request.form['username']
    password = request.form['password']
    user = DoctorUser.query.filter_by(doctor_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username
        return redirect(url_for('doctor_routes.doctor_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("doctor_routes.doctor_loginpage"))


@doctor_routes.route("/doctorregister", methods=["POST", "GET"])
def doctor_register():
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

    return redirect(url_for('admin_routes.admin_dashboard'))


# Dashboard
@doctor_routes.route("/doctor_dashboard")
def doctor_dashboard():
    if "username" in session:
    
     number_of_patients = db.session.query(PatientUser).count()
     heart_disease_count = db.session.query(HeartPrediction).filter(HeartPrediction.prediction_result == 'positive').count()
     diabetes_disease_count = db.session.query(DiabetesPrediction).filter(DiabetesPrediction.prediction_result == 'positive').count()
     parkinsons_prediction_count = db.session.query(ParkinsonsPrediction).filter(DiabetesPrediction.prediction_result == 'positive').count()
     print("Negative:",heart_disease_count)
    return render_template("doctor_dashboard.html", username=session['username'], heart_disease_count=heart_disease_count,diabetes_disease_count=diabetes_disease_count,parkinsons_prediction_count=parkinsons_prediction_count,number_of_patients=number_of_patients)
    return redirect(url_for('home'))
