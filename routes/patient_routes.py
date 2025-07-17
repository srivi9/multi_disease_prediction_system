from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from database_models.patient import PatientUser
from database_models.database import db
from database_models.heartprediction import HeartPrediction
from database_models.heart_symptoms import HeartSymptoms
from database_models.diabetes_expert_pred import DiabetesSymptoms
from database_models.diabetes_system_pred import DiabetesPrediction
from database_models.parkinsons_system_pred import ParkinsonsPrediction
from database_models.parkinsons_expert_pred import ParkinsonsSymptoms
from database_models.notification import Notification
import re 
from helpers import get_all_heart_data

#this function is to display the registration page to the patient
patient_routes = Blueprint('patient_routes', __name__)
@patient_routes.route("/patientregistrationpage")
def patientregistration_page():
    return render_template("patient_registrationpage.html")

#this function is to display the login page to the patient
@patient_routes.route("/patientloginpage")
def patient_loginpage():
    return render_template("patientspage.html")

#this function handles the functionality to let the patient login
@patient_routes.route("/patientlogin", methods=["POST", "GET"])
def patient_login():
    username = request.form['username']
    password = request.form['password']
    user = PatientUser.query.filter_by(patient_username=username).first()
    
    if user and user.check_password(password):
        session['username'] = username 
        session["patient_id"]= user.patient_id
        return redirect(url_for('patient_routes.patient_dashboard'))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("patient_routes.patient_loginpage"))


#this function handles the functionality to let the patient register
@patient_routes.route("/patientregister", methods=["POST", "GET"])
def patient_register():
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
    user = PatientUser.query.filter_by(patient_username=username).first()
    session["patient_id"]= user.patient_id
    

    return redirect(url_for('patient_routes.patient_dashboard'))


# this function displays the patients dashboard
@patient_routes.route("/patient_dashboard")
def patient_dashboard():
    if "patient_id" in session:
        patient = PatientUser.query.get(session["patient_id"])
        print(session["patient_id"])
        return render_template("patient_dashboard.html", patient_id=session["patient_id"], new_user=patient)
    return redirect(url_for('home'))

#this function displays the page that is processing the results
@patient_routes.route('/processedresults', endpoint="processedresults", methods=["POST"])
def processedresults():
       
    
     return render_template("processedresults.html")



#this function displays the notifications to the patients.
@patient_routes.route('/notification',methods=["POST","GET"])
def notification():
    patient_id = session["patient_id"]
    notifications = Notification.query.filter_by(patient_id=patient_id)
    return render_template('notifications.html', notifications=notifications)

#this function is for patient to view past submissions.
@patient_routes.route('/pastsubmissions')
def viewpastsubmissions():
    patient_id = session["patient_id"]
    print("PatientID",patient_id)
    submissions = get_all_heart_data(patient_id=patient_id)
    print("Submissions:", submissions) 
    return render_template('view_submissions.html',submissions=submissions)

@patient_routes.route('/editsubmission')
def editsubmission():
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    submission = HeartSymptoms.query.filter_by(id = prediction_id, PatientID = patient_id)
    print("pid",patient_id,"pid",prediction_id)
    # Render the edit form with the existing values
    return render_template('edit_submission.html',submission=submission)