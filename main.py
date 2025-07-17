from flask import Flask, render_template, redirect, session, url_for
import os
from routes.admin_routes import admin_routes
from routes.patient_routes import patient_routes
from routes.doctor_routes import doctor_routes
from routes.heart_routes import heart_routes
from routes.diabetes_routes import diabetes_routes
from routes.parkinsons_routes import parkinsons_routes
from database_models.database import db

app = Flask(__name__)
app.secret_key = "your_secret_key"
model = None
diabetes_model=None
heart_model=None
parkinsons_model = None

# Configure SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///multi_disease_system.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"

db.init_app(app) 
# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Register Blueprints
app.register_blueprint(admin_routes)
app.register_blueprint(patient_routes)
app.register_blueprint(doctor_routes)
app.register_blueprint(heart_routes)
app.register_blueprint(diabetes_routes)
app.register_blueprint(parkinsons_routes)

#this function returns the landing page
@app.route("/")
def home():
    return render_template("landingpage.html")

#this function returns the heart disease prediction form page
@app.route("/heartdisease")
def heart_disease():
    return render_template("heartdisease.html")

#this function returns the diabetes disease prediction form page
@app.route("/diabetesdisease")
def diabetes_disease():
    return render_template("diabetesdisease.html")

#this function returns the parkinsons disease prediction form page
@app.route("/parkinsonsdisease")
def parkinsons_disease():
    return render_template("parkinsonsdisease.html")

#this function returns the training page.
@app.route("/trainingloading")
def training_loading():
    return render_template("training_loading.html")

#this is the logout function
@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

#this is the main method
if __name__ == "__main__":
    with app.app_context():
        # Create all tables in the default database
        db.create_all()  
    app.run(debug=True)

