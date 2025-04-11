from database_models.database import db
from werkzeug.security import generate_password_hash, check_password_hash

class PatientUser(db.Model):
    __tablename__ = "patient_user"
    patient_id = db.Column(db.Integer, primary_key=True)
    patient_FirstName = db.Column(db.String(50), nullable=False)
    patient_LastName = db.Column(db.String(50), nullable=False)
    patient_username = db.Column(db.String(25), unique=True,  nullable=False)
    patient_email = db.Column(db.String(100), unique=True, nullable=False) 
    patient_password_hash = db.Column(db.String(150), nullable=False)
    patient_age = db.Column(db.String(50), nullable=False)
    patient_gender = db.Column(db.String(25), nullable=False)

    disease_type = db.Column(db.String(100), nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)


    def set_password(self, password):
        self.patient_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.patient_password_hash, password)

    def set_email(self, email):
        self.patient_email = email

    def set_firstname(self, firstname):
        self.patient_FirstName = firstname  

    
    def set_lastname(self, lastname):
        self.patient_LastName = lastname
    
    def set_age(self, age):
        self.patient_age = age
    
    def set_gender(self, gender):
        self.patient_gender = gender
