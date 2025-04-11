from database_models.database import db
from werkzeug.security import generate_password_hash, check_password_hash

class DoctorUser(db.Model):
    __tablename__ = "doctors_db"
    doctor_id = db.Column(db.Integer, primary_key=True)
    doctor_firstname= db.Column(db.String(50), nullable=False)
    doctor_lastname = db.Column(db.String(50), nullable=False)
    doctor_speciality= db.Column(db.String(100), nullable=False)
    doctor_username = db.Column(db.String(25), unique=True, nullable=False)
    doctor_password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.doctor_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.doctor_password_hash, password)
    
    def set_firstname(self, firstname):
        self.doctor_firstname = firstname  

    def set_lastname(self, lastname):
        self.doctor_lastname = lastname
    
    def set_speciality(self, speciality):
        self.doctor_speciality = speciality
