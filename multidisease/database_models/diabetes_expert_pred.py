from database_models.database import db


class DiabetesSymptoms(db.Model):
    __tablename__ = "diabetes_symptoms"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientName = db.Column(db.String(50), nullable=False)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"), nullable=False)
    Pregnancies = db.Column(db.String(50), nullable=False) 
    Glucose = db.Column(db.String(50), nullable=False) 
    Blood_pressure = db.Column(db.String(50), nullable=False) 
    Skin_thickness = db.Column(db.String(50), nullable=False)  
    Insulin_level = db.Column(db.String(50), nullable=False)  
    BMI = db.Column(db.String(50), nullable=False)  
    Diabetes_Pedigree = db.Column(db.String(50), nullable=False) 
    Age = db.Column(db.String(50), nullable=False)  
    target = db.Column(db.Integer, nullable=True)

    # Foreign key linking to DiabetesPrediction
    Prediction_id = db.Column(db.Integer, db.ForeignKey("diabetes_prediction.Prediction_id"), nullable=True)

    # Relationship with DiabetesPrediction
    diabetes_prediction = db.relationship('DiabetesPrediction', back_populates='diabetes_symptoms')
