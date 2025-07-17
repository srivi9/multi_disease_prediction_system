from database_models.database import db

#this table stores the patients' heart disease symptoms and the corresponding diagnosis result
class HeartSymptoms(db.Model):
    __tablename__ = "heart_symptoms"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientName = db.Column(db.String(50), nullable=False)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"), nullable=False)
    Age =db.Column(db.String(25), nullable=False)

    Sex =db.Column(db.Integer, nullable=False)

    Chest_Pain_Type=db.Column(db.Integer, nullable=False)

    Resting_Blood_Pressure=db.Column(db.String(25), nullable=False)

    Cholesterol=db.Column(db.String(25), nullable=False)

    Fasting_blood_sugar=db.Column(db.Integer, nullable=False)

    ECG_result=db.Column(db.Integer,  nullable=False)

    Heart_rate=db.Column(db.String(25),  nullable=False)

    Angina=db.Column(db.Integer,  nullable=False)

    Old_peak=db.Column(db.String(25), nullable=False)

    Slope=db.Column(db.Integer,  nullable=False)

    Number_of_major_vessels=db.Column(db.Integer, nullable=False)

    Thalassemia=db.Column(db.Integer, nullable=False)

    Target=db.Column(db.Integer, nullable=True)

    Prediction_id = db.Column(db.Integer, db.ForeignKey("heart_prediction.Prediction_id"), nullable=True)

    predictions = db.relationship('HeartPrediction', back_populates='heart_symptoms')