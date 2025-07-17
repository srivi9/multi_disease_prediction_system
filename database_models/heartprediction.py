from database_models.database import db

#this table stores the system predicted result for heart disease
class HeartPrediction(db.Model):
    __tablename__ = "heart_prediction"
    Prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"), nullable=False)
    PatientName = db.Column(db.String(50), nullable=False)
    disease_type = db.Column(db.String(100), nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)


    heart_symptoms = db.relationship('HeartSymptoms', back_populates='predictions')
    
    symptom_contributions = db.relationship('HeartSymptomContribution', back_populates='heart_prediction')