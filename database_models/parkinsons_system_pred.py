from database_models.database import db

#this table stores the symptom predicted result for the parkinsons disease patient.
class ParkinsonsPrediction(db.Model):
    __tablename__ = "parkinsons_prediction"
    Prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"),nullable=False,)
    PatientName = db.Column(db.String(50), nullable=False)
    disease_type = db.Column(db.String(100), nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)

    parkinsons_symptoms = db.relationship('ParkinsonsSymptoms', back_populates='predictions')
    
    symptom_contributions = db.relationship('ParkinsonSymptomContribution', back_populates='parkinsons_prediction')