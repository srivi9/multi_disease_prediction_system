from database_models.database import db

#this table stores the system predicted result for diabetes
class DiabetesPrediction(db.Model):
    __tablename__ = "diabetes_prediction"
    Prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"), nullable=False)
    PatientName = db.Column(db.String(50), nullable=False)
    disease_type = db.Column(db.String(100), nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)

    # Relationship with DiabetesSymptoms (One-to-Many)
    diabetes_symptoms = db.relationship('DiabetesSymptoms', back_populates='diabetes_prediction')

    # Relationship with DiabetesFeatureEffectContribution (One-to-Many)
    feature_contributions = db.relationship('DiabetesFeatureEffectContribution', back_populates='diabetes_prediction')
