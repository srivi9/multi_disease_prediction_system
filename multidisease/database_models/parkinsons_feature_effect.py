from database_models.database import db
from werkzeug.security import generate_password_hash, check_password_hash

class ParkinsonSymptomContribution(db.Model):
    __tablename__ = "parkinson_symptom_contribution"
    effect_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.String(50), nullable=False)
    feature = db.Column(db.String(50), nullable=True)
    effect = db.Column(db.String(50), nullable=True)
    contribution = db.Column(db.Float, nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)
    
    # Foreign key to ParkinsonPrediction
    prediction_id = db.Column(db.Integer, db.ForeignKey('parkinsons_prediction.Prediction_id'), nullable=False)

    # Relationship to HeartPrediction (many contributions for one prediction)
    parkinsons_prediction = db.relationship('ParkinsonsPrediction')