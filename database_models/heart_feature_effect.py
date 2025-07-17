from database_models.database import db


#this table stores the feature effect and contribution for the heart disease symptoms (Features)
class HeartSymptomContribution(db.Model):
    __tablename__ = "heart_symptom_contribution"
    effect_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.String(50), nullable=False)
    feature = db.Column(db.String(50), nullable=True)
    effect = db.Column(db.String(50), nullable=True)
    contribution = db.Column(db.Float, nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)
    
    # Foreign key to HeartPrediction
    prediction_id = db.Column(db.Integer, db.ForeignKey('heart_prediction.Prediction_id'), nullable=False)

    # Relationship to HeartPrediction (many contributions for one prediction)
    heart_prediction = db.relationship('HeartPrediction', back_populates='symptom_contributions')