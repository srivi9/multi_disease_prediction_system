from database_models.database import db

#this creates the table to store the features, contributions and effect of the feature for diabetes
class DiabetesFeatureEffectContribution(db.Model):
    __tablename__ = "diabetes_feature_effect_contribution"
    effect_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientID = db.Column(db.String(50), nullable=False)
    feature = db.Column(db.String(50), nullable=True)
    effect = db.Column(db.String(50), nullable=True)
    contribution = db.Column(db.Float, nullable=True)
    prediction_result = db.Column(db.String(100), nullable=True)

    # Foreign key linking to DiabetesPrediction
    prediction_id = db.Column(db.Integer, db.ForeignKey('diabetes_prediction.Prediction_id'), nullable=False)

    # Relationship with DiabetesPrediction (many contributions for one prediction)
    diabetes_prediction = db.relationship('DiabetesPrediction', back_populates='feature_contributions')
