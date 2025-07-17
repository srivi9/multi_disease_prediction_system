from database_models.database import db 

#this table stores the feature explanations
class FeatureExplanation(db.Model):
    __tablename__ = 'feature_explanations'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    explanation_text = db.Column(db.Text, nullable=False)
    disease_type = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Explanation Patient {self.patient_id}>"
