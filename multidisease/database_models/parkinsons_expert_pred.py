from database_models.database import db


class ParkinsonsSymptoms(db.Model):
    __tablename__ = "parkinsons_symptoms"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    PatientName = db.Column(db.String(50), nullable=False)
    PatientID = db.Column(db.Integer, db.ForeignKey("patient_user.patient_id"), nullable=False)
    Age = db.Column(db.String(50), nullable=False)  
    Gender = db.Column(db.String(50), nullable=False) 
    BMI = db.Column(db.String(50), nullable=False) 
    PhysicalActivity = db.Column(db.String(50), nullable=False) 
    DietQuality = db.Column(db.String(50), nullable=False) 
    SleepQuality = db.Column(db.String(50), nullable=False) 
    FamilyHistoryParkinsons = db.Column(db.String(50), nullable=False) 
    TraumaticBrainInjury = db.Column(db.String(50), nullable=False) 
    Hypertension = db.Column(db.String(50), nullable=False) 
    Diabetes = db.Column(db.String(50), nullable=False) 
    Depression = db.Column(db.String(50), nullable=False) 
    Stroke = db.Column(db.String(50), nullable=False) 
    Tremor = db.Column(db.String(50), nullable=False) 
    Rigidity = db.Column(db.String(50), nullable=False) 
    SpeechProblems = db.Column(db.String(50), nullable=False) 
    SleepDisorders = db.Column(db.String(50), nullable=False) 
    Constipation = db.Column(db.String(50), nullable=False) 
    Diagnosis=db.Column(db.Integer, nullable=True)

    Prediction_id = db.Column(db.Integer, db.ForeignKey("parkinsons_prediction.Prediction_id"), nullable=True)

    predictions = db.relationship('ParkinsonsPrediction', back_populates='parkinsons_symptoms')