from database_models.database import db

# this table stores the accuracy results for all the models.
class AccuracyScore(db.Model):
    __tablename__ = "accuracy_score"
    accuracy_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    train_accuracy = db.Column(db.String(50), nullable=False)
    test_accuracy = db.Column(db.String(50), nullable=False)
    train_precision = db.Column(db.String(50), nullable=False)
    test_precision = db.Column(db.String(50), nullable=False)
    train_recall = db.Column(db.String(50), nullable=False)
    test_recall = db.Column(db.String(50), nullable=False)
    disease_type = db.Column(db.String(100), nullable=True)
    is_current_model = db.Column(db.Boolean, default=False)
