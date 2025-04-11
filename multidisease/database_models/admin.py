from database_models.database import db
from werkzeug.security import generate_password_hash, check_password_hash

class AdminUser(db.Model):
    admin_id = db.Column(db.Integer, primary_key=True)
    admin_username = db.Column(db.String(25), unique=True, nullable=False)
    admin_password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.admin_password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.admin_password_hash, password)
