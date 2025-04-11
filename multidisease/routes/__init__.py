from flask import Flask
from .admin_routes import admin_routes  # Import the Blueprint
from .doctor_routes import doctor_routes
from .patient_routes import patient_routes
def create_app():
    app = Flask(__name__)
    
    app.register_blueprint(admin_routes)  # Register Blueprint properly
    app.register_blueprint(doctor_routes)
    app.register_blueprint(patient_routes)

    return app
