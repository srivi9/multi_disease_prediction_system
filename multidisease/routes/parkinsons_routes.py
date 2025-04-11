from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from database_models.database import db
from flask import Flask, render_template, request, redirect, session, url_for, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from routes.admin_routes import admin_routes
from routes.patient_routes import patient_routes
from routes.doctor_routes import doctor_routes
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np 
import joblib
import json
from database_models.modelaccuracy import AccuracyScore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score
from database_models.database import db
from database_models.patient import PatientUser
from database_models.parkinsons_expert_pred import ParkinsonsSymptoms
from database_models.parkinsons_system_pred import ParkinsonsPrediction
from database_models.parkinsons_feature_effect import ParkinsonSymptomContribution
from database_models.notification import Notification
uploaded_parkinsons_data = None
selected_features = []
app = Flask(__name__) 
app.config["UPLOAD_FOLDER"] = "uploads"
app.debug = False
MODEL_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "parkinsons_model.pkl")

def load_model():
    global parkinsons_model
    if os.path.exists(MODEL_PATH):
        parkinsons_model = joblib.load(MODEL_PATH)

def save_model(model):
    joblib.dump(model, MODEL_PATH)

parkinsons_routes = Blueprint('parkinsons_routes', __name__)
parkinsons_model = None


@parkinsons_routes.route("/park_lasso", methods=["POST"])
def park_lasso():
    global uploaded_parkinsons_data,parkinsons_model,selected_features
   
    # Try to load from pickle if global variable is None
    if uploaded_parkinsons_data is None:
            flash("No dataset uploaded. Please upload a dataset first.", "danger")
            return redirect(url_for("admin_routes.admin_dashboard"))
        
    try:
        
        
        # Start with all available features except diagnosis
        all_features = [col for col in uploaded_parkinsons_data.columns if col != "Diagnosis"]
        
        # Split data into features (X) and target labels (Y)
        X = uploaded_parkinsons_data[all_features]
        Y = uploaded_parkinsons_data["Diagnosis"]
        
        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        
        # Get regularization strength from form if provided, otherwise use default
        alpha_value = float(request.form.get("alpha_value", 0.01))
        
        # Initialize and train LASSO model using all features
        parkinsons_model = LogisticRegression(penalty='l1', solver='liblinear', C=alpha_value, max_iter=10000)
        parkinsons_model.fit(X_train, Y_train)
        
        # Extract feature importance values
        feature_importance = parkinsons_model.coef_[0]
        
        # Create importance dictionary for all features
        importance_dict = {all_features[i]: feature_importance[i] for i in range(len(all_features))}
        
        # Sort by absolute importance
        sorted_importance = {k: v for k, v in sorted(importance_dict.items(), 
                                                     key=lambda item: abs(item[1]), 
                                                     reverse=True)}
        
        # Identify features that LASSO considered important (non-zero coefficients)
        selected_features = [feature for feature, importance in importance_dict.items() 
                             if abs(importance) > 0]
        
        # If no features were selected (rare but possible with very high regularization)
        if not selected_features:
            flash("LASSO eliminated all features. Try decreasing regularization strength.", "warning")
            return redirect(url_for("admin_routes.admin_dashboard"))
        
         # STEP 2: Train a NEW model on only selected features
        X_train_selected = X_train[selected_features]  # Use only selected features
        X_test_selected = X_test[selected_features]    # Use only selected features
         
        parkinsons_model = LogisticRegression(penalty='l1', solver='liblinear', C=alpha_value, max_iter=5000)
        parkinsons_model.fit(X_train_selected, Y_train)
        
        selected_features_with_target = selected_features.copy()
        selected_features_with_target.append("Diagnosis")
        with open("selected_features_parkinsons.json", "w") as f:
            json.dump(selected_features_with_target, f)
        joblib.dump(parkinsons_model, "uploads/parkinsons_model.pkl")
        
        train_accuracy = f"{accuracy_score(Y_train, parkinsons_model.predict(X_train_selected))*100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, parkinsons_model.predict(X_test_selected))*100:.2f}%"
        train_precision = f"{precision_score(Y_train, parkinsons_model.predict(X_train_selected))*100:.2f}%"
        test_precision = f"{precision_score(Y_test, parkinsons_model.predict(X_test_selected))*100:.2f}%"
        train_recall = f"{recall_score(Y_train, parkinsons_model.predict(X_train_selected))*100:.2f}%"
        test_recall = f"{recall_score(Y_test, parkinsons_model.predict(X_test_selected))*100:.2f}%"
        

        disease_type = "Parkinsons"

        # Get the current model for this specific disease type
        previous_current = AccuracyScore.query.filter_by(is_current_model=True, disease_type=disease_type).first()

        if previous_current:
           previous_current.is_current_model = False
        db.session.commit()
       
        accuracy_entry = AccuracyScore(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_precision=train_precision,
            test_precision=test_precision,
            train_recall=train_recall,
            test_recall=test_recall,
            disease_type=disease_type,
            is_current_model=True
            )

# Add to the database
        db.session.add(accuracy_entry)
        db.session.commit()
        # Send each metric as a separate message
        flash(f"Model trained successfully with LASSO feature selection!")
        flash(f"LASSO selected {len(selected_features)} features out of {len(all_features)} available features")
        flash(f"Train Accuracy: {train_accuracy}")
        flash(f"Test Accuracy: {test_accuracy}")
        flash(f"Train Precision: {train_precision}")
        flash(f"Test Precision: {test_precision}")
        flash(f"Train Recall: {train_recall}")
        flash(f"Test Recall: {test_recall}")
        
        # Display selected features and their importance
        flash(f"Selected features by importance:")
        for feature, importance in sorted_importance.items():
            if abs(importance) > 0:  # Only show features with non-zero coefficients
                flash(f"{feature}: {importance:.4f}")
        
    except Exception as e:
        import traceback
        print(f"Error during training: {str(e)}")
        print(traceback.format_exc())
        flash(f"Error during training: {str(e)}", "danger")
        flash(f"Traceback: {traceback.format_exc()}", "danger")
    
    return redirect(url_for("training_loading"))

@parkinsons_routes.route("/train_parkinsons_model", methods=["POST"])
def train_model():
    global uploaded_parkinsons_data,parkinsons_model, selected_features

    if uploaded_parkinsons_data is None:
        flash("No dataset uploaded", "danger")
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Get selected features from form input
    selected_features = request.form.getlist("features")

    if not selected_features:
        flash("No features selected for training", "danger")
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Ensure 'target' is not selected as a feature
    selected_features = [f for f in selected_features if f != "Diagnosis"]
    selected_features.append("Diagnosis")

    try:
        # Filter dataset with selected features
        dataset = uploaded_parkinsons_data[selected_features]
        with open("selected_features_parkinsons.json", "w") as f:
            json.dump(selected_features, f)

        # Split data into features (X) and target labels (Y)
        X = dataset.drop(columns="Diagnosis", axis=1)
        Y = dataset["Diagnosis"]

        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Train the model 
        parkinsons_model = LogisticRegression(max_iter=90000000000000000000000000000000000000000000000000000)
        parkinsons_model.fit(X_train, Y_train)

        # Save the model
        joblib.dump(parkinsons_model, "uploads/parkinsons_model.pkl")

        # Calculate accuracy
        train_accuracy = f"{accuracy_score(Y_train, parkinsons_model.predict(X_train))*100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, parkinsons_model.predict(X_test))*100:.2f}%"

        # Calculate precision and recall
        train_precision = f"{precision_score(Y_train, parkinsons_model.predict(X_train))*100:.2f}%"
        test_precision = f"{precision_score(Y_test, parkinsons_model.predict(X_test))*100:.2f}%"
        train_recall = f"{recall_score(Y_train, parkinsons_model.predict(X_train))*100:.2f}%"
        test_recall = f"{recall_score(Y_test, parkinsons_model.predict(X_test))*100:.2f}%"
        
        disease_type = "Parkinsons"

        # Get the current model for this specific disease type
        previous_current = AccuracyScore.query.filter_by(is_current_model=True, disease_Type=disease_type).first()

        if previous_current:
           previous_current.is_current_model = False
        db.session.commit()
       
        accuracy_entry = AccuracyScore(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            train_precision=train_precision,
            test_precision=test_precision,
            train_recall=train_recall,
            test_recall=test_recall,
            disease_type=disease_type,
            is_current_model=True
            )

# Add to the database
        db.session.add(accuracy_entry)
        db.session.commit()

        # Send each metric as a separate message
        flash(f"Model trained successfully!")
        flash(f"Train Accuracy: {train_accuracy}")
        flash(f"Test Accuracy: {test_accuracy}")
        flash(f"Train Precision: {train_precision}")
        flash(f"Test Precision: {test_precision}")
        flash(f"Train Recall: {train_recall}")
        flash(f"Test Recall: {test_recall}")


    except Exception as e:
        flash(f"Error during training: {str(e)}", "danger")

    return redirect(url_for("training_loading"))

@parkinsons_routes.route("/parkinsonsdisease")
def parkinsons_disease():
    return render_template("parkinsonsdisease.html")

@parkinsons_routes.route("/upload_parkinsons_dataset", methods=["POST"])
def upload_parkinsons_dataset():
    global uploaded_parkinsons_data  # Store dataset globally

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"})

    try:
        # Secure filename and save it temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Read the dataset
        uploaded_parkinsons_data = pd.read_csv(filepath)

        # Ensure it has a 'target' column
        if "Diagnosis" not in uploaded_parkinsons_data.columns:
            return jsonify({"success": False, "message": "Dataset is not for parkinsons disease"})

        # Return column names to front-end
        return jsonify({"success": True, "columns": uploaded_parkinsons_data.columns.tolist()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

def get_all_patient_data():
    try:
        # Query all patient records from the database
        patients = ParkinsonsPrediction.query.all()

        # If patients are found, return their data as a list of dictionaries
        if patients:
            all_patient_data = []
            for patient in patients:
                patient_data = {
                    'patient_id': patient.PatientID,
                    'patient_username': patient.PatientName,
                    'disease_type': patient.disease_type,
                    'prediction_result': patient.prediction_result
                }
                all_patient_data.append(patient_data)
            return all_patient_data

        # If no patient data is found, return an empty list
        return []

    except Exception as e:
        print(f"Error retrieving patient data: {str(e)}")
        return []


@parkinsons_routes.route("/predict_parkinsons", methods=["POST"])
def predict_parkinsons():
    patientID = session.get("patient_id")
    if "username" not in session or not patientID:
        return redirect(url_for("home"))

    try:
        parkinsons_model = joblib.load(MODEL_PATH)
        with open("selected_features_parkinsons.json", "r") as f:
            selected_features = json.load(f)

        features_for_prediction = [f for f in selected_features if f != "Diagnosis"]
        input_data = []

        for feature in features_for_prediction:
            if feature in request.form:
                try:
                    input_data.append(float(request.form[feature]))
                except ValueError:
                    flash(f"Invalid input for {feature}.", "danger")
                    return redirect(url_for("patient_routes.patient_dashboard"))
            else:
                flash(f"Missing input for {feature}.", "danger")
                return redirect(url_for("patient_routes.patient_dashboard"))

        input_array = np.asarray(input_data).reshape(1, -1)
        prediction = parkinsons_model.predict(input_array)
        prediction_probabilities = parkinsons_model.predict_proba(input_array)[0]

        positive_probability = prediction_probabilities[1] * 100
        negative_probability = prediction_probabilities[0] * 100

        coefficients = parkinsons_model.coef_[0]
        contributions = input_array.flatten() * coefficients
        sorted_features = sorted(zip(features_for_prediction, contributions), key=lambda x: abs(x[1]), reverse=True)

        prediction_result = "positive" if prediction[0] == 1 else "negative"
       
        # Get patient name
        patient = PatientUser.query.filter_by(patient_id=patientID).first()
        patientName = patient.patient_FirstName if patient else "Unknown"

        # Save prediction result
        new_prediction = ParkinsonsPrediction(
            PatientID=patientID,
            PatientName=patientName,
            disease_type="parkinsons disease",
            prediction_result=prediction_result
        )
        db.session.add(new_prediction)
        db.session.commit()

        prediction_id = new_prediction.Prediction_id

        # Save feature contributions
        feature_explanations = []
        for feature, contribution in sorted_features[:18]:
            effect = "increased" if contribution > 0 else "lowered" if contribution < 0 else "no effect"
            feature_explanations.append((feature.replace("_", " ").title(), effect, float(contribution)))

            db.session.add(ParkinsonSymptomContribution(
                PatientID=patientID,
                feature=feature,
                effect=effect,
                contribution=contribution,
                prediction_result=prediction_result,
                prediction_id=prediction_id
            ))

        db.session.commit()

        if patient:
            patient.disease_type = "parkinsons disease"
            patient.prediction_result = prediction_result
            db.session.commit()

        session["disease_type"] = "parkinsons disease"
        session["prediction_result"] = prediction_result

        # Store symptoms
        new_entry = ParkinsonsSymptoms(
            PatientName=patientName,
            PatientID=patientID,
            Age=int(request.form['Age']),
            Gender=int(request.form['Gender']),
            BMI=int(request.form['BMI']),
            PhysicalActivity=int(request.form['PhysicalActivity']),
            DietQuality=int(request.form['DietQuality']),
            SleepQuality=int(request.form['SleepQuality']),
            FamilyHistoryParkinsons=int(request.form['FamilyHistoryParkinsons']),
            TraumaticBrainInjury=int(request.form['TraumaticBrainInjury']),
            Hypertension=int(request.form['Hypertension']),
            Diabetes=int(request.form['Diabetes']),
            Depression=int(request.form['Depression']),
            Stroke=int(request.form['Stroke']),
            Tremor=int(request.form['Tremor']),
            Rigidity=int(request.form['Rigidity']),
            SpeechProblems=int(request.form['SpeechProblems']),
            SleepDisorders=int(request.form['SleepDisorders']),
            Constipation=int(request.form['Constipation']),
            Diagnosis=int(prediction[0]),
            Prediction_id=prediction_id
        )

        db.session.add(new_entry)
        db.session.commit()

        return render_template("proc_pred.html", feature_explanations=feature_explanations)

    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "danger")
        print("Exception occurred:", str(e))
        return redirect(url_for("patient_routes.patient_dashboard"))

def get_all_contributions(patient_id=None, prediction_id=None):
    query = ParkinsonSymptomContribution.query

    if patient_id:
        query = query.filter_by(PatientID=patient_id)
    if prediction_id:
        query = query.filter_by(prediction_id=prediction_id)

    contribution_data = query.all()  # Fetch results

    if contribution_data:
        all_contribution_data = []
        for contribution in contribution_data:
            contribution_record = {
                'PatientID': contribution.PatientID,
                'effect_id': contribution.effect_id,
                'feature': contribution.feature,
                'effect': contribution.effect,
                'contribution': contribution.contribution,
                'prediction_result': contribution.prediction_result,
                'prediction_id': contribution.prediction_id
            }
            all_contribution_data.append(contribution_record)

        return all_contribution_data

    return []  # Return an empty list
   
def get_all_patient_data():
    try:
        # Query all patient records from the database
        parkinsons_patients = ParkinsonsPrediction.query.all()

        # If patients are found, return their data as a list of dictionaries
        if parkinsons_patients:
            parkinsons_patient_data = []
            for parkinson_patient in parkinsons_patients:
                parkinson_record = {
                    'prediction_id':parkinson_patient.Prediction_id,
                    'patient_id': parkinson_patient.PatientID,
                    'patient_username': parkinson_patient.PatientName,
                    'disease_type': parkinson_patient.disease_type,
                    'prediction_result': parkinson_patient.prediction_result
                }
                parkinsons_patient_data.append(parkinson_record)
            return parkinsons_patient_data

        # If no patient data is found, return an empty list
        return []

    except Exception as e:
        print(f"Error retrieving patient data: {str(e)}")
        return []

def get_all_parkinsons_data(patient_id=None,prediction_id=None):
    try:
        query = ParkinsonsSymptoms.query 

        if patient_id:
            query == query.filter_by(PatientID=patient_id)
        if prediction_id:
            query = query.filter_by(id=prediction_id)
        
        parkinsons_data = query.all()
        # If records are found, return their data as a list of dictionaries
        if parkinsons_data:
            all_parkinsons_data = []
            for parkinsons in parkinsons_data:
                parkinsons_record = {
                    'PatientID': parkinsons.PatientID,
                    'PatientName': parkinsons.PatientName,
                    'id': parkinsons.id,
                    'Age': parkinsons.Age,
                    'Gender': parkinsons.Gender,
                    'BMI': parkinsons.BMI,
                    'PhysicalActivity': parkinsons.PhysicalActivity,
                    'DietQuality': parkinsons.DietQuality,
                    'SleepQuality': parkinsons.SleepQuality,
                    'FamilyHistoryParkinsons': parkinsons.FamilyHistoryParkinsons,
                    'TraumaticBrainInjury': parkinsons.TraumaticBrainInjury,
                    'Hypertension': parkinsons.Hypertension,
                    'Diabetes': parkinsons.Diabetes,
                    'Depression': parkinsons.Depression,
                    'Stroke': parkinsons.Stroke,
                    'Tremor': parkinsons.Tremor,
                    'Rigidity': parkinsons.Rigidity,
                    'SpeechProblems': parkinsons.SpeechProblems,
                    'SleepDisorders': parkinsons.SleepDisorders,
                    'Constipation': parkinsons.Constipation,
                    'Diagnosis': parkinsons.Diagnosis
                }
                all_parkinsons_data.append(parkinsons_record)
            return all_parkinsons_data

        # If no records are found, return an empty list
        return []

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return []


@parkinsons_routes.route("/view_parkinsons_patient_results", methods=["GET","POST"])
def view_parkinsons_patient_results():
    # Fetch all patient data
    all_patient_data = get_all_patient_data()

    # If there is patient data, render the results page
    if all_patient_data:
        return render_template("parkinsonsresults.html", patients=all_patient_data)
    else:
        return redirect(url_for("doctor_routes.doctor_dashboard"))

@parkinsons_routes.route("/view_all_parkinsons_results", methods=["GET","POST"])
def view_all_parkinsons_results():
    # Fetch all patient data
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    all_parkinsons_data = get_all_parkinsons_data()

    if patient_id:
      all_parkinsons_data = get_all_parkinsons_data(patient_id=patient_id, prediction_id=prediction_id)
    else:
      all_parkinsons_data = get_all_parkinsons_data()

    if all_parkinsons_data:
        return render_template("parkinsons_patients.html", parkinsons=all_parkinsons_data)
    else:
        flash("No parkinsons disease data is available", "danger")
        return redirect(url_for("doctor_routes.doctor_dashboard"))

@parkinsons_routes.route("/view_parkinsons_contributions", methods=["GET"])
def view_parkinsons_contributions():
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    all_contributions = get_all_contributions(patient_id=patient_id,prediction_id=prediction_id)
    return render_template("diabetescontributions.html", contributions=all_contributions)

@parkinsons_routes.route("/update_status", methods=["POST"])
def update_status():
    selected_rows = []
    csv_file_path = "./uploads/parkinsons_disease_data.csv"

    try:
        # Loop to update the diagnosis in the database
        for key in request.form:
            if key.startswith("diagnosis_"):  # Check for all diagnosis updates
                parkinsons_id = key.split("_")[1]  # Extract ID from "diagnosis_<id>"
                print(parkinsons_id)
                target_value = request.form.get(key)  # Get new diagnosis
                print(target_value)
                
                # Fetch and update the correct record
                parkinsons_record = ParkinsonsSymptoms.query.get(parkinsons_id)
                
                
                if parkinsons_record:
                    parkinsons_record.Diagnosis = None if target_value == "null" else int(target_value)
                    print("hello:",parkinsons_record.id)
                    print("Diagnosis",parkinsons_record.Diagnosis)
                if not parkinsons_record:
                    print(f"Record with ID {parkinsons_id} not found.")

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating diagnosis: {e}", "danger")

    # Collect data for saving to dataset
    for key in request.form:
        if key.startswith("save_this_dataset_"):
            parkinsons_id = key.split("_")[-1]

            row_data = {
              #"PatientID": request.form.get(f"patient_id_{parkinsons_id}"),
              #"Patient Name": request.form.get(f"patient_name_{parkinsons_id}"),
              "Age": request.form.get(f"age_{parkinsons_id}"),
              "Gender": request.form.get(f"gender_{parkinsons_id}"),
              "BMI": request.form.get(f"bmi_{parkinsons_id}"),
              "Physical Activity": request.form.get(f"physical_activity_{parkinsons_id}"),
              "Diet Quality": request.form.get(f"diet_quality_{parkinsons_id}"),
              "Sleep Quality": request.form.get(f"sleep_quality_{parkinsons_id}"),
              "Family History of Parkinson's": request.form.get(f"family_history_{parkinsons_id}"),
              "Traumatic Brain Injury": request.form.get(f"tbi_{parkinsons_id}"),
              "Hypertension": request.form.get(f"hypertension_{parkinsons_id}"),
              "Diabetes": request.form.get(f"diabetes_{parkinsons_id}"),
              "Depression": request.form.get(f"depression_{parkinsons_id}"),
              "Stroke": request.form.get(f"stroke_{parkinsons_id}"),
              "Tremor": request.form.get(f"tremor_{parkinsons_id}"),
              "Rigidity": request.form.get(f"rigidity_{parkinsons_id}"),
              "Speech Problems": request.form.get(f"speech_problems_{parkinsons_id}"),
              "Sleep Disorders": request.form.get(f"sleep_disorders_{parkinsons_id}"),
              "Constipation": request.form.get(f"constipation_{parkinsons_id}"),
             "Diagnosis": request.form.get(f"diagnosis_{parkinsons_id}")
             }
            
            selected_rows.append(row_data)
            
    final_diagnosis =  request.form.get(f"diagnosis_{parkinsons_id}")

            

    # Save data to CSV file
    if 'save_this_dataset' in request.form and selected_rows:
        df_new = pd.DataFrame(selected_rows)

        if os.path.exists(csv_file_path):
            try:
                # Append to existing file (no headers to avoid duplication)
                df_new.to_csv(csv_file_path, mode="a", index=False, header=False)
            except Exception as e:
                flash(f"Error writing to CSV: {e}", "danger")
        else:
            # If file does not exist, create a new one with headers
            df_new.to_csv(csv_file_path, index=False)
    
    PatientID = request.form.get(f"patient_id_{ parkinsons_record.id }")
    print("park:", PatientID)
    PredictionID =request.form.get(f"parkinsons_id_{parkinsons_record.id}")
    print("park:",PredictionID)
   

    send_to_patient = request.form.get(f"send_to_patient_{parkinsons_record.id}") == "on"
    if send_to_patient:
        # Send results to patient
        diagnosis_text = "Positive for Parkinsons" if final_diagnosis == '1' else "Negative for Parkinsons" if final_diagnosis == '0' else "To be determined"
        message = f"Your diagnosis result is available: {diagnosis_text}"

        # Check if a notification already exists for this patient
        existing_notification = Notification.query.filter_by(patient_id=PatientID,type="Parkinsons").first()

        if existing_notification:
            # If a notification exists, update the message
            existing_notification.message = message
        else:
            # If no notification exists, create a new one
            new_notification = Notification(
                patient_id=PatientID,
                message=message,
                type ="Parkinsons"
            )
            db.session.add(new_notification)

        # Commit the changes to the database
        db.session.commit()

    return redirect(url_for("parkinsons_routes.view_all_parkinsons_results", patient_id=PatientID, prediction_id=PredictionID))


