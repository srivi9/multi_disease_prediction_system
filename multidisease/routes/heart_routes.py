from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from database_models.database import db
from flask import Flask, jsonify, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
import joblib
from routes.admin_routes import admin_routes
from routes.patient_routes import patient_routes
from routes.doctor_routes import doctor_routes
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score
from database_models.database import db
from database_models.patient import PatientUser
from database_models.heart_symptoms import HeartSymptoms
from database_models.heartprediction import HeartPrediction 
from database_models.heart_feature_effect import HeartSymptomContribution
from openpyxl import load_workbook
from database_models.modelaccuracy import AccuracyScore
import json 
from database_models.notification import Notification

uploaded_heart_data = None
selected_features = []
app = Flask(__name__)  
app.config["UPLOAD_FOLDER"] = "uploads"  
app.debug = False
MODEL_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "heart_model.pkl")
heart_routes = Blueprint('heart_routes', __name__)


@heart_routes.route("/lasso_heart", methods=["POST"])
def lasso_heart():
    global uploaded_heart_data, heart_model, selected_features

    if uploaded_heart_data is None:
        flash(("danger", "No dataset uploaded"))
        return redirect(url_for("admin_routes.admin_dashboard"))

    try:
        # Start with all available features except target
        all_features = [col for col in uploaded_heart_data.columns if col != "target"]

        # Split data into features (X) and target labels (Y)
        X = uploaded_heart_data[all_features]
        Y = uploaded_heart_data["target"]

        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Get regularization strength from form if provided, otherwise use default
        alpha_value = float(request.form.get("alpha_value", 0.01))

        # STEP 1: Train on ALL features to get importance
        lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=alpha_value, max_iter=5000)
        lasso_model.fit(X_train, Y_train)

        # Extract feature importance values
        feature_importance = lasso_model.coef_[0]

        # Create importance dictionary for all features
        importance_dict = {all_features[i]: feature_importance[i] for i in range(len(all_features))}

        # Sort by absolute importance
        sorted_importance = {k: v for k, v in sorted(importance_dict.items(),
                                                     key=lambda item: abs(item[1]),
                                                     reverse=True)}

        # Identify features that LASSO considered important (non-zero coefficients)
        selected_features = [feature for feature, importance in importance_dict.items() if abs(importance) > 0]

        if not selected_features:
            flash(("warning", "LASSO eliminated all features. Try decreasing regularization strength."))
            return redirect(url_for("admin_routes.admin_dashboard"))

        # STEP 2: Train a NEW model on only selected features
        X_train_selected = X_train[selected_features]  # Use only selected features
        X_test_selected = X_test[selected_features]    # Use only selected features

        heart_model = LogisticRegression(penalty='l1', solver='liblinear', C=alpha_value, max_iter=5000)
        heart_model.fit(X_train_selected, Y_train)

        # Save selected features (including target)
        selected_features_with_target = selected_features.copy()
        selected_features_with_target.append("target")
        with open("selected_features.json", "w") as f:
            json.dump(selected_features_with_target, f)

        # Save the model trained on selected features only
        joblib.dump(heart_model, "uploads/heart_model.pkl")

        # Calculate metrics using selected features
        train_accuracy = f"{accuracy_score(Y_train, heart_model.predict(X_train_selected)) * 100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, heart_model.predict(X_test_selected)) * 100:.2f}%"

        train_precision = f"{precision_score(Y_train, heart_model.predict(X_train_selected)) * 100:.2f}%"
        test_precision = f"{precision_score(Y_test, heart_model.predict(X_test_selected)) * 100:.2f}%"
        train_recall = f"{recall_score(Y_train, heart_model.predict(X_train_selected)) * 100:.2f}%"
        test_recall = f"{recall_score(Y_test, heart_model.predict(X_test_selected)) * 100:.2f}%"
        

        disease_type = "Heart Disease"  
        

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
        # Flash training metrics
        flash("Model trained successfully with LASSO feature selection!")
        flash(f"LASSO selected {len(selected_features)} features out of {len(all_features)} available features")
        flash(f"Train Accuracy: {train_accuracy}")
        flash(f"Test Accuracy: {test_accuracy}")
        flash(f"Train Precision: {train_precision}")
        flash(f"Test Precision: {test_precision}")
        flash(f"Train Recall: {train_recall}")
        flash(f"Test Recall: {test_recall}")

        # Display selected features by importance
        flash("Selected features by importance:")
        for feature, importance in sorted_importance.items():
            if abs(importance) > 0:
                flash(f"{feature}: {importance:.4f}")

    except Exception as e:
        import traceback
        flash(("danger", f"Error during training: {str(e)}"))
        flash(("danger", f"Traceback: {traceback.format_exc()}"))

    return redirect(url_for("training_loading"))


@heart_routes.route("/train_heart_model", methods=["POST"])
def train_model():
    global uploaded_heart_data, heart_model, selected_features

    if uploaded_heart_data is None:
        flash("No dataset uploaded", "danger")
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Get selected features from form input
    selected_features = request.form.getlist("features")

    if not selected_features:
        flash("No features selected for training", "danger")
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Ensure 'target' is not selected as a feature
    selected_features = [f for f in selected_features if f != "target"]
    selected_features.append("target")

    try:
        # Filter dataset with selected features
        dataset = uploaded_heart_data[selected_features]
        with open("selected_features.json", "w") as f:
            json.dump(selected_features, f)


        # Split data into features (X) and target labels (Y)
        X = dataset.drop(columns="target", axis=1)
        Y = dataset["target"]

        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Train the model
        heart_model = LogisticRegression(max_iter=5000)
        heart_model.fit(X_train, Y_train)

        # Save the model
        joblib.dump(heart_model, MODEL_PATH)

       # Calculate accuracy
        train_accuracy = f"{accuracy_score(Y_train, heart_model.predict(X_train))*100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, heart_model.predict(X_test))*100:.2f}%"

        # Calculate precision and recall
        train_precision = f"{precision_score(Y_train, heart_model.predict(X_train))*100:.2f}%"
        test_precision = f"{precision_score(Y_test, heart_model.predict(X_test))*100:.2f}%"
        train_recall = f"{recall_score(Y_train, heart_model.predict(X_train))*100:.2f}%"
        test_recall = f"{recall_score(Y_test, heart_model.predict(X_test))*100:.2f}%"


        disease_type = "Heart Disease"  
        

        previous_current = AccuracyScore.query.filter_by(is_current_model=True,disease_type=disease_type).first()
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

@heart_routes.route("/heartdisease")
def heart_disease():
    return render_template("heartdisease.html")


@heart_routes.route("/upload_heart_dataset", methods=["POST"])
def upload_heart_dataset():
    global uploaded_heart_data  # Store dataset globally

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
        uploaded_heart_data = pd.read_csv(filepath)

        # Ensure it has a 'target' column
        if "target" not in uploaded_heart_data.columns:
            return jsonify({"success": False, "message": "Dataset is not for heart disease"})

        # Return column names to front-end
        return jsonify({"success": True, "columns": uploaded_heart_data.columns.tolist()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    


def get_all_patient_data():
    try:
        # Query all patient records from the database
        patients = HeartPrediction.query.all()

        # If patients are found, return their data as a list of dictionaries
        if patients:
            all_patient_data = []
            for patient in patients:
                patient_data = {
                    'prediction_id':patient.Prediction_id,
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
def get_all_heart_data(patient_id=None, prediction_id=None):
    try:
        query = HeartSymptoms.query

        if patient_id:
            query = query.filter_by(PatientID=patient_id)
        if prediction_id:
            query = query.filter_by(id=prediction_id)

        heart_data = query.all()  # Fetch results

        if heart_data:
            all_heart_data = []
            for heart in heart_data:
                heart_record = {
                    'PatientID': heart.PatientID,
                    'PatientName': heart.PatientName,
                    'id': heart.id,
                    'age': heart.Age,
                    'sex': heart.Sex,
                    'chest_pain_type': heart.Chest_Pain_Type,
                    'resting_bp': heart.Resting_Blood_Pressure,
                    'cholesterol': heart.Cholesterol,
                    'fasting_blood_sugar': heart.Fasting_blood_sugar,
                    'ecg_result': heart.ECG_result,
                    'heart_rate': heart.Heart_rate,
                    'angina': heart.Angina,
                    'old_peak': heart.Old_peak,
                    'slope': heart.Slope,
                    'num_major_vessels': heart.Number_of_major_vessels,
                    'thalassemia': heart.Thalassemia,
                    'target': heart.Target  
                }
                all_heart_data.append(heart_record)

            return all_heart_data

        return []  # Return an empty list instead of None when no data is found
    except Exception as e:
        print(f"Error fetching heart data: {e}")
        return []  # Also return an empty list on exception
    
def get_all_contributions(patient_id=None, prediction_id=None):
    query = HeartSymptomContribution.query

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
                    'prediction_result' : contribution.prediction_result,
                    'prediction_id' : contribution.prediction_id
                }
                all_contribution_data.append(contribution_record)

            return all_contribution_data

    return []  # Return an empty list instead of None when no data is found

@heart_routes.route("/predict_heart", methods=["POST"])
def predict_heart():
    patientID = session.get("patient_id")
    if "username" not in session or not patientID:
        return redirect(url_for("home"))

    try:
        heart_model = joblib.load(MODEL_PATH)  # Load the saved model

        # Load selected features (used during training)
        with open("selected_features.json", "r") as f:
            selected_features = json.load(f)

        # Remove 'target' from selected features (as it is not part of the input data)
        features_for_prediction = [f for f in selected_features if f != "target"]

        input_data = []
        for feature in features_for_prediction:
            if feature in request.form:
                try:
                    input_data.append(float(request.form[feature]))  # Convert form value to float
                except ValueError:
                    flash(f"Invalid input for {feature}.", "danger")
                    return redirect(url_for("patient_routes.patient_dashboard"))
            else:
                flash(f"Missing input for {feature}.", "danger")
                return redirect(url_for("patient_routes.patient_dashboard"))

        input_array = np.asarray(input_data).reshape(1, -1)

        # Make prediction and get probabilities
        prediction = heart_model.predict(input_array)
        prediction_probabilities = heart_model.predict_proba(input_array)[0]

        positive_probability = prediction_probabilities[1] * 100
        negative_probability = prediction_probabilities[0] * 100
        coefficients = heart_model.coef_[0]
        contributions = input_array.flatten() * coefficients
        sorted_features = sorted(zip(features_for_prediction, contributions), key=lambda x: abs(x[1]), reverse=True)
        # Determine prediction result
        prediction_result = "positive" if prediction[0] == 1 else "negative"

        # Fetch patient name and save prediction result
        patient = PatientUser.query.filter_by(patient_id=patientID).first()
        patientName = patient.patient_FirstName if patient else "Unknown"

        new_prediction = HeartPrediction(
            PatientID=patientID,
            PatientName=patientName,
            disease_type="heart disease",
            prediction_result=prediction_result
        )
        db.session.add(new_prediction)
        db.session.commit()

        prediction_id = new_prediction.Prediction_id

        # Save feature explanations
        feature_explanations = []
        for feature, importance in zip(selected_features, input_array.flatten() * heart_model.coef_[0]):
            effect = "increased" if importance > 0 else "lowered" if importance < 0 else "no effect"
            feature_explanations.append((feature.replace("_", " ").title(), effect, float(importance)))

            db.session.add(HeartSymptomContribution(
                PatientID=patientID,
                feature=feature,
                effect=effect,
                contribution=importance,
                prediction_result=prediction_result,
                prediction_id=prediction_id
            ))

        db.session.commit()

        # Update patient record
        if patient:
            patient.disease_type = "heart disease"
            patient.prediction_result = prediction_result
            db.session.commit()
    
        session["disease_type"] = "heart disease"
        session["prediction_result"] = prediction_result
        
        # Extract form data
        patientID = session["patient_id"]
        age = int(request.form['age'])
        sex = int(request.form['gender'])
        chest_pain_type = int(request.form['chestpain'])
        resting_bp = float(request.form['resting-bloodpressure'])
        cholesterol = float(request.form['cholestoral'])
        fasting_blood_sugar = int(request.form['fasting-bloodsugar'])
        ecg_result = int(request.form['resting-heartrate'])
        heart_rate = int(request.form['max-heartrate'])
        angina = int(request.form['exercise-induced-angina'])
        old_peak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        num_major_vessels = int(request.form['number-of-major-vessels'])
        thalassemia = int(request.form['thalassemia'])

        # Store patient symptoms in database
        new_entry = HeartSymptoms(
            PatientName = patientName,
            PatientID = patientID,
            Age=age,
            Sex=sex,
            Chest_Pain_Type=chest_pain_type,
            Resting_Blood_Pressure=resting_bp,
            Cholesterol=cholesterol,
            Fasting_blood_sugar=fasting_blood_sugar,
            ECG_result=ecg_result,
            Heart_rate=heart_rate,
            Angina=angina,
            Old_peak=old_peak,
            Slope=slope,
            Number_of_major_vessels=num_major_vessels,
            Thalassemia=thalassemia,
            Target=None,
            Prediction_id=prediction_id
        )
        
        db.session.add(new_entry)
        db.session.commit()
        
        db.session.add(new_entry)
        db.session.commit()
        return render_template("proc_pred.html", feature_explanations=feature_explanations)

    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "danger")
        return redirect(url_for("patient_routes.patient_dashboard"))


@heart_routes.route("/view_all_patient_results", methods=["GET","POST"])
def view_all_patient_results():
    # Fetch all patient data
    all_patient_data = get_all_patient_data()

    # If there is patient data, render the results page
    if all_patient_data:
        return render_template("patientresults.html", patients=all_patient_data)
    else:
        flash("No patient data available", "danger")
        return redirect(url_for("doctor_routes.doctor_dashboard"))

@heart_routes.route("/view_all_contributions", methods=["GET"])
def view_all_contributions():
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    all_contributions = get_all_contributions(patient_id=patient_id,prediction_id=prediction_id)
    return render_template("heartcontributions.html", contributions=all_contributions)
    
@heart_routes.route("/view_all_heart_results", methods=["GET"])
def view_all_heart_results():
    patient_id = request.args.get("patient_id") 
    prediction_id = request.args.get("prediction_id")
    print(f"Received Patient ID: {patient_id}, Prediction ID: {prediction_id}")

    if patient_id:
        # Fetch symptoms for only the selected patient
        all_heart_data = get_all_heart_data(patient_id=patient_id, prediction_id=prediction_id)
    else:
        # Fetch all data if no patient_id is provided
        all_heart_data = get_all_heart_data()

    if all_heart_data:
        return render_template("heartresults.html", hearts=all_heart_data)
    else:
        flash("No heart disease data available", "danger")
        return redirect(url_for("doctor_routes.doctor_dashboard"))



@heart_routes.route("/update_diagnosis", methods=["POST"])
def update_diagnosis():
    selected_rows = []
    csv_file_path = "./uploads/heart.csv"

    try:
        # Loop to update the diagnosis in the database
        for key in request.form:
            if key.startswith("target_"):  # Check for all diagnosis updates
                heart_id = key.split("_")[1]  # Extract heart ID from "target_<heart_id>"
                target_value = request.form.get(key)  # Get new diagnosis

                # Fetch and update the correct record
                heart_record = HeartSymptoms.query.get(heart_id)
                if heart_record:
                    heart_record.Target = None if target_value == "null" else int(target_value)

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating diagnosis: {e}", "danger")

    # Collect data for saving to dataset
    for key in request.form:
        if key.startswith("save_to_dataset_"):
            heart_id = key.split("_")[-1]

            row_data = {
                "Age": request.form.get(f"age_{heart_id}"),
                "Sex": request.form.get(f"sex_{heart_id}"),
                "Chest Pain Type": request.form.get(f"chest_pain_type_{heart_id}"),
                "Resting BP": request.form.get(f"resting_bp_{heart_id}"),
                "Cholesterol": request.form.get(f"cholesterol_{heart_id}"),
                "Fasting Blood Sugar": request.form.get(f"fasting_blood_sugar_{heart_id}"),
                "ECG Result": request.form.get(f"ecg_result_{heart_id}"),
                "Heart Rate": request.form.get(f"heart_rate_{heart_id}"),
                "Angina": request.form.get(f"angina_{heart_id}"),
                "Old Peak": request.form.get(f"old_peak_{heart_id}"),
                "Slope": request.form.get(f"slope_{heart_id}"),
                "Major Vessels": request.form.get(f"num_major_vessels_{heart_id}"),
                "Thalassemia": request.form.get(f"thalassemia_{heart_id}"),
                "Diagnosis": request.form.get(f"target_{heart_id}")
            }

            selected_rows.append(row_data)

    final_diagnosis = request.form.get(f"target_{heart_id}")

    # Save data to CSV file
    if 'save_to_dataset' in request.form and selected_rows:
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

    PatientID = request.form.get(f"patient_id_{heart_id}")
    PredictionID = request.form.get(f"heart_id_{heart_id}")

    send_to_patient = request.form.get(f"send_to_patient_{heart_id}") == "on"
    if send_to_patient:
        # Send results to patient
        diagnosis_text = "Positive for heart disease" if final_diagnosis == '1' else "Negative for heart disease" if final_diagnosis == '0' else "To be determined"
        message = f"Your diagnosis result is available: {diagnosis_text}"

        # Check if a notification already exists for this patient
        existing_notification = Notification.query.filter_by(patient_id=PatientID,type="heart disease").first()

        if existing_notification:
            # If a notification exists, update the message
            existing_notification.message = message
        else:
            # If no notification exists, create a new one
            new_notification = Notification(
                patient_id=PatientID,
                message=message,
                type ="heart disease"
            )
            db.session.add(new_notification)

        # Commit the changes to the database
        db.session.commit()

    return redirect(url_for("heart_routes.view_all_heart_results", patient_id=PatientID, prediction_id=PredictionID))
