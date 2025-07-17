from flask import Blueprint, render_template, request, redirect, session, url_for, flash
from database_models.database import db
from flask import Flask,jsonify, render_template, request, redirect, session, url_for, flash
import os
import json
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from database_models.patient import PatientUser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score
from database_models.diabetes_expert_pred import DiabetesSymptoms
from database_models.diabetes_system_pred import DiabetesPrediction
from database_models.diabetes_feature_effect import DiabetesFeatureEffectContribution
import joblib
from database_models.feature_explanations import FeatureExplanation
from database_models.notification import Notification
from database_models.modelaccuracy import AccuracyScore

uploaded_diabetes_data = None
selected_features = []
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.debug = False
DIABETES_MODEL_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "diabetes_model.pkl")

# this function is to load the existing model
def load_model():
    global diabetes_model
    if os.path.exists(DIABETES_MODEL_PATH):
        diabetes_model = joblib.load(DIABETES_MODEL_PATH)

# this function is to save the model
def save_model(model):
    joblib.dump(model, DIABETES_MODEL_PATH)

diabetes_routes = Blueprint('diabetes_routes', __name__)
diabetes_model = None

# this function is the lasso algorithm
@diabetes_routes.route("/train_lasso", methods=["POST"])
def train_lasso():
    global uploaded_diabetes_data, selected_features

    if uploaded_diabetes_data is None:
        flash(("danger", "No dataset uploaded"))
        return redirect(url_for("admin_routes.admin_dashboard"))

    try:
        # Start with all available features except outcome
        all_features = [col for col in uploaded_diabetes_data.columns if col != "Outcome"]

        # Split data into features (X) and target labels (Y)
        X = uploaded_diabetes_data[all_features]
        Y = uploaded_diabetes_data["Outcome"]

        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # regularization strength
        alpha_value = 0.01

        # Train on all features to get importance
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

        # Train a new model on only selected features 
        X_train_selected = X_train[selected_features]  # 
        X_test_selected = X_test[selected_features]    # 

        diabetes_model = LogisticRegression(penalty='l1', solver='liblinear', C=alpha_value, max_iter=5000)  # 
        diabetes_model.fit(X_train_selected, Y_train)  # 

        # Save selected features with outcome
        selected_features_with_outcome = selected_features.copy()
        selected_features_with_outcome.append("Outcome")
        with open("selected_features_diabetes.json", "w") as f:
            json.dump(selected_features_with_outcome, f)

        # Save the model trained on selected features only
        joblib.dump(diabetes_model, "uploads/diabetes_model.pkl")  

        # Calculate metrics using selected features
        train_accuracy = f"{accuracy_score(Y_train, diabetes_model.predict(X_train_selected)) * 100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, diabetes_model.predict(X_test_selected)) * 100:.2f}%"

        train_precision = f"{precision_score(Y_train, diabetes_model.predict(X_train_selected)) * 100:.2f}%"
        test_precision = f"{precision_score(Y_test, diabetes_model.predict(X_test_selected)) * 100:.2f}%"
        train_recall = f"{recall_score(Y_train, diabetes_model.predict(X_train_selected)) * 100:.2f}%"
        test_recall = f"{recall_score(Y_test, diabetes_model.predict(X_test_selected)) * 100:.2f}%"
        
        disease_type = "Diabetes"

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


# this function is to train the model using the logistic regression model.
@diabetes_routes.route("/train_model", methods=["POST"])
def train_model():
    global uploaded_diabetes_data, selected_features

    if uploaded_diabetes_data is None:
        flash(("danger", "No dataset uploaded"))
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Get selected features from form input
    selected_features = request.form.getlist("features")

    if not selected_features:
        flash(("danger", "No features selected for training"))
        return redirect(url_for("admin_routes.admin_dashboard"))

    # Ensure 'target' is not selected as a feature
    selected_features = [f for f in selected_features if f != "Outcome"]
    selected_features.append("Outcome")

    try:
        # Filter dataset with selected features
        dataset = uploaded_diabetes_data[selected_features]
        with open("selected_features_diabetes.json", "w") as f:
            json.dump(selected_features, f)

        # Split data into features (X) and target labels (Y)
        X = dataset.drop(columns="Outcome", axis=1)
        Y = dataset["Outcome"]

        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Train the model
        diabetes_model = LogisticRegression(max_iter=5000)
        diabetes_model.fit(X_train, Y_train)

        # Save the model
        joblib.dump(diabetes_model, "uploads/diabetes_model.pkl")

        # Calculate accuracy
        train_accuracy = f"{accuracy_score(Y_train, diabetes_model.predict(X_train))*100:.2f}%"
        test_accuracy = f"{accuracy_score(Y_test, diabetes_model.predict(X_test))*100:.2f}%"

        # Calculate precision and recall
        train_precision = f"{precision_score(Y_train, diabetes_model.predict(X_train))*100:.2f}%"
        test_precision = f"{precision_score(Y_test, diabetes_model.predict(X_test))*100:.2f}%"
        train_recall = f"{recall_score(Y_train, diabetes_model.predict(X_train))*100:.2f}%"
        test_recall = f"{recall_score(Y_test, diabetes_model.predict(X_test))*100:.2f}%"
        
        disease_type = "Diabetes"

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
        flash(("danger", f"Error during training: {str(e)}"))

    return redirect(url_for("training_loading"))

    
# this function is to upload the diabetes dataset
@diabetes_routes.route("/upload_diabetes_dataset", methods=["POST"])
def upload_diabetes_dataset():
    global uploaded_diabetes_data
 
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
        uploaded_diabetes_data = pd.read_csv(filepath)

        # Ensure it has a 'target' column
        if "Outcome" not in uploaded_diabetes_data.columns:
            return jsonify({"success": False, "message": "Dataset is not for diabetes disease"})
        
        
        # Return column names to front-end
        return jsonify({"success": True, "columns": uploaded_diabetes_data.columns.tolist()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# this function is to predict the likelihood of diabetes being present in the person.
@diabetes_routes.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    patientID = session.get("patient_id")
    if "username" not in session or not patientID:
        return redirect(url_for("home"))

    try:
        diabetes_model = joblib.load("uploads/diabetes_model.pkl")

        with open("selected_features_diabetes.json", "r") as f:
            selected_features = json.load(f)

        features_for_prediction = [f for f in selected_features if f != "Outcome"]

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

        prediction = diabetes_model.predict(input_array)
        prediction_probabilities = diabetes_model.predict_proba(input_array)[0]

        positive_probability = prediction_probabilities[1] * 100
        negative_probability = prediction_probabilities[0] * 100

        coefficients = diabetes_model.coef_[0]
        contributions = input_array.flatten() * coefficients
        sorted_features = sorted(zip(features_for_prediction, contributions), key=lambda x: abs(x[1]), reverse=True)

        prediction_result = "positive" if prediction[0] == 1 else "negative"
        

        # Fetch patient name
        patient = PatientUser.query.filter_by(patient_id=patientID).first()
        patientName = patient.patient_FirstName if patient else "Unknown"

        # Save prediction result
        new_prediction = DiabetesPrediction(
            PatientID=patientID,
            PatientName=patientName,
            disease_type="diabetes",
            prediction_result=prediction_result
        )
        db.session.add(new_prediction)
        db.session.commit()

        prediction_id = new_prediction.Prediction_id

        # Save feature explanations
        feature_explanations = []
        for feature, contribution in sorted_features[:9]:
            effect = "increased" if contribution > 0 else "lowered" if contribution < 0 else "no effect"
            feature_explanations.append((feature.replace("_", " ").title(), effect, float(contribution)))

            db.session.add(DiabetesFeatureEffectContribution(
                PatientID=patientID,
                feature=feature,
                effect=effect,
                contribution=contribution,
                prediction_result=prediction_result,
                prediction_id=prediction_id
            ))

        db.session.commit()

        if patient:
            patient.disease_type = "diabetes"
            patient.prediction_result = prediction_result
            db.session.commit()

        session["disease_type"] = "diabetes"
        session["prediction_result"] = prediction_result

        patientName = session["username"]
        patientID = session["patient_id"]
        pregnancies = request.form['Pregnancies']
        glucose = request.form['Glucose']
        blood_pressure = request.form['BloodPressure']
        skin_thickness = request.form['SkinThickness']
        insulin_level = request.form['Insulin']
        BMI = request.form['BMI']
        diabetes_pedigree=request.form['DiabetesPedigreeFunction']
        age = request.form['Age']
       

        # Store the new patient entry in the database
        new_entry = DiabetesSymptoms(
            PatientName=patientName,
            PatientID=patientID,
            Pregnancies=pregnancies,
            Glucose=glucose,
            Blood_pressure=blood_pressure,
            Skin_thickness=skin_thickness,
            Insulin_level=insulin_level,
            BMI=BMI,
            Diabetes_Pedigree=diabetes_pedigree,
            Age=age,
            target = int(prediction[0])
        )

        db.session.add(new_entry)
        db.session.commit()

       
        
        
        return render_template("proc_pred.html")

    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "danger")
        return redirect(url_for("patient_routes.patient_dashboard"))



#this function is to get all the contributions regarding the features
def get_all_contributions(patient_id=None, prediction_id=None):
    query = DiabetesFeatureEffectContribution.query

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

#this function is to view the contributions
@diabetes_routes.route("/view_diabetes_contributions", methods=["GET"])
def view_diabetes_contributions():
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    all_contributions = get_all_contributions(patient_id=patient_id,prediction_id=prediction_id)
    return render_template("diabetescontributions.html", contributions=all_contributions)

#this is to get all the patient data
def get_all_patient_data():
    try:
        # Query all patient records from the database
        diabetes_patients = DiabetesPrediction.query.all()

        # If patients are found, return their data as a list of dictionaries
        if diabetes_patients:
            diabete_patient_data = []
            for diabete_patient in diabetes_patients:
                diabete_record = {
                    'prediction_id':diabete_patient.Prediction_id,
                    'patient_id': diabete_patient.PatientID,
                    'patient_username': diabete_patient.PatientName,
                    'disease_type': diabete_patient.disease_type,
                    'prediction_result': diabete_patient.prediction_result
                }
                diabete_patient_data.append(diabete_record)
            return diabete_patient_data

        # If no patient data is found, return an empty list
        return []

    except Exception as e:
        print(f"Error retrieving patient data: {str(e)}")
        return []

#this function is to get all the diabetes symptoms for the patient
def get_all_diabetes_data(patient_id=None,prediction_id=None):
    try:
        query = DiabetesSymptoms.query

        if patient_id:
            query = query.filter_by(PatientID=patient_id)
        if prediction_id:
            query = query.filter_by(id=prediction_id)
        
        diabetes_data = query.all()
        
        # If records are found, return their data as a list of dictionaries
        if diabetes_data:
            all_diabetes_data = []
            for diabetes in diabetes_data:
                diabetes_record = {
                    'PatientID': diabetes.PatientID,
                    'PatientName': diabetes.PatientName,
                    'id': diabetes.id,
                    'Pregnancies': diabetes.Pregnancies,
                    'Glucose': diabetes.Glucose,
                    'BloodPressure': diabetes.Blood_pressure,  
                    'SkinThickness': diabetes.Skin_thickness,
                    'InsulinLevel': diabetes.Insulin_level,
                    'BMI': diabetes.BMI,
                    'DiabetesPedigree': diabetes.Diabetes_Pedigree,
                    'Age': diabetes.Age,
                    'target': diabetes.target
                }
                all_diabetes_data.append(diabetes_record)
            return all_diabetes_data

        # If no records are found, return an empty list
        return []

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return []


# this function is to view the diabete results
@diabetes_routes.route("/view_diabete_patient_results", methods=["GET", "POST"])
def view_diabete_patient_results():
    # Fetch all patient data
    all_patient_data = get_all_patient_data()

    # If there is patient data, render the results page
    if all_patient_data:
        return render_template("diabete_patients.html", diabetes_patients=all_patient_data)
    else:
        return redirect(url_for("doctor_routes.doctor_dashboard"))

#this it to view the symptoms of the patients
@diabetes_routes.route("/view_all_diabetes_results", methods=["GET", "POST"])
def view_all_diabetes_results():
    # Fetch patient ID from request parameters
    patient_id = request.args.get("patient_id")
    prediction_id = request.args.get("prediction_id")
    print(f"Received Patient ID: {patient_id}, Prediction ID: {prediction_id}")

    if patient_id:
        
        all_diabetes_data = get_all_diabetes_data(patient_id=patient_id, prediction_id=prediction_id)
    else:
        all_diabetes_data = get_all_diabetes_data()

    # If there is patient data, render the results page
    if all_diabetes_data:
        return render_template("diabetesresults.html", diabetes=all_diabetes_data)
    else:
        return redirect(url_for("doctor_routes.doctor_dashboard"))

#this function is to update the Diagnosis
@diabetes_routes.route("/update_result", methods=["POST"])
def update_result():
    print("Form submitted:", request.form)
    selected_rows = []
    csv_file_path = "./uploads/diabetes.csv"

    try:
        for key in request.form:
            if key.startswith("target_"):
                diabetes_id = key.split("_")[1]
                target_value = request.form.get(key)

                diabetes_record = DiabetesSymptoms.query.get(diabetes_id)
                print(diabetes_record)
                if diabetes_record:
                    diabetes_record.target = None if target_value == "null" else int(target_value)
                    db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error: {e}")  
        flash(f"Error updating diagnosis: {e}", "danger")

    # Collect data for saving to dataset
    for key in request.form:
        if key.startswith("savedata_"):
            diabetes_id = key.split("_")[-1]
            row_data = {
                "Pregnancies": request.form.get(f"pregnancies_{diabetes_id}"),
                "Glucose": request.form.get(f"glucose_{diabetes_id}"),
                "BloodPressure": request.form.get(f"blood_pressure_{diabetes_id}"),
                "SkinThickness": request.form.get(f"skin_thickness_{diabetes_id}"),
                "InsulinLevel": request.form.get(f"insulin_level_{diabetes_id}"),
                "BMI": request.form.get(f"bmi_{diabetes_id}"),
                "DiabetesPedigree": request.form.get(f"diabetes_pedigree_{diabetes_id}"),
                "Age": request.form.get(f"age_{diabetes_id}"),
                "Outcome": request.form.get(f"target_{diabetes_id}"),
            }
            selected_rows.append(row_data)
    outcome = request.form.get(f"target_{diabetes_id}")

    # Save data to CSV file
    if "savedata" in request.form and selected_rows:
        df_new = pd.DataFrame(selected_rows)

        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Ensure the directory exists

        try:
            with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
                if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
                    f.write("\n")  # Ensure a newline before appending data
                df_new.to_csv(f, index=False, header=False)

        except Exception as e:
            flash(f"Error writing to CSV: {e}", "danger")
    patient_id = request.form.get(f"patient_id_{ diabetes_record.id}")
    prediction_id = request.form.get(f"diabetes_id_{diabetes_record.id}")

    send_to_patient = request.form.get(f"send_to_patient_{diabetes_record.id}") == "on"
    if send_to_patient:
        # Send results to patient
        diagnosis_text = "Positive for diabetes please book an appointment to see Dr Livingstone by using our our online booking portal" if outcome == '1' else "Negative for diabetes" if outcome == '0' else "Further testing recommended. Please visit the clinic for additional evaluation."
        message = f"Your diagnosis result is available: {diagnosis_text}"

        # Check if a notification already exists for this patient
        existing_notification = Notification.query.filter_by(patient_id=patient_id,type="diabetes").first()

        if existing_notification:
            # If a notification exists, update the message
            existing_notification.message = message
        else:
            # If no notification exists, create a new one
            new_notification = Notification(
                patient_id=patient_id,
                message=message,
                type="diabetes"
            )
            db.session.add(new_notification)

        # Commit the changes to the database
        db.session.commit()
    return redirect(url_for("diabetes_routes.view_all_diabetes_results",patient_id =patient_id,prediction_id=prediction_id))


#this function is to send the explanation of the risk factors to the patient.
@diabetes_routes.route("/diabetesexplanation", methods=["POST"])
def diabetes_explanation():
    patient_id = request.form.get("patient_id")
    send_explanation_checked = request.form.get("send_explanation") == "yes"

    if not send_explanation_checked:
        flash("Explanation not sent. Checkbox was not selected.")
        return redirect(request.referrer)

    # Retrieve contributions for that patient
    contributions = get_all_contributions(patient_id)

    explanation = f"Hello,\n\n"
    explanation += "Here is your diabetes risk analysis:\n\n"

    for c in contributions:
        if c['effect'] == "increased":
            explanation += f"- Your {c['feature']} is **high**, which may increase your diabetes risk.\n"
        elif c['effect'] == "lowered":
            explanation += f"- Your {c['feature']} is **low**, which may lower your diabetes risk.\n"
        else:
            explanation += f"- Your {c['feature']} has **no significant effect** on diabetes risk.\n"

    # Save or update in the database
    existing = FeatureExplanation.query.filter_by(
        patient_id=patient_id,
        disease_type="diabetes"
    ).first()

    if existing:
        existing.explanation_text = explanation
    else:
        new_explanation = FeatureExplanation(
            patient_id=patient_id,
            explanation_text=explanation,
            disease_type="diabetes"
        )
        db.session.add(new_explanation)

    db.session.commit()

    flash("Explanation saved for patient.")
    return redirect(request.referrer)  # Stay on the same page

# this is the function invoked when the patient wishes to view their explanation of the risk factors.
@diabetes_routes.route("/view_explanation_diabetes")
def view_explanation_diabetes():
    # Fetch the explanation from the database
    patient_id = session["patient_id"]
    explanation = FeatureExplanation.query.filter_by(
        patient_id=patient_id,
        disease_type="diabetes"
    ).first()

    if not explanation:
        flash("No explanation found for this patient.")
        return redirect(request.referrer)

    return render_template("view_explanation.html", explanation=explanation)
