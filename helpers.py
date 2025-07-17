
from database_models.heart_symptoms import HeartSymptoms
#this function is to get all the data regarding the symptoms of the patient
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
