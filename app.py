from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow requests from React

model = joblib.load('xgb_heart_readmission_model.pkl')

# Keep your helper functions (encode_flag, calculate_length_of_stay, etc.) unchanged

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and process values
    ntprobnp = float(data.get('ntprobnp'))
    creatinine = float(data.get('creatinine'))
    urea_nitrogen = float(data.get('urea_nitrogen'))
    sodium = float(data.get('sodium'))
    potassium = float(data.get('potassium'))
    albumin = float(data.get('albumin'))
    crp = float(data.get('c_reactive_protein'))
    hemoglobin = float(data.get('hemoglobin'))
    hematocrit = float(data.get('hematocrit'))
    magnesium = float(data.get('magnesium'))

    flags = {
        "ntprobnp_flag": encode_flag(data.get("ntprobnp_flag")),
        "creatinine_flag": encode_flag(data.get("creatinine_flag")),
        "urea nitrogen_flag": encode_flag(data.get("urea_nitrogen_flag")),
        "sodium_flag": encode_flag(data.get("sodium_flag")),
        "potassium_flag": encode_flag(data.get("potassium_flag")),
        "albumin_flag": encode_flag(data.get("albumin_flag")),
        "c-reactive protein_flag": encode_flag(data.get("c_reactive_protein_flag")),
        "hemoglobin_flag": encode_flag(data.get("hemoglobin_flag")),
        "hematocrit_flag": encode_flag(data.get("hematocrit_flag")),
        "magnesium_flag": encode_flag(data.get("magnesium_flag")),
    }

    admission = encode_admission_type(data.get("admission_type"))
    discharge = encode_discharge_location(data.get("discharge_location"))
    insurance_risk = get_insurance_risk(data.get("insurance"))

    length_of_stay = calculate_length_of_stay(data.get("admit_time"), data.get("discharge_time"))
    admit_weekday = get_admit_weekday(data.get("admit_time"))

    result = {
        "ntprobnp": ntprobnp,
        "creatinine": creatinine,
        "urea_nitrogen": urea_nitrogen,
        "sodium": sodium,
        "potassium": potassium,
        "albumin": albumin,
        "c_reactive_protein": crp,
        "hemoglobin": hemoglobin,
        "hematocrit": hematocrit,
        "magnesium": magnesium,
        **flags,
        **admission,
        **discharge,
        "insurance_risk": insurance_risk,
        "length_of_stay": length_of_stay,
        "admit_weekday": admit_weekday,
    }

    feature_order = [
        'ntprobnp', 'ntprobnp_flag', 'creatinine', 'creatinine_flag', 'urea nitrogen', 'urea nitrogen_flag',
        'sodium', 'sodium_flag', 'potassium', 'potassium_flag', 'albumin', 'albumin_flag',
        'c-reactive protein', 'c-reactive protein_flag', 'hemoglobin', 'hemoglobin_flag',
        'hematocrit', 'hematocrit_flag', 'magnesium', 'magnesium_flag',
        'admission_type_EMERGENCY', 'admission_type_URGENT',
        'discharge_location_HOME', 'discharge_location_HOME HEALTH CARE', 'discharge_location_SNF',
        'discharge_location_SHORT TERM HOSPITAL', 'discharge_location_REHAB/DISTINCT PART HOSP',
        'discharge_location_OTHER FACILITY', 'insurance_risk', 'length_of_stay', 'admit_weekday'
    ]

    X = np.array([result.get(f, 0) for f in feature_order]).reshape(1, -1)
    prediction = model.predict(X)[0]

    return jsonify({'prediction': int(prediction)})
