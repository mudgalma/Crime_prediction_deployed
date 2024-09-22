from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__,template_folder='./Templates')

# Define the crime category mapping
crime_mapping = {
    0: 'crimes against person',
    1: 'crimes against public order',
    2: 'fraud and white-collar crimes',
    3: 'other crimes',
    4: 'property crimes',
    5: 'violent crimes'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_data = request.form.to_dict()

    # Get current date for Date_Reported fields
    current_date = datetime.now()
    Date_Reported_month = current_date.month  # Current month
    Date_Reported_dayofweek = current_date.weekday()  # Day of the week (Monday=0, Sunday=6)
    Date_Reported_dayofmonth = current_date.day  # Day of the month

    # Handle form inputs
    try:
        # Numerical inputs
        Victim_Age = form_data.get('Victim_Age')
        if Victim_Age: Victim_Age = int(Victim_Age)
        Reporting_District_no = int(form_data.get('Reporting_District_no'))
        hour = int(form_data.get('hour'))
        Date_Occurred_month = int(form_data.get('Date_Occurred_month'))
        Date_Occurred_dayofweek = int(form_data.get('Date_Occurred_dayofweek'))
        Date_Occurred_dayofmonth = int(form_data.get('Date_Occurred_dayofmonth'))
        Premise_Code = int(form_data.get('Premise_Code'))

        # Categorical inputs
        Area_Name = form_data.get('Area_Name')
        Victim_Sex = form_data.get('Victim_Sex', 'M')  # Default to 'M'
        Victim_Descent = form_data.get('Victim_Descent', 'H')  # Default to 'H'
        Status_Description = form_data.get('Status_Description', 'IC')  # Default to 'IC'
        Location = form_data.get('Location')

        # Default values for the missing features
        Modus_Operandi = '0416 1241 1243 1813 1821 2000'  # Default value for Modus_Operandi
        Weapon_Used_Code = 400.0  # Default value for Weapon_Used_Code
        Part_1_2 = 2.0  # Default value for Part 1-2

        # Prepare the input DataFrame
        input_data = pd.DataFrame([{
            'Victim_Age': Victim_Age,
            'Reporting_District_no': Reporting_District_no,
            'hour': hour,
            'Date_Reported_month': Date_Reported_month,
            'Date_Reported_dayofweek': Date_Reported_dayofweek,
            'Date_Reported_dayofmonth': Date_Reported_dayofmonth,
            'Date_Occurred_month': Date_Occurred_month,
            'Date_Occurred_dayofweek': Date_Occurred_dayofweek,
            'Date_Occurred_dayofmonth': Date_Occurred_dayofmonth,
            'Premise_Code': Premise_Code,
            'Area_Name': Area_Name,
            'Victim_Sex': Victim_Sex,
            'Victim_Descent': Victim_Descent,
            'Status_Description': Status_Description,
            'Location': Location,
            'Modus_Operandi': Modus_Operandi,
            'Weapon_Used_Code': Weapon_Used_Code,
            'Part 1-2': Part_1_2
        }])

        # Ensure all columns are in the expected format
        input_data['Victim_Age'] = pd.to_numeric(input_data['Victim_Age'], errors='coerce')
        input_data['Reporting_District_no'] = input_data['Reporting_District_no'].astype(float)
        input_data['hour'] = input_data['hour'].astype(int)
        input_data['Premise_Code'] = input_data['Premise_Code'].astype(float)
        input_data['Weapon_Used_Code'] = input_data['Weapon_Used_Code'].astype(float)
        input_data['Part 1-2'] = input_data['Part 1-2'].astype(float)

        # Transform categorical and numerical data using SimpleImputer
        cat_cols = input_data.select_dtypes(include='object').columns
        num_cols = input_data.select_dtypes(exclude='object').columns

        transformer = ColumnTransformer([
            ('num_cols', SimpleImputer(strategy='mean'), num_cols),  # Numerical columns
            ('cat_cols', SimpleImputer(strategy='most_frequent'), cat_cols)  # Categorical columns
        ], verbose_feature_names_out=False)

        # Fit and transform the data
        transformer.fit(input_data)
        input_clean = transformer.transform(input_data)

        # Convert to DataFrame for consistency
        input_clean_df = pd.DataFrame(input_clean, columns=transformer.get_feature_names_out())

        # Predict the output using the model
        prediction = model.predict(input_clean_df)
        
        # Map the predicted category to the actual crime description
        predicted_category = crime_mapping.get(int(prediction[0]), "Unknown Category")
        output = f"Predicted Crime Category: {predicted_category}"

    except ValueError as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)

