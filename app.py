from flask import Flask, jsonify,request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

def get_cleaned_data(fd):
    
    ges = float(fd["gestation"])
    parity = 1 if fd["parity"]=='on' else 0
    age = float(fd["age"])
    height = float(fd["height"])
    weight = float(fd["weight"])
    smoke = 1 if fd["smoke"] else 0
    
    cleaned_data = {
            "gestation": [ges],
            "parity":[parity],
            "age": [age],
            "height":[height],
            "weight": [weight],
            "smoke":[smoke]
    
    }
    print(cleaned_data)
    return cleaned_data


@app.route('/')
def home():
    # return "<h1>Welcome to the ML Model for prediction Baby's birth weight affecting factors. </h1>"
    return render_template('index.html')

# define your endpoints here
@app.route('/predict',methods=["POST"])
def get_prediction():
    #get data from user form

    baby_data_form = request.form
    baby_data_json = get_cleaned_data(baby_data_form)

    baby_df = pd.DataFrame(baby_data_json)

    # Loading the Pickle file - Deserializing the trained model using pickle.load(<model.pkl>) 
    # print("current dir:", os.getcwd())

    model_path = os.path.join(os.getcwd(), "model.pkl")


    with open(model_path,"rb") as obj:
        mymodel = pickle.load(obj)


    #make prediction with user data.
    prediction = mymodel.predict(baby_df)
    model_prediction  = round(float(prediction),2)

    #Return respoinse in json format
    response = f"Prediction of Birth weight of the baby: {model_prediction}"
    return render_template('index.html', prediction =response)

if __name__ == '__main__':
    app.run(debug=True)