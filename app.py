from flask import Flask, request
import numpy as np
import joblib
import flasgger
from flasgger import Swagger
from pickle import load

app=Flask(__name__)
Swagger(app)

loaded_model = load(open('best_model.pkl', 'rb'))
poly = load(open('poly.pkl', 'rb'))
sc = load(open('scalar.pkl', 'rb'))


@app.route('/',methods=["Get"])
def predict():

    """House Price Prediction
    This is using docstrings for specifications.
    ---
    parameters:
        - name: House Age
          in: query
          type: number
          description: "Enter age of the house"
          required: true
        - name: Distance_to_the_nearest_MRT_station
          in: query
          type: number
          description: ""
          required: true
        - name: number_of_convenience_stores
          in: query
          type: number
          description: ""
          required: true
        - name: Latitude
          in: query
          type: number
          description: ""
          required: true
        - name: Longitude
          in: query
          type: number
          description: ""
          required: true
    responses:
          200:
              description: The output values
    """
    l=[]
    i1=request.args.get('House Age')
    print('1')
    l.append(i1)
    i2=request.args.get('Distance_to_the_nearest_MRT_station')
    print('2')
    l.append(i2)
    i3=request.args.get('number_of_convenience_stores')
    l.append(i3)
    i4=request.args.get('Latitude')
    l.append(i4)
    i5=request.args.get('Longitude')
    l.append(i5)
    arr = np.array([l])
    print('3')
    arr = poly.transform(arr)
    print('4')
    scaled_arr = sc.transform(arr)
    print('5')
    p = round(loaded_model.predict(scaled_arr)[0][0],2)
    print('6')
    return "Price of the house per unit area: "+str(p)









if __name__=='__main__':
    app.run()
