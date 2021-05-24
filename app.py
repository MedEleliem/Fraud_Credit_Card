from flask import  Flask, request, jsonify, render_template
import pickle 
import numpy as np

app = Flask(__name__)
model = pickle.load(open('results/logit.pkl', 'rb'))
finalvar = pickle.load(open('results/finalvar.pkl', 'rb'))


    
@app.route('/')
def home():
    return render_template('index.html', var_text = finalvar)

@app.route('/predict', methods = ['POST'])

def predict():
    """ For Rendering results on HTML"""
    def flatten(l):
      try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
      except IndexError:
        return []
    l = [ x.strip().split("\n")  for x in request.form.values() ]
    val = flatten(l)
    val = [float(i) for i in val]
    val.insert(0,1)
    prediction_unrounded = model.predict(val)
    prediction = map(round, prediction_unrounded)
    output = prediction
    if output == 0 :
        x = "Non Fraud"
    else :
        x = "Fraud"
   
    return render_template('index.html', prediction_text='This Credit card is {}'.format(x))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """For Direct API calls through request"""
    data = request.get_json(force=True)
    prediction_unrounded = model.predict(np.array(list(data.values())))
    prediction = list(map(round, prediction_unrounded))
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__" :
    app.run(debug = True)