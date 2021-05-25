import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'V4':-1.35, 
                            'V12':-0.07, 
                            'V14':2.35, 
                            'Amount':300,
                            'V28':1.37,
                            'V8':-0.08,
                            'V11':-0.07,
                            'V20':0.36,
                            'V5':0.018})

print(r.json())