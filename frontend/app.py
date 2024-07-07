from flask import Flask, render_template, jsonify, request
import requests
import csv
import io
import time
import model
import pandas as pd
import json

app = Flask(__name__)

inference = model.ModelWrapper("../weights/v3.cbm")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/moscow_districts')
def moscow_districts():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    area[name="Москва"]->.searchArea;
    relation["admin_level"="8"](area.searchArea);
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return jsonify(data)

@app.route('/export_markers', methods=['POST'])
def export_markers():
    markers = request.json
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['latitude', 'longitude'])
    cw.writerows(markers)
    response = app.response_class(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=markers.csv'}
    )
    return response

@app.route('/import_markers', methods=['POST'])
def import_markers():
    file = request.files['file']
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    markers = [row for row in csv_input]
    return jsonify(markers[1:])  # Skip header

@app.route('/get_rate', methods=['POST'])
def get_rate():
    data = request.json
    markers = data['markers']
    dropdown1 = data['dropdown1']
    dropdown2 = data['dropdown2']
    dropdown3 = data['dropdown3']
    dropdown4 = data['dropdown4']

    ms = []
    for marker in markers:
        m=dict()
        m['lat'] = float(marker[0])
        m['lon'] = float(marker[1])
        ms.append(m)

    ms = json.dumps(ms)
    target_audience = [{'name': 'All 25-45 BC', 'gender': dropdown1, 'ageFrom': dropdown2, 'ageTo': dropdown3, 'income': dropdown4}]

    df = pd.DataFrame({"points": ms, "targetAudience": target_audience}, index=[0])

    print(df)

    rate = inference.predict(df)
    return jsonify({'rate': rate})

if __name__ == '__main__':
    app.run(debug=True)