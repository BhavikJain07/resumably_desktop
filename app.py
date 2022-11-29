from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import json
import static.ResumablyFinal as R
from flaskwebgui import FlaskUI
##Variables
app = Flask(__name__)
CORS = CORS(app)
CONFIG = "static/model"
WEIGHTS = "static/model_v2.onnx"

##App Routes
@app.route('/',methods=["GET","POST"])
@cross_origin()
def index():
    if request.method == "POST": #cHANGE TO POST 
        job_desc = request.form["description"]
        f = request.files["upload"]
        f.save(f.filename)
        resume_text, job_desc, _ = R.preprocess(f.filename,job_desc)
        similiarity = float(R.similarity_score(resume_text, job_desc)) 
        resume_classifier = list(R.classifier(resume_text))
        resume_classifier[0] = list(resume_classifier[0].astype("float"))
        resume_classifier[1] = list(resume_classifier[1].astype("float"))
        job_classifier = list(R.classifier(job_desc))
        job_classifier[0] = list(job_classifier[0].astype("float"))
        job_classifier[1] = list(job_classifier[1].astype("float"))
        attention = float(R.selfAttention(resume_text).argmax())
        response = {
            "resume": resume_classifier,
            "job": job_classifier,
            "similiarity": similiarity
        }
        return render_template("result.html", response=json.dumps(response))
    # if request.method == "POST":
    #     response = {'resume': [[6535.0, 5142.0, 1191.0, 12710.0, 1106.0, 20261.0, 6515.0, 19181.0, 9083.0, 28716.0], [0.682318925857544, 0.5667913556098938, 0.5514358282089233, 0.5487678050994873, 0.5444765686988831, 0.5419421792030334, 0.535834550857544, 0.5330401062965393, 0.5329717993736267, 0.5315340161323547]], 'job': [[6515.0, 6535.0, 2425.0, 12710.0, 6648.0, 6790.0, 6579.0, 19181.0, 1191.0, 15585.0], [0.6813809871673584, 0.6029932498931885, 0.5543699264526367, 0.5514858961105347, 0.5436246395111084, 0.5395663380622864, 0.535553514957428, 0.5302432775497437, 0.5243686437606812, 0.5237132906913757]], 'similiarity': 0.771192098072217}
    #     return render_template("result.html", response=json.dumps(response))
    if request.method == "GET":
        return render_template("index.html")

##Runner functions
if __name__ == "__main__":
    FlaskUI(app=app, server="flask").run()
    # app.run(
    #     debug=True,
    #     host="localhost",
    #     port=5000
    # ) 