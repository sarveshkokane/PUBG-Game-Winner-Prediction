from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        kills = float(request.form["kills"])
        walkDistance = float(request.form["walkDistance"])
        damageDealt = float(request.form["damageDealt"])

        prediction = min(1, (kills*0.3 + walkDistance*0.0003 + damageDealt*0.002))

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
