from flask import Flask, render_template, request
import os
from utils.rag_pipeline import summarize

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            summary = summarize(file_path)

    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True, port=5001)