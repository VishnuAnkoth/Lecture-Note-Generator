import os
from flask import Flask, flash, request, render_template, redirect, url_for, make_response,send_file
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle

from io import BytesIO

from transformers import pipeline
import speech_recognition as sr

summarizer = pipeline("summarization")
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist

nltk.download("punkt")
nltk.download("stopwords")
import whisper



app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "upload"
app.config["ALLOWED_EXTENSIONS"] = {"mp3"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]
    )


def summarize_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename, fp16=False)
    article = result["text"]
    sentences = sent_tokenize(article)
    words = word_tokenize(article)

    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.casefold() not in stop_words and word.isalpha()
    ]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    fdist = FreqDist(stemmed_words)

    #top_words = [pair[0] for pair in fdist.most_common(10)]
    top_words = [pair[0] for pair in fdist.most_common(5)]

    summary = []
    for sentence in sentences:
        for word in top_words:
            if word in sentence:
                summary.append(sentence)
                break

    return summary


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            summary = summarize_audio(
                os.path.join(app.config["UPLOAD_FOLDER"], filename)
            )
            return render_template("index.html", summary=summary)
    return render_template("index.html")


#@app.route("/summary")
#def summary():
    #summary = request.args.get("summary")
    #return render_template("summary.html", summary=summary)

@app.route('/download_summary/<summary>')
def download_summary(summary):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=inch, rightMargin=inch)
    style = ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=12, leading=16)
    p = Paragraph(summary, style=style)
    doc.build([Spacer(1, 2*inch), p])
    response = make_response(buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=summary.pdf'
    response.headers['Content-Type'] = 'application/pdf'
    return response

if __name__ == "__main__":
    app.run(debug=True)
