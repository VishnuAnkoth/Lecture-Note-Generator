import os
from flask import Flask, flash, request, render_template, redirect, url_for, make_response,send_file
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle

import re
from io import BytesIO

from transformers import pipeline
import speech_recognition as sr

summarizer = pipeline("summarization",model="facebook/bart-large-cnn")
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

     ## Using Whisper API For Transcribing
    model = whisper.load_model("base")
    result = model.transcribe(filename, fp16=False)

    article = result["text"]

    ## Using BART To Create a Summary
    summary = summarizer(article, max_length = 300, min_length = 30, do_sample=False)[0]['summary_text']


    sumlist = summary.split('. ')
    for i in range(len(sumlist)):
        if not sumlist[i].endswith('.'):
            sumlist[i] += '.'
    print(len(sumlist))
    print(sumlist)
    return sumlist

 

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

    # style_heading = ParagraphStyle(name='Normal', fontName='Times-Roman', fontSize=18, leading=22, alignment='center')
    style = ParagraphStyle(name='Normal', fontName='Times-Roman', fontSize=14, leading=16)

    # Split summary into sentences
    sentences = summary.split("', '")

    # Reformatting the text
    sentences[0] = sentences[0][2:]
    sentences[-1] = sentences[-1][:-2]

    # Create a list of bullet points
    bullet_points = []

    # Add the heading
    heading = Paragraph('Notes', style)
    bullet_points.append(heading)
    bullet_points.append(Spacer(1, 0.5*inch))

    # Add the sentences as bullet points
    for sentence in sentences:
        bullet_points.append(Paragraph(f'• {sentence}', style))

    # Build the PDF document
    doc.build(bullet_points)

    # Prepare the response
    response = make_response(buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=demo.pdf'
    response.headers['Content-Type'] = 'application/pdf'
    return response

if __name__ == "__main__":
    app.run(debug=True)
