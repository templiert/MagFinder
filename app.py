from flask import Flask, render_template, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
import email, smtplib, ssl
import requests
import pipeline

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
LOGIN = "tt0307842@gmail.com"
PASSWORD = "Test1234."

app = Flask(__name__,  static_url_path='', template_folder=TEMPLATES_DIR)

# TODO: put path to the Google Bucket
UPLOAD_FOLDER = "/sectionsegmentationml/bucket/wafer_dir/"  #os.path.realpath('uploads') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_wafer_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(["tif"])

def allowed_template_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(["json"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test")
def test():
    return jsonify(status="Success")

@app.route("/bucket/wafer_dir/output.json")
def result():
    return app.send_static_file(os.path.join(app.config['UPLOAD_FOLDER'], "output.json"))

def sendemail(receiver_email):
    subject = "Collectome | Pipeline Output"
    body = "In the attached files you can find the output of our BrainSegmentation Pipeline"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = LOGIN
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = ""  # TODO: put Thomas email?

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = "output.json"  # In same directory as script
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Open JSON file in binary mode
    with open(file_path, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(LOGIN, PASSWORD)
        server.sendmail(LOGIN, receiver_email, text)



@app.route('/train', methods=['POST'])
def train():
    if 'wafer_original' not in request.files:
        return "No wafer image!"
    if 'wafer_fluo' not in request.files:
        return "No wafer fluo image!"
    if 'template_init' not in request.files:
        return "No template!"
    wafer_original = request.files['wafer_original']
    wafer_fluo = request.files['wafer_fluo']
    template_init = request.files['template_init']

    if not wafer_original.filename:
        return "No wafer original file!"
    if not wafer_fluo.filename:
        return "No wafer fluo file!"
    if not template_init.filename:
        return "No template init file!"

    if wafer_original and allowed_wafer_file(wafer_original.filename) and \
        wafer_fluo and allowed_wafer_file(wafer_fluo.filename) and \
        template_init and allowed_template_file(template_init.filename):

        #wafer_original_filename = secure_filename(wafer_original.filename)
        wafer_original.save(os.path.join(app.config['UPLOAD_FOLDER'], "wafer.tif"))
        
        #wafer_fluo_filename = secure_filename(wafer_fluo.filename)
        wafer_fluo.save(os.path.join(app.config['UPLOAD_FOLDER'], "wafer_fluo.tif"))
        
        #template_init_filename = secure_filename(template_init.filename)
        template_init.save(os.path.join(app.config['UPLOAD_FOLDER'], "init_labelme.json"))
        
    else:
        raise NotImplementedError()

    pipeline.pipeline(UPLOAD_FOLDER)

    email = request.form["input_email"]
    
    if email:
        sendemail(receiver_email=email)

    output_file_path = os.path.join(os.path.relpath(UPLOAD_FOLDER), "output.json")

    return render_template("training.html", email=email, output_file_path=output_file_path)    
    

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)




