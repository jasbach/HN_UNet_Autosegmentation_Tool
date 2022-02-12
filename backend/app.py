# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:15 2020

@author: johna

To run on local server (not production level):
Open Anaconda PowerShell Prompt
Navigate to folder containing the app

$env:FLASK_APP="app.py"
flask run --without-threads


Alternately, run this script in any Python IDE. 
Final allows it to be run directly this way.
"""

import os
import datetime
import threading
from urllib import response

import pydicom

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import main_script
# main_script handles all deep learning backend functions

# ==== Define settings referenced throughout the app ====
app = Flask(__name__)
CORS(app,origins=['http://localhost:3000'],methods=['GET','POST','OPTIONS','PUT','DELETE'],send_wildcard=True)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = 'jessalee'
UPLOAD_FOLDER = app.root_path + '\\uploadfiles'
OUTPUT_FOLDER = app.root_path +'\\generatedfiles'
TODAY = datetime.date.today()
ALLOWED_EXTENSIONS = {'dcm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    # function that checks to ensure only DICOM files are uploaded to the app
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/files', methods=['GET','POST','DELETE'])
def files():
    if request.method == 'GET':
        file_list = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({'files':file_list})

    if request.method == 'POST':
        app.logger.info(request.files)
        files = request.files
        for k,file in files.items():
            #if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return app.response_class(status=201,response="files saved")
    if request.method == 'DELETE':
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            os.remove(os.path.join(app.config['OUTPUT_FOLDER'],file))
        return app.response_class(status=200)

@app.route('/api/cleanup',methods=['DELETE'])
def delete_files():
    if request.method == "DELETE":
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            os.remove(os.path.join(app.config['OUTPUT_FOLDER'],file))
        return app.response_class(status=200,response="files deleted")
        

if __name__ == '__main__':
    app.run(debug=True)