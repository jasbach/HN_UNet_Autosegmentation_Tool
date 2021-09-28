# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:56:15 2020

@author: johna

To run on local server (not production level):
Open Anaconda PowerShell Prompt
Navigate to folder containing the app

$env:FLASK_APP="app.py"
flask run --without-threads


Alternately, run this script in any Python IDE. Line 136 allows it to be run directly this way.
"""

import os
import datetime

import pydicom

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import main_script

app = Flask(__name__)
app.secret_key = 'jessalee'
UPLOAD_FOLDER = app.root_path + '\\uploadfiles'
OUTPUT_FOLDER = app.root_path +'\\generatedfiles'
TODAY = datetime.date.today()
ALLOWED_EXTENSIONS = {'dcm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main_menu():
    return render_template('mainpage.html')

@app.route('/login', methods=['GET','POST'])
def user_redirect():
    if request.method == 'POST':
        if request.form['username'] == "":
            flash('No username entered!')
            return redirect(url_for('main_menu'))
        app.config['USERNAME'] = request.form['username']
        app.config['UPLOAD_FOLDER'] = os.path.join(UPLOAD_FOLDER,app.config['USERNAME'])
        app.config['OUTPUT_FOLDER'] = os.path.join(OUTPUT_FOLDER,app.config['USERNAME'])
        print("Username:",app.config['USERNAME'])
        print("Input Path:",app.config['UPLOAD_FOLDER'])
        print("Output Path:",app.config['OUTPUT_FOLDER'])
        if request.form['process'] == "new":
            return redirect(url_for('upload_files',username=app.config['USERNAME']))
        elif request.form['process'] == "last":
            assert len(os.listdir(app.config['OUTPUT_FOLDER'])) == 1 #<---- need better system
            for file in os.listdir(app.config['OUTPUT_FOLDER']):
                app.config['FILENAME'] = file
            return redirect(url_for('retrieve_ss',username=app.config['USERNAME'])) #<------build
    return redirect(url_for('main_menu'))

@app.route('/<username>/retrieve_ss', methods=['GET','POST'])
def retrieve_ss(username):
    filename = app.config['FILENAME']
    flash('Found file {}'.format(filename))
    return render_template('retrieve_ss.html')#,date_created=date_created)

@app.route('/<username>/uploadform')
def upload_files(username):
    return render_template('uploadform.html',username=username)

@app.route('/upload_files', methods=['GET','POST'])
def contour():
    #main menu provides user space to upload files, these files will be passed to the contour method here
    if request.method == 'POST':
        if 'dicom_files' not in request.files:
            flash('No file part')
            return redirect(url_for('main_menu'))
        file = request.files['dicom_files']
        if file.filename == '':
            flash('No file selected!')
            return redirect(url_for('main_menu'))
        files = request.files
        
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])
        if not os.path.exists(app.config['OUTPUT_FOLDER']):
            os.mkdir(app.config['OUTPUT_FOLDER'])
        
        for file in os.listdir(app.config['UPLOAD_FOLDER']): #ensure folders are empty
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            os.remove(os.path.join(app.config['OUTPUT_FOLDER'],file))
            
        filelist = files.getlist('dicom_files')
        for file in filelist:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        
        invalid_files = 0
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],file)
            try:
                pydicom.dcmread(filepath)
            except:
                invalid_files += 1
                os.remove(filepath)
        if invalid_files > 1:
            flash("%d files could not be validated as DICOM and were removed." % invalid_files)
        return render_template('upload_files.html', number_of_files=len(os.listdir(app.config['UPLOAD_FOLDER'])), uploaded=True)
    
    return render_template('upload_files.html', number_of_files=len(os.listdir(app.config['UPLOAD_FOLDER']), uploaded=False))
                       
@app.route('/create_ss',methods=['GET','POST'])
def create_ss():
    main_script.generate_ss(app.config['UPLOAD_FOLDER'],app.config['OUTPUT_FOLDER'],app.config['USERNAME'])
    filename = "RS." + app.config['USERNAME'] + "-CNN.dcm"
    app.config['FILENAME'] = filename
    return render_template('create_ss.html', number_of_files=len(os.listdir(app.config['UPLOAD_FOLDER'])))

@app.route('/contour_download/', methods=['GET','POST'])
def download_redirect():
    return redirect(url_for('serve_ss',filename=app.config['FILENAME']))

@app.route('/contour_download/<path:filename>', methods=['GET','POST'])
def serve_ss(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'],filename=filename)

if __name__ == '__main__':
    app.run(threaded=False)
