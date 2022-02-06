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

import pydicom

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import main_script
# main_script handles all deep learning backend functions

# ==== Define settings referenced throughout the app ====
app = Flask(__name__)
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

@app.route('/')
def main_menu():
    # initial function call which returns the main page of the browser app
    return render_template('mainpage.html')

@app.route('/login', methods=['GET','POST'])
def user_redirect():
    """
    Receives username input (no constraints, any username is accepted) and
    redefines the storage folders to be user-specific.
    
    No password protection is currently implemented, and no DICOM anonymization
    exists.
    """
    
    if request.method == 'POST':
        if request.form['username'] == "":
            # if user does not fill in the field, reset form
            flash('No username entered!')
            return redirect(url_for('main_menu'))
        app.config['USERNAME'] = request.form['username']
        app.config['UPLOAD_FOLDER'] = os.path.join(UPLOAD_FOLDER,app.config['USERNAME'])
        app.config['OUTPUT_FOLDER'] = os.path.join(OUTPUT_FOLDER,app.config['USERNAME'])
        print("Username:",app.config['USERNAME'])
        print("Input Path:",app.config['UPLOAD_FOLDER'])
        print("Output Path:",app.config['OUTPUT_FOLDER'])
        if request.form['process'] == "new":
            # process to generate a new set of contours
            return redirect(url_for('upload_files',username=app.config['USERNAME']))
        elif request.form['process'] == "last":
            # process to retrieve the last set generated
            if not os.path.isdir(app.config['OUTPUT_FOLDER']):
                # if the username is not in the records
                flash("No previous files detected.")
                return redirect(url_for('main_menu'))
            
            if len(os.listdir(app.config['OUTPUT_FOLDER'])) != 1:
                # should only ever be one file in this folder, but in case something goes wrong
                flash("No valid previous files detected.")
                return redirect(url_for('main_menu'))
            
            app.config['FILENAME'] = os.listdir(app.config['OUTPUT_FOLDER'])[0]
            # sends to the structure set download page
            return redirect(url_for('retrieve_ss',username=app.config['USERNAME']))
        
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
    # main menu provides user space to upload files, these files will be passed to the contour method here
    if request.method == 'POST':
        if 'dicom_files' not in request.files:
            # if no files are attached, reroute user to the main menu
            flash('No file part')
            return redirect(url_for('main_menu'))
        file = request.files['dicom_files']
        if file.filename == '':
            flash('No file selected!')
            return redirect(url_for('main_menu'))
        files = request.files
        
        # if this is the first time a username is accessing the system, create
        # new user-specific folders for them
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])
        if not os.path.exists(app.config['OUTPUT_FOLDER']):
            os.mkdir(app.config['OUTPUT_FOLDER'])
        
        # empty the folders of any contents - system only allows storage of one instance at a time
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            os.remove(os.path.join(app.config['OUTPUT_FOLDER'],file))
            
        #filter the submitted files for name acceptability, save to server storage
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
    """
    At this point, the files are staged in the upload folder. Process can now be
    passed over to the main script, which handles the deep learning backend.
    """
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
    app.run()