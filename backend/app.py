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
import random
from urllib import response

import pickle
import numpy as np

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

import main_script
import createdicomfile
import model
import image_prep
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

threads = {}


class UploadThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        super().__init__()
    
    def run(self,request):
        files = request.files
        files_processed = 0
        for k,file in files.items():
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            files_processed += 1
            self.progress = int((files_processed / len(files)) * 100)
            app.logger.info("Files {} percent uploaded".format(self.progress))

class CNNThread(threading.Thread):
    def __init__(self):
        self.progress = ""
        super().__init__()

    def generate_ss(self):
        OARs = ['Brainstem', 'Cochlea L','Cochlea R','Parotid L','Parotid R',
                'Submandibular L','Submandibular R','Brachial Plexus',
                'Brain','Larynx','Spinal Cord']
        imagefolder = app.config['UPLOAD_FOLDER']
        genfilesfolder = app.config['OUTPUT_FOLDER']

        image_size = 256

        patient_volume = np.load(os.path.join(genfilesfolder,'patient_volume.npy'))
        with open(os.path.join(genfilesfolder,'slice_height_map.pckl'), 'rb') as pickle_file:
            heightlist = pickle.load(pickle_file)

        filters = {"bone":[2000, 400], "tissue":[400,40],"none":[4500,1000]}
        neuralnet = model.get_unet(image_size)
        structuresetdata = []
        for i,OAR in enumerate(OARs):
            self.progress = '\n'.join(['{} complete.'.format(OARs[j]) for j in range(i)]) + '\nWorking on {}...'.format(OAR)
            if any((OAR == 'Spinal Cord', OAR == 'Brachial Plexus')):
                filter = filters['bone']
            else:
                filter = filters['tissue']

            neuralnet.load_weights(os.path.join('weights','{}.hdf5'.format(OAR.replace(" ",""))))

            filtered_patient_volume = image_prep.apply_window_level(patient_volume,filter[0],filter[1])
            prediction = neuralnet.predict(filtered_patient_volume,verbose=0)
            structuresetdata.append([OAR,prediction,heightlist]) #list of OAR name, prediction array (not yet binarized, this will happen in the create_dicom function), and the previously created height list map

        self.progress = '\n'.join(['{} complete.'.format(OARs[j]) for j in range(len(OARs))]) + '\nAll OARs complete. Structure set file ready for download.'
        patient_data,UIDdict = createdicomfile.gather_patient_data(imagefolder)
        structure_set = createdicomfile.create_dicom(patient_data,UIDdict,structuresetdata,image_size=image_size)
        SS_fileID = str(random.randint(0,10000))
        structure_set.save_as(os.path.join(app.config['OUTPUT_FOLDER'],'RS.CNN_created.{}.dcm'.format(SS_fileID)), write_like_original=False)
        return SS_fileID

@app.route('/api/threads/create', methods=['GET'])
def instantiate_thread():
    global threads
    thread_id = str(random.randint(0,10000))
    if request.args.get('threadtype') == 'upload':
        threads[thread_id] = UploadThread()
        return thread_id
    if request.args.get('threadtype') == 'neural':
        threads[thread_id] = CNNThread()
        return thread_id

@app.route('/api/files', methods=['GET','POST'])
def files():
    global threads

    if request.method == 'GET':
        file_list = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({'files':file_list})
    if request.method == 'POST':
        app.logger.info(request)
        thread_id = request.args.get('thread_id')
        threads[thread_id].run(request)
        return app.response_class(status=201,response="files saved")

@app.route('/api/files/download',methods=['GET'])
def download_ss():
    file_id = request.args.get('file_id')
    filename = "RS.CNN_created.{}.dcm".format(file_id)
    return send_from_directory(app.config['OUTPUT_FOLDER'],filename,as_attachment=True,mimetype='application/dicom')


@app.route('/api/threads/<thread_id>/progress', methods=['GET'])
def check_progress(thread_id):
    global threads
    return str(threads[thread_id].progress)

@app.route('/api/files/validate', methods=['GET'])
def validate_files():
    if request.method == 'GET':
        message = main_script.validate_files(app.config['UPLOAD_FOLDER'],app.config['OUTPUT_FOLDER'])
    return app.response_class(status=200,response=message)

@app.route('/api/inference',methods=['GET'])
def create_structure_set():
    global threads

    thread_id = request.args.get('thread_id')
    file_id = threads[thread_id].generate_ss()
    return file_id

@app.route('/api/cleanup',methods=['DELETE'])
def delete_files():
    if request.method == "DELETE":
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            os.remove(os.path.join(app.config['OUTPUT_FOLDER'],file))
        return app.response_class(status=200,response="All stored files deleted.")
        

if __name__ == '__main__':
    app.run(debug=True)