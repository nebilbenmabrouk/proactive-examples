#!/usr/bin/env python3

# import os, sys, bz2, uuid, pickle, json, connexion, wget
# import numpy as np
import os
import sys
import glob
import signal
import json
import connexion
import subprocess
import numbers
import pandas as pd
import argparse
import psutil
import utils
import wget
import zipfile
import flask
import requests
import numpy as np
import shutil
#import datetime
#datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

from cryptography.fernet import Fernet
from datetime import datetime as dt
from os.path import join, exists, isfile
from binascii import hexlify
from shutil import move
from flask_cors import CORS
from joblib import load
from flask import jsonify
from tempfile import TemporaryFile

from scipy.stats import norm
from scipy.stats import wasserstein_distance
from urllib.parse import quote
from distutils.util import strtobool


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# General parameters
REST_API_PORT = 8501
MODEL_BASE_PATH = "/tmp"
APP_BASE_DIR = "/model_as_service"
UPLOAD_MODELS_FOLDER = "/model_as_service"
PROMETHEUS_SERVICE_PORT = 9091
MODEL_CONFIG_FILE = "models.config"
MODEL_FILE_EXT = '.model'
META_FILE_EXT = '.meta'
CURRENT_META_FILE = join(UPLOAD_MODELS_FOLDER, 'model_last' + META_FILE_EXT)    # default meta path
TRACE_FILE = join(UPLOAD_MODELS_FOLDER, 'trace.txt')  # default trace file
CONFIG_FILE = join(UPLOAD_MODELS_FOLDER, 'config.json')  # default config file
PREDICTIONS_FILE = join(UPLOAD_MODELS_FOLDER, 'predictions.csv')  # default predictions file
TOKENS = {
    'user': hexlify(os.urandom(16)).decode(),  # api key
    'test': hexlify(os.urandom(16)).decode()
}  # user tokens


DEBUG_ENABLED = True
TRACE_ENABLED = False
DRIFT_ENABLED = False
DRIFT_THRESHOLD = None
DRIFT_NOTIFICATION = False
LOG_PREDICTIONS = True
HTTPS_ENABLED = False
USER_KEY = "user"
assert USER_KEY is not None, "USER_KEY is required!"
USER_KEY = str(USER_KEY).encode()

# Decrypt user credentials
USER_DATA_FILE = join(APP_BASE_DIR, 'user_data.enc')
with open(USER_DATA_FILE, 'rb') as f:
    encrypted_data = f.read()
fernet = Fernet(USER_KEY)
decrypted_data = fernet.decrypt(encrypted_data)
message = decrypted_data.decode()
user_credentials = json.loads(message)


# Get proactive server url
proactive_rest = user_credentials['ciUrl']
proactive_url = proactive_rest[:-5]

# Check if there is already a configuration file
if not isfile(CONFIG_FILE):
    print("Generating the configuration file")
    config = {
        'DEBUG_ENABLED': DEBUG_ENABLED,
        'TRACE_ENABLED': TRACE_ENABLED,
        'DRIFT_ENABLED': DRIFT_ENABLED,
        'DRIFT_THRESHOLD': DRIFT_THRESHOLD,
        'DRIFT_NOTIFICATION': DRIFT_NOTIFICATION,
        'LOG_PREDICTIONS': LOG_PREDICTIONS,
        'HTTPS_ENABLED': HTTPS_ENABLED
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    print("Done")


# ----- Helper functions ----- #


def get_config(param, default_value=None):
    # print("Loading parameters from the configuration file")
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    if param in config:
        return config[param]
    else:
        return default_value


def set_config(param, value):
    # print("Writing to the configuration file")
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    config[param] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def trace(message, token=""):
    if get_config('TRACE_ENABLED'):
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        with open(TRACE_FILE, "a") as f:
            f.write("%s|%s|%s\n" % (datetime_str, token, message))


def log(message, token=""):
    trace(message, token)
    if get_config('DEBUG_ENABLED'):
        datetime_str = dt.today().strftime('%Y-%m-%d %H:%M:%S')
        print(datetime_str, token, message)
    return message

def submit_workflow_from_catalog(bucket_name, workflow_name, workflow_variables={}, token=""):
    result = False
    try:
        log("Connecting on " + proactive_url, token)
        gateway = proactive.ProActiveGateway(proactive_url, [])
        gateway.connect(
            username=user_credentials['ciLogin'],
            password=user_credentials['ciPasswd'])
        if gateway.isConnected():
            log("Connected to " + proactive_url, token)
            try:
                log("Submitting a workflow from the catalog", token)
                jobId = gateway.submitWorkflowFromCatalog(bucket_name, workflow_name, workflow_variables)
                assert jobId is not None
                assert isinstance(jobId, numbers.Number)
                workflow_path = bucket_name + "/" + workflow_name
                log("Workflow " + workflow_path + " submitted successfully with jobID: " + str(jobId), token)
                result = True
            finally:
                log("Disconnecting from " + proactive_url, token)
                gateway.disconnect()
                log("Disconnected from " + proactive_url, token)
                gateway.terminate()
                log("Connection finished from " + proactive_url, token)
        else:
            log("Couldn't connect to " + proactive_url, token)
    except Exception as e:
        log("Error while connecting on " + proactive_url, token)
        log(str(e), token)
    return result


def submit_web_notification(message, token):
    return submit_workflow_from_catalog("notification-tools", "Web_Notification", {'MESSAGE': message}, token)


def auth_token(token):
    for user, key in TOKENS.items():
        if key == token:
            return True
    return False


def get_token_user(token):
    for user, key in TOKENS.items():
        if key == token:
            return user
    return None

# ----- REST API endpoints ----- #


def get_token_api(user) -> str:
    user = connexion.request.form["user"]
    addr = connexion.request.remote_addr
    token = TOKENS.get(user)
    if not token:
        log('Invalid user: {u} ({a})'.format(u=user, a=addr))
        return "Invalid user"
    else:
        log('{u} token is {t} ({a})'.format(u=user, t=token, a=addr))
        return token


def predict_api(api_token, model_name, image) -> str:
    api_token = connexion.request.form['api_token']
        #if model_version is empty, model_version will be set on None

    if auth_token(api_token):
        model_name = connexion.request.form['model_name']
        image = connexion.request.files["image"]
        log("calling predict_api", api_token)
        image.save('/model_as_service/sample_image.png')

        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = utils.get_biggest_deployed_model_version(model_name)
            pass

        model_version_status = utils.check_model_name_version(model_name, model_version)

        if model_version_status == "version deployed":
            try:
                #data_preprocessing
                img = utils.load_image('/model_as_service/sample_image.png')
        
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

                data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})

                headers = {"content-type": "application/json"}
                prediction_link = "http://localhost:8501/v1/models/"+model_name+"/versions/"+str(model_version)+":predict"
                json_response = requests.post(prediction_link, data=data, headers=headers)
                predictions = json.loads(json_response.text)['predictions']
                return class_names[np.argmax(predictions[0])]
                #return np.argmax(predictions[0])
            except Exception as e:
                return log(str(e), api_token)
        else:
            return model_version_status
    else:
        return log("Invalid token", api_token)


def deploy_api(model_name, model_file) -> str:
    os.environ['REST_API_PORT'] = "8501"
    os.environ['MODEL_CONFIG_FILE'] = "/model_as_service/models.config"
    os.environ['CONFIG_FILE_POLL_SECONDS'] = "30"
    os.environ['MODELS_PATH'] = "/tmp"
    model_zip_path = os.environ['MODELS_PATH']+"/new_model.zip"
    api_token = connexion.request.form["api_token"]
    model_name = connexion.request.form["model_name"]
    append = connexion.request.form["append"]
    model_file = connexion.request.files["model_file"]
    model_download_path = os.environ['MODELS_PATH'] + "/" + model_name
    download_model = True

    #if model_version is empty, model_version will be set on None
    try:
        model_version = int(connexion.request.form["model_version"])
    except:
        model_version = None
        pass

    log("Calling deploy_api", api_token)
    if auth_token(api_token):
        #Service Status Management
        tensorflow_model_server_down = True
        for proc in psutil.process_iter():
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            if (pinfo['name']=="tensorflow_model_server"):
                print("[INFO] TensorFlow model server is already up")
                tensorflow_model_server_down = False
        if tensorflow_model_server_down:
            print("[INFO] Starting a new tensorflow_model_server")
            tf_server = subprocess.Popen(["tensorflow_model_server "
                                "--rest_api_port=$REST_API_PORT "
                                "--model_config_file_poll_wait_seconds=$CONFIG_FILE_POLL_SECONDS "
                                "--model_config_file=$MODEL_CONFIG_FILE > server_test.log 2>&1"],
                                stdout=subprocess.DEVNULL,
                                shell=True,
                                preexec_fn=os.setsid)

        #Model Versionning Management
        # if model_version was not specified, it will be set by default as "the latest model_version number + 1"
        if model_version is None:
            if not os.path.exists(model_download_path):
                os.makedirs(model_download_path)
                model_version = 1
            else:
                listOfFile = os.listdir(model_download_path)
                model_versions = []
                for file in listOfFile:
                    file_casted = file
                    try:
                        file_casted = int(file)
                        model_versions.append(file_casted)
                    except:
                        pass
                #check if the model directory is empty or not
                if not model_versions:
                    model_version = 1
                else:
                    model_versions.sort()     
                    model_version = model_versions[-1]+1
            print("[INFO] new version to be deployed : ",model_version)
        else:
            version_path = model_download_path + "/" + str(model_version)
            if os.path.isdir(version_path):
                download_model = False
                print("[WARN] This model version already exists. The uploaded model version will be ignored. The existing version will be deployed.")
        #Model Downloading
        #if the specified model version doesn't exist in the directory, the zip file uploaded by the user will be downloaded
        if download_model:
            print("[INFO] Downloading the new model in ", model_file)
            model_file.save(model_zip_path)
            with zipfile.ZipFile(model_zip_path,"r") as zip_ref:
                version_path = model_download_path + "/" + str(model_version)
                zip_ref.extractall(version_path)
                os.remove(model_zip_path)
        #Model Deployment
        print("[INFO] Deploying the version ",model_version, "of test_model")
        if append == "true":
            deployment_status = utils.append_version_model_service_config(model_name, model_download_path, model_version)
        else:
            deployment_status = utils.add_version_model_service_config(model_name, model_download_path, model_version)
        #print("The new tensorflow model file was deployed successfully at: ",os.environ['MODELS_PATH'], model_version)
        return deployment_status
    else:
        return log("Invalid token", api_token)


def list_deployed_models_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling list_models_api", api_token)
    if auth_token(api_token):
        models_config_list = utils.read_config_file()
        log("List of deployed models:\n" + str(models_config_list))
        return str(models_config_list)
    else:
        return log("Invalid token", api_token)


def undeploy_model_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling undeploy_api", api_token)
    if auth_token(api_token):
        model_name = connexion.request.form["model_name"]
        #if model_version is empty, model_version will be set on None
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
        status = utils.delete_version_model_service_config(model_name,model_version)
        log("Model removed:\n" + str(model_name))
        return log(status)
    else:
        return log("Invalid token", api_token)


def redeploy_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("calling redeploy_api", api_token)
    if auth_token(api_token):
        log("Calling redeploy_api", api_token)
    if auth_token(api_token):
        #Service Status Management
        tensorflow_model_server_down = True
        for proc in psutil.process_iter():
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            if (pinfo['name']=="tensorflow_model_server"):
                print("[INFO] TensorFlow model server is up")
                tensorflow_model_server_down = False
        if tensorflow_model_server_down:
            print("[INFO] Starting a new tensorflow_model_server")
            tf_server = subprocess.Popen(["tensorflow_model_server "
                                "--rest_api_port=$REST_API_PORT "
                                "--model_config_file_poll_wait_seconds=$CONFIG_FILE_POLL_SECONDS "
                                "--model_config_file=$MODEL_CONFIG_FILE > server_test.log 2>&1"],
                                stdout=subprocess.DEVNULL,
                                shell=True,
                                preexec_fn=os.setsid)
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
        pass
        model_name = connexion.request.form["model_name"]
        append = connexion.request.form["append"]
        model_download_path = os.environ['MODELS_PATH'] + "/" + model_name
        #Model Versionning Management
        # if model_version was not specified, it will be set by default as "the latest model_version number + 1"
        if model_version is None:
            if not os.path.exists(model_download_path):
                deployment_status = "There is no model downloaded with this name" + model_name +". Please choose an already downloaded model." 
            else:
                listOfFile = os.listdir(model_download_path)
                model_versions = []
                for file in listOfFile:
                    file_casted = file
                    try:
                        file_casted = int(file)
                        model_versions.append(file_casted)
                    except:
                        pass
                #check if the model directory is empty or not
                if not model_versions:
                    model_version = 1
                else:
                    model_versions.sort()     
                    model_version = model_versions[-1]
                    version_path = model_download_path + "/" + str(model_version)
            print("[INFO] the version that will be redeployed is : ",model_version)
        else:
            version_path = model_download_path + "/" + str(model_version)
            if not os.path.isdir(version_path):
                deployment_status = "This version path doesn't exist: " + version_path +". Please choose an already downloaded model." 
        #Model Deployment
        print("[INFO] Redeploying the version ",model_version, "of test_model")
        if append == "true":
            deployment_status = utils.append_version_model_service_config(model_name, version_path, model_version)
        else:
            deployment_status = utils.add_version_model_service_config(model_name, version_path, model_version)
        #print("The new tensorflow model file was deployed successfully at: ",os.environ['MODELS_PATH'], model_version)
        return deployment_status
    else:
        return log("Invalid token", api_token)

def download_model_config_api() -> str:
    api_token = connexion.request.form["api_token"]
    log("downloading model config", api_token)
    if auth_token(api_token):
        return flask.send_from_directory(directory='/model_as_service', filename='models.config',as_attachment=True)
    else:
        return log("Invalid token", api_token)

def upload_model_config_api() -> str:
    os.environ['MODEL_CONFIG_FILE'] = "/model_as_service/models.config"
    api_token = connexion.request.form["api_token"]
    model_config_file = connexion.request.files["model_config_file"]
    if auth_token(api_token):
        model_config_file.save(os.environ['MODEL_CONFIG_FILE'])
        return log("Model config file was successefuly uploaded")
    else:
        return log("Invalid token", api_token)

#apt-get install tree
def list_dowloaded_models() -> str:
    api_token = connexion.request.form["api_token"]
    log("Displaying the list of downloaded models", api_token)
    if auth_token(api_token):
        os.environ['MODELS_PATH'] = "/tmp"
        model_download_path = 'tree '+str(os.environ['MODELS_PATH'])
        tree_model_download_path = os.popen(model_download_path).read()
        return tree_model_download_path
    else:
        return log("Invalid token", api_token)

def clean_downloaded_models(model_name) -> str:
    api_token = connexion.request.form["api_token"]
    log("Cleaning the downloaded models", api_token)
    if auth_token(api_token):
        model_name = connexion.request.form["model_name"]
        os.environ['MODELS_PATH'] = "/tmp"
        model_path = os.environ['MODELS_PATH'] + "/" + model_name
        try:
            model_version = int(connexion.request.form["model_version"])
        except:
            model_version = None
            pass
        if not utils.check_deployed_model_name_version(model_name,model_version):
            if not os.path.exists(model_path):
                clean_status = "Model folder " + str(model_path) + " doesn't exist."
            elif model_version is None:
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    clean_status = "Model folder " + str(model_path) + " was successefuly deleted."
            else:
                model_version_path = model_path + "/" + str(model_version)
                if os.path.exists(model_version_path):
                    shutil.rmtree(model_version_path)
                    clean_status = "Model version folder " + str(model_path) + "/"+ str(model_version) + " was successefuly deleted."
                else:
                    clean_status = "Model version folder " + str(model_path) + "/"+ str(model_version) + " doesn't exist."
        else:
            if model_version is None:
                clean_status = "The model " + model_name + " is deployed. To be able to delete it, you should undeploy it first."
            else :
                clean_status = "The version " + str(model_version) + " of the model " + model_name + " is deployed. To be able to delete it, you should undeploy it first."
    else:
        return log("Invalid token", api_token)
    return clean_status

#TODO_LISR
# predict array as input (took me alot of time 2nd sprint)
# predict version empty accepted (done)
#specify clean status endpoint (partially done) (to be tested)
#List available versions of a specific model_name (directory centric) (done)
# redeploy endpoint  
#synch .config and .bin file (2) (blocked) (took alot of time 2nd sprint)
#stop the TF Service (check pid) (can be done)

#clean_data endpoint




# ----- Main entry point ----- #
if __name__ == '__main__':
    #logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090, help="set the port that will be used to deploy the service")
    args = parser.parse_args()
    app = connexion.FlaskApp(__name__, port=args.port, specification_dir="/model_as_service")
    CORS(app.app)
    app.add_api('dl_service-api.yaml', arguments={'title': 'Deep Learning Model Service'})
    
    app.run(debug=DEBUG_ENABLED)