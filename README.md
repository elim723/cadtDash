# cadtDash

cadtDash is a Python library that visualizes clinical data via a web interface.

## Installation

This tool is deveoped using Python 3.9.4. A requirement.txt is provided to set up a virtual environment for this tool. Once you install python, set up the virtual environment.

```bash

## 1. Click the above green “Code” button, then click “Download Zip”.

## 2. Extract that downloaded Zip file to some folder.  In that folder there will be a folder called cadtDash-main.  That will be our working folder.

## 3. Start an Anaconda prompt.

## 4. In that Anaconda prompt change directories to that cadtDash-main folder:
cd  pathtoextractedFolder\cadtDash-main

## 5. Create a new virtual environment
python3 -m venv .

## 2. Source the environment
.\Scripts\activate
# source bin/activate # if Linux/bash

## 3. Install all required packages
pip install -r requirements.txt 
```
The last step may take awhile.  If the installation hangs, you may have to press the enter key.

## Usage

First, put your file "ExampleData.xlsx" in the pathtoextractedFolder\cadtDash-main folder.
This Excel file must have three sheets in the following orders: 
- 'CT PE Studies' from PACS
- Aidoc Studies from AIDoc
- mPower from mPower

Then, run the python script.
```bash

python extractFromMPower.py
```

A message will pop up.
```bash 
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'extractFromMPowerData' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```

Open a webbrowser (preferably Edge or Chrome) and enter the URL in the message. For example, the above message says the URL is "http://127.0.0.1:8050/".

