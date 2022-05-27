# cadtDash

cadtDash is a Python library that visualize clinical data via a web interface.

## Installation

This tool is deveoped using Python 3.9.4. A requirement.txt is provided to set up a virtual environment for this tool. Once you install python, set up the virtual environment.

```bash

## 1. Create a new virtual environment
python3 -m venv /path/to/new/virtual/environment

## 2. Source the environment
cd /path/to/new/virtual/environment
source bin/activate
# if windows:
#.\Scripts\activate

## 3. Install all required packages
pip install -r /path/to/cadtDash/requirements.txt 
```
The last step may take awhile.

## Usage

First, run the python script.
```bash

python /path/to/cadtDash/extractFromMPower.py
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

