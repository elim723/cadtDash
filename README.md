# cadtDash

cadtDash is a Python library that visualizes clinical data via a web interface.

## Installation

This tool is deveoped using Python 3.9.4. A requirement.txt is provided to set up a virtual environment for this tool. 

If this is your first time, you will need to set up a virtual environment. If you are updating the software to the latest version, you may re-use the previous environment unless new packages are included in the requirements.txt.

### First time installation

#### Cloning

There are two ways: one through the GitHub repo interface, and the other via terminal. The following steps assume a Windows operating system but can be easily converted to Unix or Linux systems.

##### Via the GitHub repo interface

1. Click the above green “Code” button, then click “Download Zip”. A cadtDash-main.zip file will be downloaded.

2. Save the zip file to your Downloads/ folder. Open a File Explorer, navigate to the Downloads/ folder and extract the cadtDash-main.zip. You may change the destination where files will be extracted. This destination will be our working folder `\you\path\to\cadtDash-main\`.

3. In the working folder, there is a sub-folder called cadtDash-main: `\you\path\to\cadtDash-main\cadtDash-main\`. This is where all the codes are located.

##### Via terminal (Windows PowerShell)

1. Click the magnifying glass in the Windows menu to search for "Windows PowerShell". A terminal will pop up.

2. Do the following command. 

```bash

## 1. Proceed to your Downloads/ folder
cd .\Downloads\

## 2. Create a sub-folder to be our working folder
mkdir cadtDash-main

## 3. Proceed to your Downloads/cadtDash-main/ folder
cd .\cadtDash-main\

## 2. Clone the cadtDash repo and rename it as cadtDash-main
git clone https://github.com/elim723/cadtDash.git cadtDash-main
```

All the codes are located in the folder `\you\path\to\cadtDash-main\cadtDash-main\`. 

#### Create virtual environment

There are two ways: one through Anaconda in Anaconda prompt, and the other using pip in Windows PowerShell. 

##### Via Anaconda

1. Click the magnifying glass in the Windows menu to search for "Anaconda prompt". A terminal will pop up.

2. Do the following commands.

``` bash
## 1. Proceed to the working folder
cd  \you\path\to\cadtDash-main\

## 2. Create a new virtual environment
#  A new folder "\you\path\to\cadtDash-main\venv" will appear.
python3 -m venv .

## 3. Source the environment
.\venv\Scripts\activate # if Windows
# source bin/activate # if Linux/bash

## 4. Install all required packages
pip install -r cadtDash-main/requirements.txt 
```

The last step may take awhile. If the installation hangs, you may have to press the enter key.

##### Via Pip

1. If you haven't had a terminal opened, click the magnifying glass in the Windows menu to search for "Windows PowerShell". A terminal will pop up.

2. Do the following commands.

``` bash
## 1. Proceed to the working folder
cd  \you\path\to\cadtDash-main\

## 2. Install virtualenv package
pip install virtualenv

## 2. Create a new virtual environment
#  A new folder "\you\path\to\cadtDash-main\venv" will appear.
#  If "virtualenv not found", it fails to recognize the path to your python executable.
#  Make sure you know where virtualenv.exe is located.
virtualenv venv

## 3. Source the environment
.\venv\Scripts\activate # if Windows
# source venv/bin/activate # if Linux/bash

## 4. Install all required packages
pip install -r cadtDash-main/requirements.txt 
```

The last step may take awhile. If the installation hangs, you may have to press the enter key.

### Update repo



## Usage

1. Put your "ExampleData.xlsx" in the the working folder `\you\path\to\cadtDash-main\`. This Excel file must have three sheets (with the exact names) in the following orders: 
- 'CT PE Studies' from PACS
- 'Aidoc Studies' from AIDoc
- 'mPower' from mPower

2. Open a terminal (either Anaconda Prompt or Windows PowerShell) and activate the virtual environment.
```bash

## 1. Proceed to the working folder
cd  \you\path\to\cadtDash-main\

## 2. Source the environment
.\venv\Scripts\activate # if Windows
# source venv/bin/activate # if Linux/bash
```

3. Run the python script `extractFromMPowerData.py`
```bash

## 1. Proceed to the sub-folder
cd  \you\path\to\cadtDash-main\cadtDash-main\

## 2. Run the python script
python .\extractFromMPowerData.py
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

4. Open a web browser (preferably Edge or Chrome) and enter the URL in the message. For example, the above message says the URL is "http://127.0.0.1:8050/". The dashboard will show up.

## Exit

To exit, simply close the web browser and the terminal.
