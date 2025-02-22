# Medicine Agents

This repo contains the code to run multi-agent system.


## Running the code

## Deploying

- Go to [Groq console](https://console.groq.com/playground) and register for an account and create a free API key
- Fork the repo on GitHub
- On [Streamlit](https://share.streamlit.io) and sign in with GitHub
- On the top right corner click Create App
- Click Deploy a public app from GitHub
- Select the repo from your account
- Branch is `main`
- Main file path is `main.py`
- Set the App URL you want
- IMPORTANT: In main.py, go to line 43 and replace
- ```python
  client = Groq(api_key="ENTER_YOUR_OWN_API_KEY")
- Click save
- Click Deploy


## Creating the database to run

Here's an example script that can create the SQLite database assuming the following directory structure

```text
data
|____patients
| |____p001
| | |____pathology_report.txt
| | |____biomarker_report.txt
| | |____imaging_report.txt
| |____p002
| | |____pathology_report.txt
| | |____biomarker_report.txt
| | |____imaging_report.txt
```



```python
import os
from dataclasses import dataclass, field

import sqlite3


@dataclass
class Patient:
    patient_id: str
    biomarker_report: str
    imaging_report: str
    pathology_report: str


def load_patient_data(patient_id: str) -> Patient:
    biomarker_path = os.path.join('data', 'patients', patient_id, 'biomarker_report.txt')
    imaging_path = os.path.join('data', 'patients', patient_id, 'imaging_report.txt')
    pathology_path = os.path.join('data', 'patients', patient_id, 'pathology_report.txt')
    with open(biomarker_path, 'r') as f:
        biomarker_report = f.read()
    with open(imaging_path, 'r') as f:
        imaging_report = f.read()
    with open(pathology_path, 'r') as f:
        pathology_report = f.read()
    return Patient(patient_id, biomarker_report, imaging_report, pathology_report)


patients = os.listdir(os.path.join('data', 'patients'))


conn = sqlite3.connect('patients.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS patients
             (id text, biomarker text, imaging text, pathology text)''')

for patient_id in patients:
    patient = load_patient_data(patient_id)
    c.execute("INSERT INTO patients VALUES (?, ?, ?, ?)", (patient_id, patient.biomarker_report, patient.imaging_report, patient.pathology_report))

conn.commit()
conn.close()


```
