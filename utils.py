import os  # Used for handling file paths
import re  # Used for regular expressions to extract <think> tags
from dataclasses import dataclass  # Used to define structured data objects

import yaml  # Used to load AI prompts from YAML files
from graph import Prompt  # Imports the Prompt class from graph.py

# Regular expression to find <think> tags in AI responses
think_regex = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.MULTILINE)

# Data class to represent a Patient, storing their reports
@dataclass
class Patient:
    patient_id: str  # Unique identifier for the patient
    biomarker_report: str  # Report from biomarker analysis
    imaging_report: str  # Report from imaging analysis
    pathology_report: str  # Report from pathology analysis

# Data class to store AI agent prompts
@dataclass
class CrewPrompts:
    biomarker: Prompt  # Prompt for biomarker analysis agent
    imaging: Prompt  # Prompt for imaging analysis agent
    pathology: Prompt  # Prompt for pathology analysis agent
    oncologist: Prompt  # Prompt for oncologist decision-making agent

# Function to extract <think> tags from AI responses
def get_think_tag(response: str) -> str:
    think_tags = think_regex.findall(response)  # Finds all <think> tags in the response
    return "".join(think_tags)  # Joins and returns them as a string

# Function to remove <think> tags from AI responses
def remove_think_tag(response: str) -> str:
    return think_regex.sub("", response).strip()  # Removes <think> tags and trims whitespace

# Function to retrieve patient data from the SQLite database
def get_patient_data(patient_id: str, cursor) -> Patient:
    cursor.execute(
        "SELECT biomarker, imaging, pathology FROM patients WHERE id = ?",
        (patient_id,),
    )
    biomarker_report, imaging_report, pathology_report = cursor.fetchone()  # Fetches patient reports
    return Patient(patient_id, biomarker_report, imaging_report, pathology_report)  # Returns a Patient object

# Function to load patient data from an SQLite database file
def load_patient_data_from_sqlite_file(path):
    import sqlite3  # SQLite module for database interaction

    conn = sqlite3.connect(path)  # Connects to the database
    c = conn.cursor()
    c.execute("SELECT * FROM patients")  # Retrieves all patient records
    patient_ids = [row[0] for row in c.fetchall()]  # Extracts patient IDs
    
    # Creates a dictionary mapping patient IDs to their corresponding Patient objects
    patient_id_to_patient = {pid: get_patient_data(pid, c) for pid in patient_ids}
    
    c.close()  # Closes the cursor
    conn.close()  # Closes the database connection
    return patient_id_to_patient  # Returns the patient data mapping

# Function to load AI agent prompts from YAML files
def load_prompts():
    # Helper function to read a YAML file and convert it into a Prompt object
    def load_prompt(prompt_path):
        with open(prompt_path, mode="r", encoding="utf-8") as f:
            prompt_dict = yaml.safe_load(f)  # Loads YAML data into a dictionary
            return Prompt(
                text=prompt_dict["prompt"],  # Extracts prompt text
                parameters=prompt_dict["parameters"],  # Extracts required parameters
            )

    # File paths for the AI agent prompts
    biomarker_prompt_path = os.path.join("prompts", "biomarker_analyst.yaml")
    imaging_prompt_path = os.path.join("prompts", "imaging_analyst.yaml")
    pathology_prompt_path = os.path.join("prompts", "pathology_analyst.yaml")
    oncologist_prompt_path = os.path.join("prompts", "oncologist.yaml")

    # Loads the prompts from the YAML files
    biomarker_prompt = load_prompt(biomarker_prompt_path)
    imaging_prompt = load_prompt(imaging_prompt_path)
    pathology_prompt = load_prompt(pathology_prompt_path)
    oncologist_prompt = load_prompt(oncologist_prompt_path)

    # Returns an object containing all the loaded prompts
    return CrewPrompts(
        biomarker_prompt, imaging_prompt, pathology_prompt, oncologist_prompt
    )
