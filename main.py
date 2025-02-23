from datetime import datetime  # Used to get the current time for logging
import tempfile  # Used to create temporary files for patient data uploads
from dataclasses import dataclass  # Used to define structured data objects (RunResult, MedicalAgent)

from groq import Groq  # API for calling the AI model
import streamlit as st  # Used for building the web-based user interface
from dotenv import load_dotenv  # Loads environment variables from a .env file

# Importing necessary modules from other files in the project
from graph import Graph, Agent, ExecutionContext, Prompt  # Handles AI agent execution
from utils import (
    get_think_tag,  # Extracts "think" tags from AI responses
    remove_think_tag,  # Cleans AI-generated responses
    load_patient_data_from_sqlite_file,  # Loads patient data from a SQLite database
    load_prompts,  # Loads AI prompts from YAML files
)

# Configures the Streamlit web page with a wide layout and a DNA emoji as the page icon
st.set_page_config(
    layout="wide",
    page_title="UVA Bot 🤖",
    page_icon=":dna:",
)

# Initializes a log list in Streamlit's session state if it doesn't exist
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to log messages with timestamps
def log(message):
    time = datetime.now().strftime("%H:%M:%S")  # Gets the current time in HH:MM:SS format
    st.session_state.logs.append(f"[{time}] {message}")  # Appends log messages to session state

# Defines the AI model name being used
model_name = "deepseek-r1-distill-qwen-32b"

# Defines a class for AI-powered agents, inheriting from the base Agent class
@dataclass(frozen=True)  # Makes the class immutable
class MedicalAgent(Agent):
    model_name: str  # Stores the name of the AI model
    client = Groq(api_key="your-api-key")  # Initializes the Groq API client

    def run(self, context: ExecutionContext) -> str:
        log(f"Running {self.name}")  # Logs which agent is running
        prompt = self._prepare_prompt(context)  # Prepares the AI prompt based on the execution context
        log(f"{self.name} Prompt: {prompt}")

        # Calls the AI model to generate a response
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0.0,  # Ensures deterministic responses (no randomness)
            top_p=1.0,
            seed=0,  # Ensures repeatability of responses
        )

        answer = chat_completion.choices[0].message.content  # Extracts AI response
        think_tag = get_think_tag(answer)  # Extracts any <think> tags
        log(f"{self.name} Think Tag: {think_tag}")  # Logs the extracted think tag
        cleaned_answer = remove_think_tag(answer)  # Removes <think> tags from AI response
        log(f"{self.name} Response: {cleaned_answer}")  # Logs the cleaned response
        context.set(self.name, cleaned_answer)  # Stores the response in the execution context
        return context  # Returns the updated execution context

# Defines a structured data object to store the final reports
dataclass
class RunResult:
    biomarker_report: str
    imaging_report: str
    pathology_report: str
    oncologist_report: str

# Main function that runs the web application
def main():
    load_dotenv()  # Loads API keys and other environment variables
    crew_prompts = load_prompts()  # Loads AI prompts for each agent

    # Helper function to create AI agents dynamically
    def agent(name: str, prompt: Prompt) -> MedicalAgent:
        return MedicalAgent(name=name, model_name=model_name, prompt=prompt)

    patient_id_to_patient = {}  # Dictionary to store patient data

    # Function to process a specific patient
    def run_patient(patient_id: str) -> RunResult:
        patient = patient_id_to_patient[patient_id]  # Retrieves patient data
        context = ExecutionContext()  # Creates an execution context for AI agents
        
        # Stores patient reports in the execution context
        context.set("BiomarkersReport", patient.biomarker_report)
        context.set("ImagingReport", patient.imaging_report)
        context.set("PathologyReport", patient.pathology_report)

        # Defines the AI processing pipeline as a graph
        graph = Graph.from_nodes(
            nodes={
                agent("biomarkers", crew_prompts.biomarker),
                agent("imaging", crew_prompts.imaging),
                agent("pathology", crew_prompts.pathology),
                agent("oncologist", crew_prompts.oncologist),
            },
            edges={
                "biomarkers -> oncologist",  # Biomarker results feed into oncology agent
                "imaging -> oncologist",  # Imaging results feed into oncology agent
                "pathology -> oncologist",  # Pathology results feed into oncology agent
            },
        )

        result = graph.run(context)  # Executes the AI agents in sequence

        # Stores the final AI-generated reports
        run_result = RunResult(
            biomarker_report=result.get("biomarkers"),
            imaging_report=result.get("imaging"),
            pathology_report=result.get("pathology"),
            oncologist_report=result.get("oncologist"),
        )
        return run_result

    # Creates a sidebar for file uploads
    st.sidebar.title(":orange[__UVA Bot__] 🤖")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())  # Saves uploaded file to a temp location
            db_path = tmp_file.name  # Gets the file path
            patient_id_to_patient = load_patient_data_from_sqlite_file(db_path)  # Loads patient data

        patient_ids = list(patient_id_to_patient.keys())  # Retrieves available patient IDs
        patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)  # UI dropdown to select a patient
        patient = patient_id_to_patient[patient_id]  # Gets selected patient's data

        # Displays patient reports in tabs
        tab1, tab2, tab3 = st.tabs(["Biomarkers Report", "Imaging Report", "Pathology Report"])

        with tab1:
            st.text(patient.biomarker_report)
        with tab2:
            st.text(patient.imaging_report)
        with tab3:
            st.text(patient.pathology_report)

        st.divider()  # Adds a UI divider

        # When submit is clicked, process the patient
        submit = st.sidebar.button("Submit")
        if submit:
            run_result = run_patient(patient_id)
            
            # Displays results in tabs
            t1, t2, t3, t4, t5 = st.tabs(["Biomarkers", "Imaging", "Pathology", "Oncologist", "Logs"])

            with t1:
                st.markdown(run_result.biomarker_report)
            with t2:
                st.markdown(run_result.imaging_report)
            with t3:
                st.markdown(run_result.pathology_report)
            with t4:
                st.markdown(run_result.oncologist_report)
            with t5:
                for log_message in st.session_state.logs:
                    with st.expander(log_message.splitlines()[0]):
                        st.code(log_message, language="text")

if __name__ == "__main__":
    main()  # Runs the main function
