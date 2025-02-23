from datetime import datetime
import tempfile
from dataclasses import dataclass

from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# Importing custom modules that handle AI agent execution and utilities
from graph import Graph, Agent, ExecutionContext, Prompt
from utils import (
    get_think_tag,
    remove_think_tag,
    load_patient_data_from_sqlite_file,
    load_prompts,
)

# Set up the Streamlit web interface configuration
st.set_page_config(
    layout="wide",  # Expands the interface to full width
    page_title="UVA Bot 🤖",  # Sets the webpage title
    page_icon=":dna:",  # Uses a DNA emoji as the favicon
)

# Initialize session state logs if not present
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to log messages with timestamps
def log(message):
    time = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{time}] {message}")

# Define the AI model to be used
model_name = "deepseek-r1-distill-qwen-32b"

# Define an AI agent class that extends the generic Agent class
@dataclass(frozen=True)
class MedicalAgent(Agent):
    model_name: str  # Model name for the LLM
    client = Groq(api_key="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")  # API client for interacting with the Groq AI model

    # Function to execute the agent's task using AI
    def run(self, context: ExecutionContext) -> str:
        log(f"Running {self.name}")  # Log execution start
        prompt = self._prepare_prompt(context)  # Prepare AI prompt
        log(f"{self.name} Prompt: {prompt}")

        # Call the AI model to generate a response
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0.0,  # Ensure deterministic output
            top_p=1.0,
            seed=0,
        )

        # Extract AI-generated response
        answer = chat_completion.choices[0].message.content
        think_tag = get_think_tag(answer)  # Extract thought process tags (if any)
        log(f"{self.name} Think Tag: {think_tag}")
        cleaned_answer = remove_think_tag(answer)  # Remove AI internal thoughts
        log(f"{self.name} Response: {cleaned_answer}")
        context.set(self.name, cleaned_answer)  # Store the AI response in execution context
        return context

# Dataclass to store structured AI-generated reports
@dataclass
class RunResult:
    biomarker_report: str
    imaging_report: str
    pathology_report: str
    oncologist_report: str

# Main function to orchestrate AI execution
def main():
    load_dotenv()  # Load environment variables
    crew_prompts = load_prompts()  # Load AI prompts from YAML files

    # Helper function to create an AI agent
    def agent(name: str, prompt: Prompt) -> MedicalAgent:
        return MedicalAgent(name=name, model_name=model_name, prompt=prompt)

    # Dictionary to store patient data
    patient_id_to_patient = {}

    # Function to process patient data and execute AI agents
    def run_patient(patient_id: str) -> RunResult:
        patient = patient_id_to_patient[patient_id]  # Retrieve patient data
        context = ExecutionContext()
        
        # Set patient reports in execution context
        context.set("BiomarkersReport", patient.biomarker_report)
        context.set("ImagingReport", patient.imaging_report)
        context.set("PathologyReport", patient.pathology_report)

        # Define the graph structure for AI agent execution
        graph = Graph.from_nodes(
            nodes={
                agent("biomarkers", crew_prompts.biomarker),
                agent("imaging", crew_prompts.imaging),
                agent("pathology", crew_prompts.pathology),
                agent("oncologist", crew_prompts.oncologist),
            },
            edges={
                "biomarkers -> oncologist",  # Biomarker results feed into the oncologist
                "imaging -> oncologist",  # Imaging results feed into the oncologist
                "pathology -> oncologist",  # Pathology results feed into the oncologist
            },
        )

        result = graph.run(context)  # Execute AI processing graph

        # Collect the AI-generated reports
        run_result = RunResult(
            biomarker_report=result.get("biomarkers"),
            imaging_report=result.get("imaging"),
            pathology_report=result.get("pathology"),
            oncologist_report=result.get("oncologist"),
        )
        return run_result

    # Sidebar UI setup
    st.sidebar.title(":orange[__UVA Bot__] 🤖")

    # File uploader for patient database
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            db_path = tmp_file.name  # Get the file path of the uploaded database

            patient_id_to_patient = load_patient_data_from_sqlite_file(db_path)

        # Display patient selection dropdown
        patient_ids = list(patient_id_to_patient.keys())
        patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)
        patient = patient_id_to_patient[patient_id]

        # Create tabs to display different patient reports
        tab1, tab2, tab3 = st.tabs([
            "Biomarkers Report", "Imaging Report", "Pathology Report"
        ])

        with tab1:
            st.text(patient.biomarker_report)
        with tab2:
            st.text(patient.imaging_report)
        with tab3:
            st.text(patient.pathology_report)

        st.divider()

        # Submit button to process patient data
        submit = st.sidebar.button("Submit")
        if submit:
            run_result = run_patient(patient_id)  # Run AI analysis
            t1, t2, t3, t4, t5 = st.tabs([
                "Biomarkers", "Imaging", "Pathology", "Oncologist", "Logs"
            ])

            # Display AI-generated reports in respective tabs
            with t1:
                st.markdown(run_result.biomarker_report)
            with t2:
                st.markdown(run_result.imaging_report)
            with t3:
                st.markdown(run_result.pathology_report)
            with t4:
                st.markdown(run_result.oncologist_report)
            
            # Display AI logs
            with t5:
                for i, log_message in enumerate(st.session_state.logs):
                    with st.expander(log_message.splitlines()[0]):
                        st.code(log_message, language="text")
                    if i != 0 and i % 3 == 0:
                        st.divider()

# Execute the main function if script is run directly
if __name__ == "__main__":
    main()
