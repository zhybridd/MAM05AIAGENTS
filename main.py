from datetime import datetime  # Used for logging timestamps
import tempfile  # Used for handling temporary file storage
from dataclasses import dataclass  # Used for defining structured data objects

from groq import Groq  # API client for AI model execution
import streamlit as st  # Used to build the web interface
from dotenv import load_dotenv  # Loads environment variables

from graph import Graph, Agent, ExecutionContext, Prompt  # Imports graph execution model
from utils import (
    get_think_tag,  # Extracts think tags from AI responses
    remove_think_tag,  # Cleans AI responses
    load_patient_data_from_sqlite_file,  # Loads patient data from database
    load_prompts,  # Loads AI prompts from YAML files
)

# Configures the Streamlit web page layout and metadata
st.set_page_config(
    layout="wide",
    page_title="UVA Bot 🤖",
    page_icon=":dna:",
)

# Initializes a log in the session state if it doesn't already exist
if "logs" not in st.session_state:
    st.session_state.logs = []

# Function to add logs with timestamps
def log(message):
    time = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{time}] {message}")

# Defines the AI model being used
model_name = "deepseek-r1-distill-qwen-32b"

# Defines the AI agent class, which interacts with the AI model
@dataclass(frozen=True)
class MedicalAgent(Agent):
    model_name: str  # AI model name
    client = Groq(api_key="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")  # API client

    def run(self, context: ExecutionContext) -> str:
        log(f"Running {self.name}")
        prompt = self._prepare_prompt(context)  # Prepares the AI prompt
        log(f"{self.name} Prompt: {prompt}")
        
        # Sends prompt to the AI model
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )

        answer = chat_completion.choices[0].message.content  # Extracts AI response
        think_tag = get_think_tag(answer)  # Extracts think tags if present
        log(f"{self.name} Think Tag: {think_tag}")
        cleaned_answer = remove_think_tag(answer)  # Cleans AI response
        log(f"{self.name} Response: {cleaned_answer}")
        context.set(self.name, cleaned_answer)  # Stores the result in execution context
        return context

# Defines a structured object to store results
dataclass
class RunResult:
    biomarker_report: str
    imaging_report: str
    pathology_report: str
    oncologist_report: str

# Main function to run the Streamlit application
def main():
    load_dotenv()  # Loads API keys and other environment variables
    crew_prompts = load_prompts()  # Loads AI agent prompts

    # Function to create an AI agent instance
def agent(name: str, prompt: Prompt) -> MedicalAgent:
        return MedicalAgent(name=name, model_name=model_name, prompt=prompt)

    patient_id_to_patient = {}  # Dictionary to store patient data

    # Function to process a specific patient
def run_patient(patient_id: str) -> RunResult:
        patient = patient_id_to_patient[patient_id]  # Retrieves patient data
        context = ExecutionContext()  # Creates an execution context

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
                "biomarkers -> oncologist",
                "imaging -> oncologist",
                "pathology -> oncologist",
            },
        )

        result = graph.run(context)  # Executes AI agents

        # Stores the final AI-generated reports
        run_result = RunResult(
            biomarker_report=result.get("biomarkers"),
            imaging_report=result.get("imaging"),
            pathology_report=result.get("pathology"),
            oncologist_report=result.get("oncologist"),
        )
        return run_result

    # UI Elements
    st.sidebar.title(":orange[__UVA Bot__] 🤖")

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            db_path = tmp_file.name
            patient_id_to_patient = load_patient_data_from_sqlite_file(db_path)

        patient_ids = list(patient_id_to_patient.keys())
        patient_id = st.sidebar.selectbox("Select Patient ID", patient_ids)
        patient = patient_id_to_patient[patient_id]

        # Display patient reports
        tab1, tab2, tab3 = st.tabs(["Biomarkers Report", "Imaging Report", "Pathology Report"])

        with tab1:
            st.text(patient.biomarker_report)
        with tab2:
            st.text(patient.imaging_report)
        with tab3:
            st.text(patient.pathology_report)

        st.divider()

        # Button to start AI processing
        submit = st.sidebar.button("Submit")
        if submit:
            run_result = run_patient(patient_id)
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
                for i, log_message in enumerate(st.session_state.logs):
                    with st.expander(log_message.splitlines()[0]):
                        st.code(log_message, language="text")
                    if i != 0 and i % 3 == 0:
                        st.divider()

if __name__ == "__main__":
    main()  # Runs the main function
