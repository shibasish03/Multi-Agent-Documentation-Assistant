import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM


GROQ_API_KEY = "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"

INDIAN_LANGUAGES = [
    "Hindi", "Bengali", "Marathi", "Telugu", "Tamil",
    "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia",
    "Urdu", "Assamese"
]


st.set_page_config(
    page_title="CrewAI Summarizer & Translator",
    layout="centered",
    initial_sidebar_state="collapsed", 
)


def run_crew_workflow(api_key: str, paragraph: str, target_language: str):
    """
    Initializes and runs the CrewAI workflow for summarization and dynamic translation.
    """
    if not api_key:
        return "Error: The Groq API Key is missing or invalid."
    
    if not paragraph or len(paragraph.strip()) < 20:
        return "Error: Please enter a substantial paragraph (at least 20 characters) to summarize."

    
    os.environ['GROQ_API_KEY'] = api_key
    
    st.info(f"Initializing CrewAI with Groq model for translation into **{target_language}**...")
    
    try:
        
        llm = LLM(model="groq/llama-3.3-70b-versatile")

        
        summarizer = Agent(
            role='Documentation Summarizer',
            goal='Create concise summaries of the provided technical or general documentation',
            backstory='Technical writer who excels at simplifying complex concepts into brief, easy-to-digest summaries',
            llm=llm,
            verbose=False,
            allow_delegation=False
        )

        
        translator = Agent(
            role=f'Expert Technical Translator ({target_language})',
            goal=f'Translate the provided text summary into the {target_language} language',
            backstory=f'Expert technical translator specializing in converting content accurately into fluent {target_language}',
            llm=llm,
            verbose=False,
            allow_delegation=False
        )

        
        summary_task = Task(
            description=f'Summarize the following text:\n\n---\n{paragraph}\n---',
            expected_output="A clear, concise, and structured summary of the input text in English.",
            agent=summarizer
        )

        
        translation_task = Task(
            description=f"Translate the summary provided in the dependency task into highly accurate, fluent {target_language} language. Provide ONLY the translated text.",
            expected_output=f"The {target_language} translation of the summarized text.",
            agent=translator,
            dependencies=[summary_task]
        )

        
        crew = Crew(
            agents=[summarizer, translator],
            tasks=[summary_task, translation_task],
            verbose=True 
        )

        
        with st.spinner(f"Crew is working... Summarizing and translating to {target_language}. This may take a moment."):
            result = crew.kickoff()
        
        return result

    except Exception as e:
        return f"An error occurred during the CrewAI execution. Please check the model name or API Key validity: {e}"



st.title("📄 Multi-Step AI Workflow: Summarize & Translate")
st.markdown("Your Groq API Key is now securely embedded in the script.")


paragraph_input = st.text_area(
    "1. Enter the Paragraph to Process:",
    height=250,
    placeholder="Paste your technical documentation or general text here.",
)


selected_language = st.selectbox(
    "2. Select Target Translation Language:",
    options=INDIAN_LANGUAGES,
    index=INDIAN_LANGUAGES.index("Bengali"), 
)


if st.button("3. Run Workflow", type="primary"):
    
    if not paragraph_input:
        st.warning("Please enter a paragraph to process.")
    else:
        
        with st.expander("Crew Execution Log", expanded=False):
            
            final_output = run_crew_workflow(GROQ_API_KEY, paragraph_input, selected_language)

        st.subheader("✅ Final Result")
        st.markdown(final_output)

        if isinstance(final_output, str) and "Error:" not in final_output:
            st.subheader("Breakdown (Check Log for Intermediate Steps)")
            st.info(f"The final output above is the **{selected_language}** translation. Expand the **Crew Execution Log** to see the English summary generated in the first step.")
        


