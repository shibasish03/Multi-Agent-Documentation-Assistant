from crewai import Agent, Task, Crew, LLM
import os

# Set your API key (you can also use `setx GROQ_API_KEY` in terminal to make it permanent)
os.environ['GROQ_API_KEY'] = "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"

# Initialize Groq LLM
llm = LLM(model="groq/llama-3.3-70b-versatile")

# --- Step 1: Take user input ---
paragraph = input("\nEnter the paragraph you want to summarize and translate:\n\n")

# --- Step 2: Create Agents ---
summarizer = Agent(
    role='Documentation Summarizer',
    goal='Create concise summaries of technical documentation',
    backstory='Technical writer who excels at simplifying complex concepts',
    llm=llm,
    verbose=True
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to Bengali language',
    backstory='Expert technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

# --- Step 3: Define Tasks ---
summary_task = Task(
    description=f'Summarize this text:\n\n{paragraph}',
    expected_output="A clear, concise summary of the text",
    agent=summarizer
)

translation_task = Task(
    description='Translate the above summary into Bengali.',
    expected_output="The Bengali translation of the summarized text.",
    agent=translator,
    dependencies=[summary_task]
)

# --- Step 4: Create Crew ---
crew = Crew(
    agents=[summarizer, translator],
    tasks=[summary_task, translation_task],
    verbose=True
)

# --- Step 5: Run the workflow ---
result = crew.kickoff()
print("\n✅ Final Output:")
print(result)
