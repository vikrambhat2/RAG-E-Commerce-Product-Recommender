from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from textblob import TextBlob  # Importing TextBlob for sentiment analysis
import streamlit as st


# Load environment variables from the .env file
load_dotenv()

# Access the variables using os.getenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Initialize the Google Serper API wrapper
google_serper = GoogleSerperAPIWrapper(api_key=serper_api_key)

llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Initialize the Wikipedia API wrapper
wikipedia = WikipediaAPIWrapper()

# Define tools for the Evidence Retrieval Agent
retrieval_tools = [
    Tool(
        name="Google Search",
        func=google_serper.run,
        description="Useful for searching general evidence and articles on the web"
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="Useful for finding reliable and structured information on Wikipedia"
    ),
]

# Evidence Retrieval Agent
evidence_retrieval_agent = initialize_agent(
    tools=retrieval_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Summarization Agent (Tool)
def summarize_evidence(evidence_text):
    """
    Summarizes the retrieved evidence into concise and meaningful insights.
    """
    prompt = f"Summarize the following evidence into concise insights:\n\n{evidence_text}"
    return llm.predict(prompt)

summarization_tool = Tool(
    name="Summarization Tool",
    func=summarize_evidence,
    description="Summarizes retrieved evidence into concise and user-friendly insights"
)

# Fact-Checking Agent (Tool)
def fact_check(claim, evidence_summary):
    """
    Fact-checks the claim against the summarized evidence.
    """
    prompt = (
        f"Claim: {claim}\n"
        f"Evidence Summary: {evidence_summary}\n"
        "Based on the evidence, is the claim supported, unsupported, or inconclusive? Provide a clear verdict and justification."
    )
    return llm.predict(prompt)

fact_checking_tool = Tool(
    name="Fact-Checking Tool",
    func=lambda input_text: fact_check(input_text['claim'], input_text['summary']),
    description="Fact-checks the claim against summarized evidence to provide a verdict"
)

# Sentiment Analysis Tool
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text and returns a composite score.
    The score is calculated as: (polarity * (1 - subjectivity)).
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Composite score: Adjust the weight of subjectivity as needed
    composite_score = polarity * (1 - subjectivity)
    return composite_score

sentiment_analysis_tool = Tool(
    name="Sentiment Analysis Tool",
    func=analyze_sentiment,
    description="Analyzes the sentiment of the input text and returns polarity and subjectivity scores"
)

# Combine tools for the full pipeline
pipeline_tools = [
    Tool(
        name="Retrieve Evidence",
        func=lambda claim: evidence_retrieval_agent.run(claim),
        description="Retrieves evidence for the given claim using Google and Wikipedia"
    ),
    summarization_tool,
    fact_checking_tool,
    sentiment_analysis_tool  # Add the sentiment analysis tool to the pipeline
]

# Initialize the full pipeline
full_pipeline_agent = initialize_agent(
    tools=pipeline_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to process claims through the full pipeline
def process_claim(claim):
    """
    Processes a claim through the full pipeline: Evidence Retrieval, Summarization, Fact-Checking, and Sentiment Analysis.
    """
    print("Step 1: Retrieving evidence...")
    evidence = evidence_retrieval_agent.run(claim)
    
    print("\nStep 2: Summarizing evidence...")
    summary = summarize_evidence(evidence)
    
    print("\nStep 3: Fact-checking the claim...")
    fact_check_result = fact_check(claim, summary)
    
    print("\nStep 4: Analyzing sentiment...")
    sentiment_result = analyze_sentiment(summary)  # Analyze sentiment of the summary
    
    return {
        "claim": claim,
        "evidence": evidence,
        "summary": summary,
        "fact_check_result": fact_check_result,
        "sentiment": sentiment_result  # Include sentiment analysis result
    }
def main():
    # Set page title, layout, and background
    st.set_page_config(page_title="Claim Verification & Sentiment Analysis", layout="wide")
    
    # Custom background style
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f1f1f1;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Display instructions
    with st.sidebar:
        st.title("Instructions")
        st.write("""
            1. Enter a claim or text that you want to verify in the text area.
            2. Click **Verify Now** to see the results of evidence retrieval, summarization, fact-checking, and sentiment analysis.
            3. View the sentiment score as a color-coded number and understand how positive or negative the claim's context is.
        """)
        st.write("---")

    # Main section: Display claim text input
    st.markdown(
    f'<h1 style="text-align:center; color:#007BFF;">Responsible AI: Fact-Checking Claims and Analyzing Sentiment</h1>',
    unsafe_allow_html=True)
    

    user_input = st.text_area("Enter your claim or question for verification:", height=100)

    if st.button("Verify Now"):
        if user_input:
            # Process the claim
            result = process_claim(user_input)

            # Display results
            #st.header("Final Output")
            st.subheader(f"Claim: {result['claim']}")
            st.write(f"**Evidence:**\n{result['evidence']}")
            st.write(f"**Summary:**\n{result['summary']}")
            st.write(f"**Fact-Check Result:**\n{result['fact_check_result']}")

            # Sentiment Score
            sentiment = result['sentiment']

            # Define sentiment labels and colors
            if sentiment > 0.1:
                sentiment_label = "Positive"
                sentiment_color = "green"
            elif sentiment < -0.1:
                sentiment_label = "Negative"
                sentiment_color = "red"
            else:
                sentiment_label = "Neutral"
                sentiment_color = "orange"

            # Display sentiment score inline with color
            st.write(f"**Sentiment Score:**")
            st.markdown(f'<h3 style="color:{sentiment_color}; display:inline;">{sentiment:.2f}</h3>', unsafe_allow_html=True)
            #st.write(f"The sentiment score indicates that the claim is **{sentiment_label}**.")

        else:
            st.warning("Please enter a claim or text to verify.")

if __name__ == "__main__":
    main()