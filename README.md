# Responsible AI - Claim Verification App

This application performs fact-checking, evidence retrieval, summarization, and sentiment analysis for user-provided claims. It leverages advanced language models and tools to deliver a responsible AI-driven solution for verifying claims and analyzing the sentiment of evidence summaries.

## Features

1. **Evidence Retrieval**: Gathers evidence from Google Search and Wikipedia.
2. **Summarization**: Condenses the retrieved evidence into concise insights.
3. **Fact-Checking**: Validates claims against the summarized evidence to determine their accuracy.
4. **Sentiment Analysis**: Analyzes the sentiment of the evidence summary to provide a composite score indicating its positivity, negativity, or neutrality.
5. **Interactive Streamlit Interface**: Users can input claims and view the results of the pipeline.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/your-project.git
   cd your-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser using the URL provided in the terminal (e.g., `http://localhost:8501`).

3. Enter a claim in the text area and click **Verify Now** to see the results.

## Tools and Technologies

- **Google Search (via Google Serper API)**: For retrieving general web evidence.
- **Wikipedia Search**: For structured and reliable information.
- **ChatGroq LLM**: A large language model for summarization and fact-checking.
- **TextBlob**: For sentiment analysis.
- **Streamlit**: For building the user interface.

## How It Works

1. **Evidence Retrieval**:
   - Retrieves evidence from Google and Wikipedia based on the user-provided claim.

2. **Summarization**:
   - Summarizes the retrieved evidence into user-friendly insights using the ChatGroq LLM.

3. **Fact-Checking**:
   - Compares the claim against the evidence summary to determine if the claim is supported, unsupported, or inconclusive.

4. **Sentiment Analysis**:
   - Analyzes the sentiment of the evidence summary, providing a composite score based on polarity and subjectivity.

5. **Results Display**:
   - Outputs the claim, evidence, summary, fact-check verdict, and sentiment score with intuitive color coding.

## Example

Input:
- Claim: *"Climate change is causing more frequent hurricanes."*

Output:
- Evidence: (Retrieved articles and summaries)
- Summary: *"Recent studies indicate an increase in hurricane frequency and intensity due to climate change, though attribution to specific events remains challenging."*
- Fact-Check Result: *"Supported"*
- Sentiment Score: *0.45 (Positive)*

## Screenshots

- **Input Screen**: Users can enter claims and view instructions.
- **Results Display**: Outputs evidence, summary, fact-check verdict, and sentiment score.
