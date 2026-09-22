# AI Credit Score Guidance Agent

An AI-powered credit score guidance application that analyzes a user's credit profile and provides personalized recommendations, score projections, visual insights, and an interactive AI chatbot.

## Features

* Credit score profile analysis
* Payment history analysis
* Credit utilization analysis
* Credit history age analysis
* Hard inquiry analysis
* Rule-based credit assessment
* AI-generated financial guidance using Claude
* Credit score projection
* Visual score trajectory chart
* Detailed parameter comparison table
* Interactive AI chatbot
* Rule-based fallback when the LLM is unavailable

## Technology Stack

* Python 3.13
* Gradio
* Pandas
* NumPy
* Matplotlib
* Anthropic Claude API

## AI Model

The application uses the Anthropic API with the Claude Sonnet model to generate natural-language explanations and answer follow-up questions.

The application also contains a rule-based fallback system so that the core analysis can continue when the API is unavailable.

## Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd AI-Credit-Score-Agent
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## API Key Setup

Set your Anthropic API key as an environment variable.

### Windows PowerShell

```powershell
$env:ANTHROPIC_API_KEY="your_api_key"
```

Never upload your API key directly to GitHub.

## Run Locally

```bash
python credit.py
```

The Gradio application will start locally and provide a web interface.

## Deployment

The application can be deployed as a Python Web Service on Render.

### Render Configuration

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
python credit.py
```

Add the following environment variable in Render:

```text
ANTHROPIC_API_KEY = your_api_key
```

## Project Architecture

```text
User Input
    ↓
Credit Profile Parameters
    ↓
Rule-Based Analysis Engine
    ↓
Score Classification & Recommendations
    ↓
Claude AI Layer
    ↓
AI-Generated Explanation
    ↓
Gradio Interface
    ↓
Interactive Chatbot
```
## Screenshots

### 1. Input Financial Parameters Form

<img width="463" height="331" alt="Screenshot 2026-09-03 211702" src="https://github.com/user-attachments/assets/90f02ec3-eeeb-4024-8d61-44c0aa64c836" />


### 2. Detailed Metric Breakdown Against Optimal Benchmarks

<img width="449" height="236" alt="Screenshot 2026-09-03 211742" src="https://github.com/user-attachments/assets/5d7a0f79-b46b-45f8-bef9-7e1b52d02027" />


### 3. Forecasted Score Trajectory

<img width="446" height="213" alt="Screenshot 2026-09-03 211731" src="https://github.com/user-attachments/assets/e5f488c4-d378-49be-9dd3-d432ea4074e3" />


### 4. Overall Assessment, Actionable Recommendations & Agent Execution Logs

<img width="427" height="334" alt="Screenshot 2026-09-03 211720" src="https://github.com/user-attachments/assets/24543936-b990-4c06-9607-29e89bfc650a" />

### 5. Ask the Agent — grounded conversational guidance based on the user's credit-score analysis.

<img width="683" height="282" alt="Screenshot 2026-09-20 230457" src="https://github.com/user-attachments/assets/aca81af0-0054-4603-8ef9-65af7a060c2a" />



## Disclaimer

This project is developed for educational and demonstration purposes. The generated guidance should not be considered professional financial or credit advice.
