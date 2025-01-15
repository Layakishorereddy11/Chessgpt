# ChessGPT

## 1. INTRODUCTION

### 1.1 THE CHALLENGE OF CHESS MOVE PREDICTION
In the realm of artificial intelligence, the game of chess has long served as a benchmark for evaluating the strategic reasoning capabilities of AI systems. This project, titled "ChessGPT," focuses on the intricate task of predicting future moves in a chess game using natural language processing (NLP) techniques and language models. Our primary objective was to develop an AI agent capable of analyzing a sequence of chess moves and generating informed recommendations for the next best move, ultimately aiming to win the game.

### 1.2 TREATING CHESS MOVES AS A LANGUAGE
The core concept behind our approach involves treating chess moves as a language. We leveraged a causal language model, specifically a pre-trained GPT-2, to learn the patterns and strategies inherent in chess games. This innovative approach views the prediction of the next chess move as a language modeling task, where the model learns to generate a sequence of moves that aligns with the rules and strategies of chess. By considering previous moves as context, the model can make informed predictions about the most suitable next move.

### 1.3 BASELINE MODEL: THE FOUNDATION
Our exploration began with the standard GPT-2 architecture with an AutoTokenizer as our baseline model. This baseline served as a benchmark, allowing us to assess the performance improvements achieved through subsequent fine-tuning and architectural modifications.

### 1.4 EXPLORING ARCHITECTURES: A JOURNEY OF REFINEMENT
We embarked on a journey of exploring various architectures to identify the most effective approach for chess move prediction:
1. **Base GPT-2**: The standard GPT-2 model served as the initial baseline.
2. **GPT-2 with LoRA**: GPT-2 fine-tuned using Low-Rank Adaptation (LoRA) for efficient training with reduced computational resources.
3. **GPT-2 with Reinforced Fine-Tuning**: GPT-2 with enhancements like positional embeddings, custom tokenization, and a custom loss function to penalize illegal moves.
4. **Chess Transformers from Scratch**: A custom transformer model coded from scratch, built specifically for chess move prediction, with a vocabulary containing just chess moves in SAN format.

### 1.5 SUMMARY OF RESULTS: THE BEST APPROACH
Among the architectures we explored, the custom-built Chess Transformers model (Model 4) emerged as the most successful. It consistently generated legal moves and completed entire games without errors. Other models showed varying degrees of success in generating legal moves, with the base GPT-2 model generating up to 15 legal moves before encountering difficulties, similarly Model 2 and Model 3 generated up to 20 and 30 moves respectively.

## 2. INSTRUCTIONS TO RUN LOCALLY

To run the ChessGPT project locally, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine:
```sh
git clone repo
cd ChessGPT
```
### Step 2: Create a Virtual Environment
Create and activate a virtual environment to manage dependencies:

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### Step 3: Install Required Libraries
Install the required libraries listed in the requirements.txt file:
```sh
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
Run the Streamlit application using the following command:
```sh
streamlit run main.py
```
### Step 5: Open the Application
Open your web browser and navigate to http://localhost:8501 to interact with the ChessGPT application.

<img width="985" alt="image" src="https://github.com/user-attachments/assets/70b378b5-2e2a-4ea3-a1c2-2e00ad698d2d" />




