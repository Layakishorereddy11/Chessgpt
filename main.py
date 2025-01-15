import streamlit as st
import torch
import chess
import chess.svg
from pathlib import Path
from ChessTransformer import Tokenizer, Transformer

# Constants
n_positions = 80
dim_model = 768
d_hid = 3072
num_heads = 12
num_layers = 12
dropout_p = 0.1

def initialize_model():
    model_path = "/Users/layakishorereddy/Desktop/Rutgers/ChessGPT/ChesssFinal/ChessGpt_Scratch.pth"  # Update with your model path
    tokenizer_path = "/Users/layakishorereddy/Desktop/Rutgers/ChessGPT/ChesssFinal/kaggle2_vocab.txt"  # Update with your tokenizer path
    
    tokenizer = Tokenizer(tokenizer_path)
    model = Transformer(
        tokenizer=tokenizer,
        num_tokens=tokenizer.vocab_size(),
        dim_model=dim_model,
        d_hid=d_hid,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_p=dropout_p,
        n_positions=n_positions,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, tokenizer

def get_chess_board_svg(board):
    return chess.svg.board(board=board, size=400)

def main():
    st.title("Chess GPT - Play Against AI")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.tokenizer = initialize_model()
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'game_history' not in st.session_state:
        st.session_state.game_history = ["<bos>"]
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False

    # Display chess board
    st.write(get_chess_board_svg(st.session_state.board), unsafe_allow_html=True)

    # Input for player's move
    move = st.text_input("Enter your move (e.g., e4):", key="move_input")
    
    if st.button("Make Move"):
        if move and not st.session_state.game_over:
            try:
                # Process player's move
                st.session_state.game_history[-1] += f" {move}"
                st.session_state.board.push_san(move)
                
                # Get AI's response
                ai_response = st.session_state.model.predict(
                    st.session_state.game_history[-1],
                    stop_at_next_move=True,
                    temperature=0.2,
                )
                
                # Extract and make AI's move
                ai_move = ai_response.split()[-1]
                if ai_move != "<eos>":
                    st.session_state.board.push_san(ai_move)
                    st.session_state.game_history.append(ai_response)
                    st.success(f"AI played: {ai_move}")
                else:
                    st.session_state.game_over = True
                    st.warning("Game Over!")
                
                # Rerun to update the board
                st.rerun()
                
            except ValueError:
                st.error("Invalid move! Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Game controls
    if st.button("Reset Game"):
        st.session_state.board = chess.Board()
        st.session_state.game_history = ["<bos>"]
        st.session_state.game_over = False
        st.rerun()

    # Display game status
    if st.session_state.board.is_checkmate():
        st.warning("Checkmate!")
        st.session_state.game_over = True
    elif st.session_state.board.is_stalemate():
        st.warning("Stalemate!")
        st.session_state.game_over = True
    elif st.session_state.board.is_check():
        st.warning("Check!")

    # Display move history
    with st.expander("View Move History"):
        st.write(" ".join(st.session_state.game_history))

if __name__ == "__main__":
    main()
