import streamlit as st
import torch
import chess
import streamlit.components.v1 as components
from pathlib import Path
from ChessTransformer import Tokenizer, Transformer

# Constants
n_positions = 80
dim_model = 768
d_hid = 3072
num_heads = 12
num_layers = 12
dropout_p = 0.1

# Modified JavaScript to handle moves properly
CHESSBOARD_HTML = """
<head>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
</head>
<body>
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <div id="board" style="width: 400px; margin: 20px;"></div>
    </div>
    <script>
        var board = null;
        var game = new Chess('{fen}');

        function sendMove(source, target) {{
            const data = {{
                from: source,
                to: target
            }};
            window.parent.postMessage({{
                type: 'streamlit:component-input',
                move: data
            }}, '*');
        }}

        function onDragStart(source, piece) {{
            if (game.game_over()) return false;
            if (piece.search(/^b/) !== -1) return false;
        }}

        function onDrop(source, target) {{
            var move = game.move({{
                from: source,
                to: target,
                promotion: 'q'
            }});

            if (move === null) return 'snapback';
            
            sendMove(source, target);
            return false;
        }}

        var config = {{
            position: '{fen}',
            draggable: true,
            onDragStart: onDragStart,
            onDrop: onDrop,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png'
        }};

        board = Chessboard('board', config);
        window.addEventListener('resize', board.resize);
    </script>
</body>
"""

def initialize_model():
    try:
        # Update these paths to your actual model and tokenizer paths
        model_path = "/Users/layakishorereddy/Desktop/Rutgers/ChessGPT/ChesssFinal/ChessGpt_Scratch.pth"  # Update with your model path
        tokenizer_path = "/Users/layakishorereddy/Desktop/Rutgers/ChessGPT/ChesssFinal/kaggle2_vocab.txt" 
        
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            return None, None
            
        if not tokenizer_path.exists():
            st.error(f"Tokenizer file not found at: {tokenizer_path}")
            return None, None

        # Initialize tokenizer first
        tokenizer = Tokenizer(str(tokenizer_path))
        if not tokenizer:
            st.error("Failed to initialize tokenizer")
            return None, None

        # Create model with explicit device placement
        device = torch.device('cpu')
        model = Transformer(
            tokenizer=tokenizer,
            num_tokens=tokenizer.vocab_size(),
            dim_model=dim_model,
            d_hid=d_hid,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p,
            n_positions=n_positions,
        ).to(device)

        # Load state dict with safe loading
        try:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
                pickle_module=torch.serialization.pickle
            )
            
            # If checkpoint is a state dict, use it directly
            if isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                st.error("Invalid model checkpoint format")
                return None, None

            model.load_state_dict(state_dict)
            model.eval()
            
            # Disable gradient computation
            torch.set_grad_enabled(False)
            
            st.success("Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            st.error(f"Error loading model weights: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None

def main():
    st.title("Chess GPT - Play Against AI")
    
    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
        st.session_state.game_history = ["<bos>"]
        st.session_state.model, st.session_state.tokenizer = initialize_model()
        st.session_state.last_move = None

    # Game controls
    if st.button("Reset Game"):
        st.session_state.board = chess.Board()
        st.session_state.game_history = ["<bos>"]
        st.session_state.last_move = None
        st.rerun()

    # Display the chess board
    component_value = components.html(
        CHESSBOARD_HTML.format(fen=st.session_state.board.fen()),
        height=500,
    )

    # Handle player's move
    if component_value and isinstance(component_value, dict):
        try:
            # Extract move data
            move_from = component_value.get('from')
            move_to = component_value.get('to')
            
            if move_from and move_to:
                # Create and validate the move
                move = chess.Move.from_uci(f"{move_from}{move_to}")
                if move in st.session_state.board.legal_moves:
                    # Make player's move
                    san_move = st.session_state.board.san(move)
                    st.session_state.board.push(move)
                    st.session_state.game_history[-1] += f" {san_move}"
                    
                    # Get AI's move
                    if st.session_state.model:
                        ai_response = st.session_state.model.predict(
                            st.session_state.game_history[-1],
                            stop_at_next_move=True,
                            temperature=0.2,
                        )
                        
                        # Process AI's move
                        ai_move = ai_response.split()[-1]
                        if ai_move and ai_move != "<eos>":
                            try:
                                st.session_state.board.push_san(ai_move)
                                st.session_state.game_history.append(ai_response)
                                st.success(f"AI played: {ai_move}")
                            except ValueError as e:
                                st.error(f"AI made invalid move: {ai_move}")
                    
                    st.rerun()
                else:
                    st.error("Invalid move!")
            
        except Exception as e:
            st.error(f"Move error: {str(e)}")

    # Display game status
    if st.session_state.board.is_game_over():
        if st.session_state.board.is_checkmate():
            st.warning("Checkmate!")
        elif st.session_state.board.is_stalemate():
            st.warning("Stalemate!")
        else:
            st.warning("Game Over!")
    elif st.session_state.board.is_check():
        st.warning("Check!")

    # Display move history
    with st.expander("View Move History"):
        st.write(" ".join(st.session_state.game_history))

if __name__ == "__main__":
    main()