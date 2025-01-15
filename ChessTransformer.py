# # prompt: install chess
# !pip install python-chess chess.engine

# !pip install python-chess~=0.26
# !pip install livelossplot==0.3.4
# !wget https://www.dropbox.com/sh/75gzfgu7qo94pvh/AACk_w5M94GTwwhSItCqsemoa/Stockfish%205/stockfish-5-linux.zip
# !unzip stockfish-5-linux.zip

# !chmod +x stockfish-5-linux/Linux/stockfish_14053109_x64

import chess
import chess.engine

# Cell 1: Imports
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import numpy as np
import re
from pathlib import Path
# from chessformers.tokenizer import Tokenizer
# from chessformers.configuration import get_configuration


import os
import re


if __name__ == "__main__":
    vocab_counter = set()

    with open(f"dataset/processed_kaggle2.txt", "w", encoding="utf-8") as outf:
        with open("/content/dataset/all_with_filtered_anotations_since1998 copy (1).txt", "r", encoding="utf-8") as inpf:
            for line in inpf:
                try:
                    ostr = line.split("###")[1].strip()
                    ostr = re.sub("W\d+.", "", ostr)
                    ostr = re.sub("B\d+.", "", ostr)

                    if len(ostr) > 0:
                        if ostr[-1] != '\n':
                            ostr = ostr + '\n'

                        outf.write(ostr)

                        for move in ostr.split(" "):
                            move = move.replace("\n", "")

                            if move != "":
                                vocab_counter.add(move)
                    else:
                        a = 0
                except:
                    pass

        os.makedirs("vocabs", exist_ok=True)

        with open(f"vocabs/kaggle2.txt", "w", encoding="utf-8") as f:
            for v in vocab_counter:
                f.write(v + "\n")

import os


VOCAB_DIR = "vocabs"


class Tokenizer:
    pad_token_index: int = 0
    bos_token_index: int = 1
    eos_token_index: int = 2
    unk_token_index: int = 3

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __init__(self, vocab_path: str = f"{VOCAB_DIR}/kaggle2_vocab.txt") -> None:
        self.vocab_dict = {
            self.pad_token: self.pad_token_index,
            self.bos_token: self.bos_token_index,
            self.eos_token: self.eos_token_index,
            self.unk_token: self.unk_token_index,
        }

        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, token in enumerate(f):
                self.vocab_dict[token.replace("\n", "")] = i + 4

    def encode(self, token_str: str, add_bos_token=True):
        encoded = []

        if add_bos_token:
            encoded.append(self.bos_token_index)

        for token in token_str.split():
            if token in self.vocab_dict:
                encoded.append(self.vocab_dict[token])
            else:
                encoded.append(self.unk_token_index)

        return encoded

    def decode(self, token_ids: list):
        decoded = []

        for token_id in token_ids:
            for token, index in self.vocab_dict.items():
                if index == token_id:
                    decoded.append(token)

        return " ".join(decoded)


    def vocab_size(self) -> int:
        return len(self.vocab_dict)


    @classmethod
    def generate_vocab(cls, dataset_path: str):
        from pathlib import Path
        from tqdm import tqdm

        vocab_counter = set()

        for game in tqdm(Path(dataset_path).glob("*.txt")):
            game = game.read_text(encoding="utf-8")
            for move in game.split(" "):
                move = move.replace("\n", "")

                if move != "":
                    vocab_counter.add(move)

        os.makedirs(VOCAB_DIR, exist_ok=True)

        with open(f"{VOCAB_DIR}/kaggle2.txt", "w", encoding="utf-8") as f:
            for v in vocab_counter:
                f.write(v + "\n")


if __name__ == "__main__":
    # Tokenizer.generate_vocab("dataset/kaggle2/")
    tokenizer = Tokenizer("/content/vocabs/kaggle2_vocab.txt")
    encoded = tokenizer.encode("d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 pepe Bb4+ Nc3 Ba5 Bf4 <eos>")
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)


# Cell 2: PGNDataset Class
class PGNDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, path: str, n_positions=512):
        self.n_positions = n_positions
        self.tokenizer = tokenizer
        self.games = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.games.append(line)

        print("Dataset read.")

    def __pad(self, sample: list):
        while len(sample) < self.n_positions:
            sample.append(self.tokenizer.pad_token_index)
        return sample[:self.n_positions]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        game = self.games[i]
        encoded = self.tokenizer.encode(game, add_bos_token=True)

        if len(encoded) < self.n_positions:
            encoded.append(self.tokenizer.eos_token_index)

        data = self.__pad(encoded)
        return torch.tensor(data)


import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer




# DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = "cuda"


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float(
        ) * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_tokens: int,
        dim_model: int,
        num_heads: int,
        d_hid: int,
        num_layers: int,
        dropout_p: float,
        n_positions: int,
    ):
        super().__init__()

        self.tokenizer = tokenizer

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.n_positions = n_positions

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=n_positions
        )
        self.embedding = nn.Embedding(
            num_tokens, dim_model, padding_idx=self.tokenizer.pad_token_index)

        encoder_layers = TransformerEncoderLayer(
            dim_model,
            num_heads,
            d_hid,
            dropout_p,
            batch_first=False,
            activation=F.gelu,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers)

        self.out = nn.Linear(dim_model, num_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, src, src_mask=None, src_pad_mask=None) -> torch.Tensor:
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer_encoder(
            src,
            src_mask,
            src_pad_mask,
        )

        out = self.out(transformer_out)

        return F.log_softmax(out, dim=-1)

    def get_src_mask(self, sz) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def get_pad_mask(self, matrix: torch.Tensor, pad_token: int) -> torch.Tensor:
        return (matrix == pad_token).t()

    def predict(
        self,
        input_string: str = "<bos>",
        max_length=80,
        stop_at_next_move=False,
        temperature=0.5
    ) -> str:
        import chess

        board = chess.Board()
        self.eval()

        input_sequence = self.tokenizer.encode(
            input_string, add_bos_token=False)

        for token in input_string.split(" ")[1:]:
            board.push_san(token)

        if board.is_checkmate():
            input_string += " <eos>"

        y_input = torch.tensor(
            [input_sequence], dtype=torch.long, device="cpu").t()

        if stop_at_next_move:
            max_length = 1
        else:
            max_length -= len(input_sequence)

        for _ in range(max_length):
            y_size = y_input.size(0)
            begin_loc = max(y_size - self.n_positions, 0)

            if y_size > self.n_positions and begin_loc % 2 != 0:
                # Let's help the model know what turn it is
                begin_loc += 1

            end_loc = min(begin_loc + self.n_positions, y_size)
            input_ids = y_input[begin_loc:end_loc]

            src_mask = self.get_src_mask(input_ids.size(0)).to("cpu")
            pad_mask = self.get_pad_mask(
                input_ids, self.tokenizer.pad_token_index).to("cpu")

            pred = self.forward(input_ids, src_mask, pad_mask)

            word_weights = pred[-1].squeeze().div(temperature).exp()
            word_idx = torch.multinomial(word_weights, 10)

            for wi in word_idx:
                decoded = self.tokenizer.decode([wi])
                try:
                    board.parse_san(decoded)
                    word_idx = wi
                    break
                except:
                    continue

            if word_idx.ndim > 0:
                # If the model doesn't know what to move, surrenders
                next_item = torch.tensor([[self.tokenizer.eos_token_index]], device="cpu")
                y_input = torch.cat((y_input, next_item), dim=0)
                break

            next_item = torch.tensor([[word_idx]], device="cpu")
            board.push_san(self.tokenizer.decode([next_item]))

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=0)

            if board.is_checkmate():
                # If it checkmates the opponent, return with <eos>
                next_item = torch.tensor([[self.tokenizer.eos_token_index]], device="cpu")
                y_input = torch.cat((y_input, next_item), dim=0)
                break

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.tokenizer.eos_token_index:
                break

        return self.tokenizer.decode(y_input.view(-1).tolist())

# Configuration Constants
n_positions = 80
dim_model = 768
d_hid = 3072
num_heads = 12
num_layers = 12
dropout_p = 0.1

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split



def _parse_args():
    parser = argparse.ArgumentParser(description='Chessformers trainer parser')

    # Providing default values for arguments
    parser.add_argument('--tokenizer', type=str, default="vocabs/kaggle2_vocab.txt", help='location of the tokenizer file')
    parser.add_argument('--dataset', type=str, default="dataset/processed_kaggle2.txt", help='location of the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam beta')
    parser.add_argument('--save_dir', type=str, default='./model', help='save model directory')
    parser.add_argument('--load_model', type=str, default=None, help='model to load and resume training')

    # Use parse_known_args to handle unknown arguments in Jupyter Notebook
    args, unknown = parser.parse_known_args()
    return args


class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, save_dir, learning_rate, num_epochs, adam_beta):
        self.save_dir = save_dir
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(adam_beta, 0.999))

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}.')

        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        train_loss = []
        for local_batch in tqdm(self.train_loader):
            X = local_batch.to(self.device).t().contiguous()
            y_input = X[:-1]
            y_expected = X[1:].reshape(-1)

            src_mask = self.model.get_src_mask(y_input.size(0)).to(self.device)
            pad_mask = self.model.get_pad_mask(
                y_input, self.model.tokenizer.pad_token_index).to(self.device)

            pred = self.model(y_input, src_mask, pad_mask)
            loss = self.loss_fn(pred.view(-1, self.model.tokenizer.vocab_size()), y_expected)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for local_batch in self.val_loader:
                X = local_batch.to(self.device).t().contiguous()
                y_input = X[:-1]
                y_expected = X[1:].reshape(-1)

                src_mask = self.model.get_src_mask(y_input.size(0)).to(self.device)
                pad_mask = self.model.get_pad_mask(
                    y_input, self.model.tokenizer.pad_token_index).to(self.device)

                pred = self.model(y_input, src_mask, pad_mask)
                loss = self.loss_fn(pred.view(-1, self.model.tokenizer.vocab_size()), y_expected)
                total_loss += loss
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            print(f"\n -------- EPOCH {epoch + 1}/{self.num_epochs} --------")
            train_loss = self.train_epoch()
            val_loss = self.test_epoch()

            print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.pth"))

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "final_model.pth"))


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer = Tokenizer(args.tokenizer)

    # Prepare the data
    dataset = PGNDataset(tokenizer, args.dataset, n_positions=n_positions)
    train_len = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Define the model
    model = Transformer(
        tokenizer=tokenizer,
        num_tokens=tokenizer.vocab_size(),
        dim_model=dim_model,
        d_hid=d_hid,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_p=dropout_p,
        n_positions=n_positions
    )

    if args.load_model:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(args.load_model))

    loss_fn = torch.nn.NLLLoss(ignore_index=tokenizer.pad_token_index)
    trainer = Trainer(model, train_loader, val_loader, loss_fn, args.save_dir, args.lr, args.epochs, args.beta1)
    trainer.train()


if __name__ == "__main__":
    args = _parse_args()
    main(args)


"""
Script used to play against the chessformers.
Human plays as white.
"""

import argparse
import torch


n_positions = 80
dim_model = 768
d_hid = 3072
num_heads = 12
num_layers = 12
dropout_p = 0.1

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Chessformers inference parser')

    parser.add_argument('--load_model', type=str, default="/content/chessformer_epoch_13.pth",
                        help='model to load and do inference')

    parser.add_argument('--tokenizer', type=str, default="/content/kaggle2_vocab.txt",
                        help='location of the tokenizer file')

    args, unknown = parser.parse_known_args()
    return args


def main(args) -> None:
    tokenizer = Tokenizer(args.tokenizer)
    model = Transformer(tokenizer,
                        num_tokens=tokenizer.vocab_size(),
                        dim_model=dim_model,
                        d_hid=d_hid,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout_p=dropout_p,
                        n_positions=n_positions,
                        )
    # model.load_state_dict(torch.load(args.load_model))
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))

    print(
        "===== CHESSFORMERS ENGINE =====\n"
    + "    Enter valid moves in PGN format.\n"
    + "    Enter \\b to undo a move.\n"
    + "    Enter \\m to show all moves\n"
    )

    input_string = "<bos>"
    boards = [input_string]

    while (len(input_string.split(" ")) < n_positions
           and input_string.split(" ")[-1] != tokenizer.eos_token):
        next_move = input("WHITE MOVE: ")

        if next_move == "\\m":
            print(input_string)
            continue
        elif next_move == "\\b":
            if len(boards) > 1:
                boards.pop()

            input_string = boards[-1]
            continue

        prev_input_string = input_string
        input_string += " " + next_move
        print(input_string)
        try:
            input_string = model.predict(
                input_string,
                stop_at_next_move=True,
                temperature=0.2,
                )
            boards.append(input_string)
            print("BLACK MOVE:", input_string.split(" ")[-1])
        except ValueError:
            input_string = prev_input_string
            print("ILLEGAL MOVE. Please, try again.")
        except Exception as e:
            print("UNHANDLED EXCEPTION. Please, try again.")

    print("--- Final board ---")
    print(input_string)


if __name__ == "__main__":
    args = _parse_args()
    main(args)