# lstm_crf.py

import torch
import torch.nn as nn
import torch.optim as optim
from TorchCRF import CRF
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from seqeval.metrics import classification_report as seq_classification_report


class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate parameters
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate parameters
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate parameters
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate parameters
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for param in [self.W_ii, self.W_hi, self.W_if, self.W_hf,
                      self.W_ig, self.W_hg, self.W_io, self.W_ho]:
            nn.init.xavier_uniform_(param)

        for param in [self.b_i, self.b_f, self.b_g, self.b_o]:
            nn.init.zeros_(param)

    def forward(self, input_seq: torch.Tensor,
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = input_seq.size()

        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size,
                              device=input_seq.device)
        else:
            h_t = h_0

        if c_0 is None:
            c_t = torch.zeros(batch_size, self.hidden_size,
                              device=input_seq.device)
        else:
            c_t = c_0

        h_seq = []

        for t in range(seq_length):
            x_t = input_seq[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ii.T + h_t @
                                self.W_hi.T + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if.T + h_t @
                                self.W_hf.T + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig.T + h_t @ self.W_hg.T + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_io.T + h_t @
                                self.W_ho.T + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            h_seq.append(h_t.unsqueeze(1))

        h_seq = torch.cat(h_seq, dim=1)
        return h_seq, (h_t, c_t)


class BidirectionalCustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(BidirectionalCustomLSTM, self).__init__()
        self.forward_lstm = CustomLSTM(input_size, hidden_size)
        self.backward_lstm = CustomLSTM(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_seq: torch.Tensor,
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Forward direction
        forward_out, (h_f, c_f) = self.forward_lstm(input_seq, h_0, c_0)

        # Backward direction
        reversed_input = torch.flip(input_seq, [1])
        backward_out, (h_b, c_b) = self.backward_lstm(reversed_input, h_0, c_0)
        backward_out = torch.flip(backward_out, [1])

        # Concatenate outputs
        h_seq = torch.cat([forward_out, backward_out], dim=2)
        h_n = torch.cat([h_f, h_b], dim=1)
        c_n = torch.cat([c_f, c_b], dim=1)

        return h_seq, (h_n, c_n)


class DataCollatorWithPadding:
    def __init__(self, pad_idx: int, device: torch.device):
        self.pad_idx = pad_idx
        self.device = device

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(x[0]) for x in batch)

        padded_X = torch.ones(len(batch), max_len).long() * self.pad_idx
        padded_y = torch.ones(len(batch), max_len).long() * self.pad_idx
        mask = torch.zeros(len(batch), max_len).bool()

        for i, (x, y) in enumerate(batch):
            seq_len = len(x)
            padded_X[i, :seq_len] = x[:seq_len]
            padded_y[i, :seq_len] = y[:seq_len]
            mask[i, :seq_len] = 1

        return padded_X.to(self.device), padded_y.to(self.device), mask.to(self.device)


class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int,
                 hidden_dim: int, padding_idx: int, dropout: float = 0.5):
        super(LSTM_CRF, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = BidirectionalCustomLSTM(embedding_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size)
        self.padding_idx = padding_idx

    def forward(self, sentences: torch.Tensor, tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.transpose(
            0, 1) if lstm_out.shape[0] == sentences.shape[1] else lstm_out
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        loss = -self.crf(emissions, tags, mask=mask)
        return loss.mean()

    def predict(self, sentences: torch.Tensor,
                mask: torch.Tensor) -> List[List[int]]:
        self.eval()
        with torch.no_grad():
            embeds = self.embedding(sentences)
            embeds = self.dropout(embeds)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = lstm_out.transpose(
                0, 1) if lstm_out.shape[0] == sentences.shape[1] else lstm_out
            lstm_out = self.dropout(lstm_out)
            emissions = self.hidden2tag(lstm_out)
            predictions = self.crf._viterbi_decode(emissions, mask=mask)
        return predictions


class NERModel:
    def __init__(self, config: Dict):

        # Replace the cuda check with mps check
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.config = config
        self.model = LSTM_CRF(
            vocab_size=config['vocab_size'],
            tagset_size=config['tagset_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            padding_idx=config['padding_idx']
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.data_collator = DataCollatorWithPadding(
            config['padding_idx'], self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, idx2tag: Dict[int, str]) -> None:
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_X, batch_y, mask in train_loader:
                self.optimizer.zero_grad()
                loss = self.model(batch_X, batch_y, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                total_loss += loss.item()

            # Validation
            val_predictions, val_labels = self.evaluate(val_loader, idx2tag)
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Training loss: {total_loss / len(train_loader):.4f}")
            print("\nValidation Metrics:")
            print(seq_classification_report(val_labels, val_predictions))

    def evaluate(self, data_loader: DataLoader,
                 idx2tag: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch_X, batch_y, mask in data_loader:
                batch_preds = self.model.predict(batch_X, mask)

                for pred_seq, true_seq, seq_mask in zip(batch_preds, batch_y, mask):
                    seq_len = seq_mask.sum().item()
                    pred_tags = [idx2tag[p] for p in pred_seq[:seq_len]]
                    true_tags = [idx2tag[t.item()] for t in true_seq[:seq_len]]
                    predictions.append(pred_tags)
                    labels.append(true_tags)

        return predictions, labels

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    @classmethod
    def load(cls, path: str) -> 'NERModel':
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model
