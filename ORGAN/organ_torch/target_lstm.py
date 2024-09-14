import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TargetLSTM(nn.Module):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token):
        super(TargetLSTM, self).__init__()
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.temperature = 1.0

        torch.manual_seed(66)

        # Embeddings
        self.g_embeddings = nn.Embedding(num_emb, emb_dim)

        # LSTM and Output Layer
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_emb)

    def forward(self, x, hidden=None):
        # x: [batch_size, sequence_length]
        embedded = self.g_embeddings(x)  # [batch_size, sequence_length, emb_dim]
        outputs, hidden = self.lstm(embedded, hidden)  # outputs: [batch_size, sequence_length, hidden_dim]
        logits = self.output_layer(outputs)  # [batch_size, sequence_length, num_emb]
        return logits, hidden

    def generate(self):
        gen_x = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long)
        hidden = (torch.zeros(1, self.batch_size, self.hidden_dim), torch.zeros(1, self.batch_size, self.hidden_dim))

        x_t = torch.full((self.batch_size,), self.start_token, dtype=torch.long)
        for i in range(self.sequence_length):
            x_t_embedded = self.g_embeddings(x_t).unsqueeze(1)  # [batch_size, 1, emb_dim]
            output, hidden = self.lstm(x_t_embedded, hidden)  # [batch_size, 1, hidden_dim]
            logits = self.output_layer(output.squeeze(1))  # [batch_size, num_emb]
            prob = F.softmax(logits / self.temperature, dim=-1)  # [batch_size, num_emb]
            x_t = torch.multinomial(prob, 1).squeeze(1)  # [batch_size]
            gen_x[:, i] = x_t

        return gen_x

    def pretrain_loss(self, x):
        logits, _ = self.forward(x)  # [batch_size, sequence_length, num_emb]
        logits = logits.view(-1, self.num_emb)  # [batch_size * sequence_length, num_emb]
        targets = x.view(-1)  # [batch_size * sequence_length]
        loss = F.cross_entropy(logits, targets)
        return loss

# Example usage
if __name__ == "__main__":
    num_emb = 5000
    batch_size = 64
    emb_dim = 32
    hidden_dim = 64
    sequence_length = 20
    start_token = 0

    model = TargetLSTM(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dummy data for pretraining
    x = torch.randint(0, num_emb, (batch_size, sequence_length), dtype=torch.long)
    pretrain_loss = model.pretrain_loss(x)

    optimizer.zero_grad()
    pretrain_loss.backward()
    optimizer.step()

    # Generate sequences
    generated_sequences = model.generate()
    print(generated_sequences)