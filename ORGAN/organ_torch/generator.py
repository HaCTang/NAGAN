import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Genarator(object):
    '''
    Class for the generator network
    '''
    
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.001, reward_gamma=0.95, grad_clip=5.0):
        """Sets parameters and defines the model architecture."""

        """
        Set specified parameters
        """
        super(Genarator, self).__init__()
        
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = torch.tensor([start_token] * self.batch_size, dtype=torch.int32)
        self.learning_rate = float(learning_rate)
        self.reward_gamma = reward_gamma
        self.temperature = 1.0
        self.grad_clip = grad_clip

        # for tensorboard
        self.g_count = 0

        self.expected_reward = torch.zeros(self.sequence_length)
        self.x = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.int32)  # true data, not including start token
        self.rewards = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.float32)  # from rollout policy and discriminator

        """
        Define the model
        """
        self.g_embeddings = nn.Embedding(self.num_emb, self.emb_dim)
        self.g_recurrent_unit = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True)
        self.g_output_unit = nn.Linear(self.hidden_dim, self.num_emb)
        
    def forward(self, x):
        # Processed for batch
        inputs = self.g_embeddings(x)
        inputs = inputs.permute(1, 0, 2)  # seq_length x batch_size x emb_dim

        # Initial states
        h0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(1, self.batch_size, self.hidden_dim)

        gen_o = []
        gen_x = []

        def _g_recurrence(i, x_t, h_tm1, c_tm1):
            h_t, (h_tm1, c_tm1) = self.g_recurrent_unit(x_t.unsqueeze(1), (h_tm1, c_tm1))  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t.squeeze(1))  # batch x vocab , logits not prob
            log_prob = F.log_softmax(o_t, dim=1)
            next_token = torch.multinomial(torch.exp(log_prob), 1).squeeze(1)
            x_tp1 = self.g_embeddings(next_token)  # batch x emb_dim
            gen_o.append(torch.sum(F.one_hot(next_token, self.num_emb) * F.softmax(o_t, dim=1), dim=1))  # [batch_size] , prob
            gen_x.append(next_token)  # indices, batch_size
            return i + 1, x_tp1, h_tm1, c_tm1

        i = 0
        x_t = self.g_embeddings(self.start_token)
        while i < self.sequence_length:
            i, x_t, h0, c0 = _g_recurrence(i, x_t, h0, c0)

        self.gen_x = torch.stack(gen_x, dim=0).permute(1, 0)  # batch_size x seq_length
        
        return self.gen_x
    
    """
    Supervised Pretraining
    """

    def pretrain(self, x):
        # Processed for batch
        inputs = self.g_embeddings(x)
        inputs = inputs.permute(1, 0, 2)  # seq_length x batch_size x emb_dim

        # Initial states
        h0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(1, self.batch_size, self.hidden_dim)

        g_predictions = []
        g_logits = []

        def _pretrain_recurrence(i, x_t, h_tm1, c_tm1):
            h_t, (h_tm1, c_tm1) = self.g_recurrent_unit(x_t.unsqueeze(1), (h_tm1, c_tm1))
            o_t = self.g_output_unit(h_t.squeeze(1))
            g_predictions.append(F.softmax(o_t, dim=1))  # batch x vocab_size
            g_logits.append(o_t)  # batch x vocab_size
            x_tp1 = inputs[i]
            return i + 1, x_tp1, h_tm1, c_tm1

        i = 0
        x_t = inputs[0]
        while i < self.sequence_length:
            i, x_t, h0, c0 = _pretrain_recurrence(i, x_t, h0, c0)

        self.g_predictions = torch.stack(g_predictions, dim=0).permute(1, 0, 2)  # batch_size x seq_length x vocab_size
        self.g_logits = torch.stack(g_logits, dim=0).permute(1, 0, 2)  # batch_size x seq_length x vocab_size

        # Pretrain loss
        one_hot_x = F.one_hot(x.view(-1), self.num_emb).float()
        g_predictions_reshaped = self.g_predictions.view(-1, self.num_emb)
        pretrain_loss = -torch.sum(one_hot_x * torch.log(torch.clamp(g_predictions_reshaped, 1e-20, 1.0))) / (self.sequence_length * self.batch_size)

        # Backward and optimize
        self.optimizer.zero_grad()
        pretrain_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return pretrain_loss.item()
    
    """
    Unsupervised Training
    """

    def unsupervised_training(self, x, rewards):
        # Unsupervised Training
        one_hot_x = F.one_hot(x.view(-1), self.num_emb).float()
        g_predictions_reshaped = self.g_predictions.view(-1, self.num_emb)
        g_loss = -torch.sum(torch.sum(one_hot_x * torch.log(torch.clamp(g_predictions_reshaped, 1e-20, 1.0)), dim=1) * rewards.view(-1))

        # Backward and optimize
        self.optimizer.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return g_loss.item()

    def generate_pretrain_summary(self, x):
        pretrain_loss = self.pretrain(x)
        return self.g_count, pretrain_loss

    def generate_gan_summary(self, x, reward):
        g_loss = self.unsupervised_training(x, reward)
        return self.g_count, g_loss

    def generate(self):
        """Generates a batch of samples."""
        with torch.no_grad():
            outputs = self.forward(self.start_token.unsqueeze(0).repeat(self.batch_size, 1))
        return outputs

    def pretrain_step(self, x):
        """Performs a pretraining step on the generator."""
        pretrain_loss = self.pretrain(x)
        return pretrain_loss

    def generator_step(self, samples, rewards):
        """Performs a training step on the generator."""
        g_loss = self.unsupervised_training(samples, rewards)
        return g_loss