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
        self.g_params = []
        self.d_params = []
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
        self.g_params.append(self.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()
        self.g_output_unit = self.create_output_unit()
        
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
            h_t, c_tm1 = self.g_recurrent_unit(x_t.unsqueeze(1), (h_tm1, c_tm1))  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t.squeeze(1))  # batch x vocab , logits not prob
            log_prob = F.log_softmax(o_t, dim=1)
            next_token = torch.multinomial(torch.exp(log_prob), 1).view(-1)
            x_tp1 = self.g_embeddings(next_token)  # batch x emb_dim
            gen_o.append(torch.sum(F.one_hot(next_token, self.num_emb) * F.softmax(o_t, dim=1), dim=1))  # [batch_size] , prob
            gen_x.append(next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, c_tm1

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
        self.g_optimizer.zero_grad()
        g_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.g_optimizer.step()

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
    
    def init_matrix(self, shape):
        """Returns a normally initialized matrix of a given shape."""
        return torch.randn(shape) * 0.1

    def init_vector(self, shape):
        """Returns a vector of zeros of a given shape."""
        return torch.zeros(shape)
    
    def create_recurrent_unit(self):
        """Defines the recurrent process in the LSTM."""
        
        # Weights and Bias for input and hidden tensors
        self.Wi = nn.Parameter(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = nn.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = nn.Parameter(self.init_vector([self.hidden_dim]))

        self.Wf = nn.Parameter(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = nn.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = nn.Parameter(self.init_vector([self.hidden_dim]))

        self.Wog = nn.Parameter(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = nn.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = nn.Parameter(self.init_vector([self.hidden_dim]))

        self.Wc = nn.Parameter(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = nn.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = nn.Parameter(self.init_vector([self.hidden_dim]))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = hidden_memory_tm1

            # Input Gate
            i = torch.sigmoid(
                torch.matmul(x, self.Wi) + torch.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = torch.sigmoid(
                torch.matmul(x, self.Wf) + torch.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = torch.sigmoid(
                torch.matmul(x, self.Wog) + torch.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = torch.tanh(
                torch.matmul(x, self.Wc) + torch.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * torch.tanh(c)

            return current_hidden_state, c

        return unit

    def create_output_unit(self):
        """Defines the output part of the LSTM."""
        
        self.Wo = nn.Parameter(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = nn.Parameter(self.init_vector([self.num_emb]))
    
        def unit(hidden_memory_tuple):
            print(hidden_memory_tuple)
            if isinstance(hidden_memory_tuple, tuple):
                hidden_state, _ = hidden_memory_tuple
            else:
                hidden_state = hidden_memory_tuple
            logits = torch.matmul(hidden_state, self.Wo) + self.bo
            return logits
        
        return unit


    def g_optimizer(self):
        """Sets the optimizer."""
        return optim.Adam(self.parameters(), lr=self.learning_rate) 
    
############################################################################################################    
# example
import torch

def test_generator():
    # Initialize parameters for the generator
    num_emb = 100  # Vocabulary size
    batch_size = 5  # Batch size
    emb_dim = 32  # Embedding dimension
    hidden_dim = 64  # Hidden layer size
    sequence_length = 10  # Length of each generated sequence
    start_token = 0  # Start token for generation
    learning_rate = 0.001
    reward_gamma = 0.95
    grad_clip = 5.0

    # Initialize the generator
    gen = Genarator(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token, learning_rate, reward_gamma, grad_clip)

    # Create dummy input data
    x = torch.randint(0, num_emb, (batch_size, sequence_length))  # Random input for pretrain
    rewards = torch.rand(batch_size, sequence_length)  # Random rewards for unsupervised training

    # Test forward function (generation process)
    generated_samples = gen.forward(x)
    assert generated_samples.shape == (batch_size, sequence_length), "Generated sample shape mismatch"
    print("Forward pass (generation) test passed!")

    # Test supervised pretraining
    pretrain_loss = gen.pretrain(x)
    assert isinstance(pretrain_loss, float), "Pretrain loss type mismatch"
    print(f"Pretrain step loss: {pretrain_loss:.4f} (test passed!)")

    # Test unsupervised training
    g_loss = gen.unsupervised_training(x, rewards)
    assert isinstance(g_loss, float), "Unsupervised training loss type mismatch"
    print(f"Unsupervised training loss: {g_loss:.4f} (test passed!)")

    # Test generation of samples
    generated = gen.generate()
    assert generated.shape == (batch_size, sequence_length), "Generated samples shape mismatch"
    print("Sample generation test passed!")

    # Test generator step (GAN step)
    gan_loss = gen.generator_step(x, rewards)
    assert isinstance(gan_loss, float), "GAN step loss type mismatch"
    print(f"GAN step loss: {gan_loss:.4f} (test passed!)")

    # Test optimizer initialization
    optimizer = gen.g_optimizer()
    assert isinstance(optimizer, optim.Adam), "Optimizer initialization failed"
    print("Optimizer initialization test passed!")

# Run the test
test_generator()

