import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class Generator(nn.Module):
    """
    Class for the generative model in PyTorch.
    """
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, start_token,
                 learning_rate=0.001, reward_gamma=0.95, grad_clip=5.0):
        super(Generator, self).__init__()

        # Initialize parameters
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.reward_gamma = reward_gamma
        self.grad_clip = grad_clip

        # Embedding layer
        self.g_embeddings = nn.Embedding(num_emb, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_emb)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def init_hidden(self):
        """
        Initializes hidden states for LSTM.
        """
        h = torch.zeros(1, self.batch_size, self.hidden_dim)
        c = torch.zeros(1, self.batch_size, self.hidden_dim)
        return h, c

    def forward(self, x, hidden):
        """
        Forward pass for the generator.
        """
        # Embedding lookup
        emb = self.g_embeddings(x)  # [batch_size, seq_len, emb_dim]
        output, hidden = self.lstm(emb, hidden)  # LSTM output
        logits = self.fc(output)  # [batch_size, seq_len, num_emb]
        return logits, hidden

    def generate(self):
        """
        Generates a batch of samples.
        """
        # Initialize input and hidden states
        gen_x = torch.zeros(self.batch_size, self.sequence_length).long()
        hidden = self.init_hidden()

        # Start token input
        x = torch.tensor([self.start_token] * self.batch_size).unsqueeze(1)  # [batch_size, 1]

        # Generate sequence
        for i in range(self.sequence_length):
            logits, hidden = self.forward(x, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # [batch_size, num_emb]
            next_token = Categorical(probs).sample().unsqueeze(1)  # [batch_size, 1]
            gen_x[:, i] = next_token.squeeze()
            x = next_token

        return gen_x

    def pretrain_step(self, x):
        """
        Performs a pretraining step on the generator.
        Also known as supervised pretraining.
        """
        self.train()
        hidden = self.init_hidden()
        self.optimizer.zero_grad()

        # Forward pass
        logits, _ = self.forward(x, hidden)
        logits = logits.view(-1, self.num_emb)  # [batch_size * seq_len, num_emb]
        target = x.view(-1)  # [batch_size * seq_len]

        # Calculate loss
        loss = F.cross_entropy(logits, target)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def train_step(self, x, rewards):
        """
        Performs a training step on the generator.
        Also known as unsupervised training.
        """
        self.train()
        hidden = self.init_hidden()
        self.optimizer.zero_grad()

        # Forward pass
        logits, _ = self.forward(x, hidden)
        logits = logits.view(-1, self.num_emb)  # [batch_size * seq_len, num_emb]
        target = x.view(-1)  # [batch_size * seq_len]

        # Calculate loss with rewards
        log_probs = F.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(target, self.num_emb).float()
        loss = -torch.sum(rewards.view(-1, 1) * one_hot * log_probs) / self.batch_size

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

'''
Embedding和LSTM层：使用nn.Embedding和nn.LSTM来实现嵌入和LSTM层。
优化器：使用torch.optim.Adam作为优化器，代替TensorFlow中的AdamOptimizer。
生成序列逻辑：通过Categorical分布和softmax的输出概率来生成序列。
损失函数和反向传播：使用PyTorch的F.cross_entropy计算预训练步骤的损失，并使用负对数似然损失加上奖励来计算训练步骤的损失。
'''

# def test_generator():
#     # 定义参数
#     num_emb = 5000  # 词汇表大小
#     batch_size = 64  # 批量大小
#     emb_dim = 32  # 嵌入维度
#     hidden_dim = 64  # 隐藏层维度
#     sequence_length = 20  # 序列长度
#     start_token = 0  # 开始标记的token
#     learning_rate = 0.001  # 学习率
#     reward_gamma = 0.95  # 奖励折扣因子
#     grad_clip = 5.0  # 梯度裁剪

#     # 实例化生成器模型
#     generator = Generator(num_emb, batch_size, emb_dim, hidden_dim,
#                           sequence_length, start_token,
#                           learning_rate, reward_gamma, grad_clip)

#     # 模拟输入数据（用于预训练步骤）
#     x = torch.randint(0, num_emb, (batch_size, sequence_length))  # 随机生成序列

#     # 模拟奖励数据（用于无监督训练步骤）
#     rewards = torch.rand(batch_size, sequence_length)  # 随机奖励

#     # 预训练步骤测试
#     pretrain_loss = generator.pretrain_step(x)
#     print(f"Pretraining loss: {pretrain_loss:.4f}")

#     # 生成样本测试
#     generated_samples = generator.generate()
#     print(f"Generated samples shape: {generated_samples.shape}")

#     # 检查生成样本的形状是否正确
#     assert generated_samples.shape == (batch_size, sequence_length), "Generated samples shape is incorrect."

#     # 无监督训练步骤测试
#     g_loss = generator.train_step(generated_samples, rewards)
#     print(f"Unsupervised training loss: {g_loss:.4f}")

#     print("All tests passed successfully!")

# # 运行测试
# test_generator()