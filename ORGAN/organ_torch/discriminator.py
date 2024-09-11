import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def linear(input_, output_size):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = list(input_.size())
    # print(shape)
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError(
            "Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    linear_layer = nn.Linear(input_size, output_size)

    return linear_layer(input_)

# 使用示例
# input_ = torch.randn(10, 5)  # 假设输入是一个形状为 [batch, n] 的张量
# output = linear(input_, 3)
# print(output)

def highway(input_, size, num_layers=1, bias=-2.0, f=F.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    for idx in range(num_layers):
        g = f(linear(input_, size))

        t = torch.sigmoid(
            linear(input_, size) + bias)

        output = t * g + (1. - t) * input_
        input_ = output

    return output

# 使用示例
# input_ = torch.randn(10, 5) 
# output = highway(input_, size=5, num_layers=2, bias=-2.0, f=F.relu, scope='Highway')
# print(output)


class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling, highway, dropout, and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, 
                 filter_sizes, num_filters, l2_reg_lambda=1.0, wgan_reg_lambda=1.0, grad_clip=1.0):
        super(Discriminator, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.wgan_reg_lambda = wgan_reg_lambda
        self.grad_clip = grad_clip

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Convolution + maxpool layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filter, (filter_size, embedding_size)) 
            for filter_size, num_filter in zip(filter_sizes, num_filters)
        ])

        # Highway layer
        num_filters_total = sum(num_filters)
        self.highway = highway

        # Dropout layer
        self.dropout = nn.Dropout()

        # Output layer
        self.fc = nn.Linear(num_filters_total, num_classes)

    def forward(self, input_x, dropout_keep_prob):
        # Embedding layer
        embedded_chars = self.embedding(input_x)
        embedded_chars = embedded_chars.unsqueeze(1)  # Add a channel dimension

        # Convolution + ReLU + Maxpool layers
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded_chars))
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))
            pooled_outputs.append(pooled.squeeze(3))

        # Concatenate the pooled outputs and flatten
        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, h_pool.size(1))

        # Highway layer
        h_highway = self.highway(h_pool_flat, h_pool_flat.size(1), 1, 0)

        # Dropout layer
        h_drop = self.dropout(h_highway)

        # Output layer
        scores = self.fc(h_drop)

        return scores

    def compute_loss(self, input_y, scores):
        # L2 loss
        l2_loss = sum([torch.sum(param ** 2) for param in self.parameters()])
        l2_loss = self.l2_reg_lambda * l2_loss

        # Cross-entropy loss
        cross_entropy_loss = F.cross_entropy(scores, input_y)

        # Wasserstein loss (simplified, adjust according to the exact TF implementation)
        scores_neg = scores[input_y == 0]
        scores_pos = scores[input_y == 1]
        wgan_loss = torch.abs(torch.mean(scores_neg) - torch.mean(scores_pos))
        wgan_loss = self.wgan_reg_lambda * wgan_loss

        # Total loss
        total_loss = l2_loss + wgan_loss + cross_entropy_loss
        return total_loss

    def train_model(self, input_x, input_y, dropout_keep_prob, optimizer):
        optimizer.zero_grad()
        scores = self.forward(input_x, dropout_keep_prob)
        loss = self.compute_loss(input_y, scores)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        optimizer.step()

        return loss.item()

# 使用示例
discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=100, embedding_size=128,
                              filter_sizes=[3, 4, 5], num_filters=[128, 128, 128], l2_reg_lambda=1.0,
                              wgan_reg_lambda=1.0, grad_clip=1.0)
input_x = torch.randint(0, 100, (10, 20))
input_y = torch.randint(0, 2, (10,))
dropout_keep_prob = 0.5
optimizer = optim.Adam(discriminator.parameters())
loss = discriminator.train_model(input_x, input_y, dropout_keep_prob, optimizer)
print(loss)
scores = discriminator.forward(input_x, dropout_keep_prob)
print(scores)
total_loss = discriminator.compute_loss(input_y, scores)
print(total_loss)
