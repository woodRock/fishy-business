import torch
import torch.nn as nn
import torch.nn.functional as F
from util import preprocess_dataset

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value, mask=None):
        """ Attention mechanism (Vaswani 2017)"""
        batch_size = query.shape[0]

        # Split the heads
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Energy-based models
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)

        x = self.fc_out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        # Dropout (Hinton 2012, Srivastava 2014)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GELU (Hendrycks 2016)
        x = F.gelu(self.fc1(x))
        # Dropout (Hinton 2012, Srivastava 2014)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.feed_forward = FeedForward(input_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Layer normalization (Ba 2016)
        # Pre-norm formulation (Xiong 2020, Karpathy 2023)
        x_norm = self.norm1(x)
        atttention = self.self_attention(x_norm, x_norm, x_norm, mask)
        # Residual connections (He 2016)
        # Dropout (Srivastava 2014, Hinton 2012)
        x = x + self.dropout1(atttention)
        x_norm = self.norm2(x)
        feed_forward_out = self.feed_forward(x_norm)
        x = x + self.dropout2(feed_forward_out)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(input_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.cross_attention = MultiHeadAttention(input_dim, num_heads)
        self.feed_forward = FeedForward(input_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """ Attention mechanism (Vasawin 2017)"""
        # Layer normalization (Ba 2016)
        # Pre-norm formulation (Xiong 2020, Karpathy 2023)
        x_norm = self.norm1(x)
        # Self attention (Vaswani 2017)
        attention = self.self_attention(x_norm, x_norm, x_norm, tgt_mask)
        # Residual connections (He 2016)
        # Dropout (Srivastava 2014, Hinton 2012)
        x = x + self.dropout1(attention)
        x_norm = self.norm2(x)
        # Cross attention (Vaswani 2017)
        cross_attention = self.cross_attention(x_norm, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(cross_attention)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(input_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    """
    References:
    1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
        A. N., ... & Polosukhin, I. (2017).
        Attention is all you need.
        Advances in neural information processing systems, 30.
    2. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
        Deep residual learning for image recognition.
        In Proceedings of the IEEE conference on
        computer vision and pattern recognition (pp. 770-778).
    3. LeCun, Y. (1989). Generalization and network design strategies.
        Connectionism in perspective, 19(143-155), 18.
    4. LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard,
        R., Hubbard, W., & Jackel, L. (1989).
        Handwritten digit recognition with a back-propagation network.
        Advances in neural information processing systems, 2.
    5. LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E.,
        Hubbard, W., & Jackel, L. D. (1989).
        Backpropagation applied to handwritten zip code recognition.
        Neural computation, 1(4), 541-551.
    6. Hendrycks, D., & Gimpel, K. (2016).
        Gaussian error linear units (gelus).
        arXiv preprint arXiv:1606.08415.
    7. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).
        Layer normalization. arXiv preprint arXiv:1607.06450.
    8. Srivastava, N., Hinton, G., Krizhevsky, A.,
        Sutskever, I., & Salakhutdinov, R. (2014).
        Dropout: a simple way to prevent neural networks from overfitting.
        The journal of machine learning research, 15(1), 1929-1958.
    9. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
        I., & Salakhutdinov, R. R. (2012).
        Improving neural networks by preventing co-adaptation of feature detectors.
        arXiv preprint arXiv:1207.0580.
    10. Glorot, X., & Bengio, Y. (2010, March).
        Understanding the difficulty of training deep feedforward neural networks.
        In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256).
        JMLR Workshop and Conference Proceedings.
    11. Loshchilov, I., & Hutter, F. (2017).
        Decoupled weight decay regularization.
        arXiv preprint arXiv:1711.05101.
    12. Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
        Deep learning. MIT press, 2016.
    13. Morgan, N., & Bourlard, H. (1989).
        Generalization and parameter estimation in feedforward nets:
        Some experiments. Advances in neural information processing systems, 2.
    14. Xiong, R., Yang, Y., He, D., Zheng, K.,
        Zheng, S., Xing, C., ... & Liu, T. (2020, November).
        On layer normalization in the transformer architecture.
        In International Conference on Machine Learning (pp. 10524-10533). PMLR.
    14. Karpathy, Andrej (2023)
        Let's build GPT: from scratch, in code, spelled out.
        YouTube https://youtu.be/kCc8FmEb1nY?si=1vM4DhyqsGKUSAdV
    15. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
        Rethinking the inception architecture for computer vision.
        In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 2818-2826).
    16. He, Kaiming, et al. "Delving deep into rectifiers:
        Surpassing human-level performance on imagenet classification."
        Proceedings of the IEEE international conference on computer vision. 2015.
    17. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013).
        Exact solutions to the nonlinear dynamics of learning in
        deep linear neural networks. arXiv preprint arXiv:1312.6120.
    """

    def __init__(self, input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, num_layers, num_heads, hidden_dim, dropout)
        self.decoder = Decoder(input_dim, num_layers, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(input_dim, output_dim)

        # for name, param in self.named_parameters():
            # if 'weight' in name and param.data.dim() == 2:
                # Xavier weight initialization (Glorot 2010)
                # nn.init.xavier_uniform_(param)
                # Kaiming weight initialization (He 2016)
                # nn.init.kaiming_uniform_(param)
                # Orthogonal weight initialization (Saxe 2013)
                # nn.init.orthogonal_(param)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        x = self.encoder(src, src_mask)
        x = self.decoder(tgt, x, src_mask, tgt_mask)
        x = self.fc(x[:, 0, :])
        return x

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'transformer_checkpoint.pth'
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, train_acc, val_loss, model, verbose=False):
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Transformer model
        """
        score = -val_loss

        # Check if the model has fit the training set.
        if train_acc == 1:
            # Employ early stopping once the model has fitted training set.
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Transformer model
        """
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 1E-5 #@param {type:"integer"}
    batch_size = 64 # @param {type:"integer"}
    num_epochs = 20 # @param {type:"integer"}
    dropout = 0.2 # @param {type: "number"}
    label_smoothing = 0.1 # @param {type: "number"}
    file_path = 'transformer_checkpoint.pth' # @param {type:"string"}

    # Regularization techniques
    is_next_spectra = True # @param {type:"boolean"}
    is_masked_spectra = True # @param {type:"boolean"}
    is_data_augmentation = True # @param {type:"boolean"}
    is_early_stopping = True # @param {type:"boolean"}

    train_loader, val_loader, test_loader, train_steps, val_steps = preprocess_dataset()

