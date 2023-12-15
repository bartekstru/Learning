import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# HYPERPARAMETERS AND GLOBAL VARIABLES
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iterations = 200
n_embed = 384                                           # dimensionality of an embedding
n_head = 6                                              # number of heads in multi headed attention
n_layer = 6
dropout = 0.2
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ENCODER AND DECODER
char_to_i = {char:i for i,char in enumerate(chars)}
i_to_char = {i:char for i,char in enumerate(chars)}
encode = lambda string: [char_to_i[char] for char in string]    # encoder inputs a string and output a list of integers
decode = lambda ints: [i_to_char[i] for i in ints]              # decoder outputs a list of integers and output a list of characters

# TRAINING AND VALIDATION SETS
data =  torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))                                        # create training and validation set with a 90/10 split
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data

    idxs = torch.randint(len(data) - block_size, (batch_size,)) # get batch size number of start indices, from which block size number of characters will be taken
    x = torch.stack([data[i:i+block_size] for i in idxs])       # (batch_size, block_size)
                                                            
    y = torch.stack([data[i+1:i+block_size+1] for i in idxs])   # within a block we have block_size number of 1-char sequences
    x, y = x.to(device), y.to(device)                           # as bigram predicts only one character at a time, for each block we will have block_size number of predictions
    return x, y                                                 # the labels are just the next characters, hence the shape of y is (batch_size, block_size)


@torch.no_grad                                  # disable gradient computation 
def estimate_loss():
    out = {}
    model.eval()                                # set the model in evaluation mode
    for split in ['train', 'val']:              # calculate average loss for both splits
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):        # averaging over many samples gives a more representative loss value
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()                               # set the model back in training mode
    return out

class Head(nn.Module):                                                              # masked multi head attention
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)                            # (n_embed, head_size)
                                                                                        # intuition for the key: what do I contain
        
        self.query = nn.Linear(n_embed, head_size, bias=False)                          # (n_embed, head_size)
                                                                                        # intuition for the query: what am I looking for
                                                                                        # key and query are used to calculate affinities between the tokens
        
        self.value = nn.Linear(n_embed, head_size, bias=False)                          # (n_embed, head_size)
                                                                                        # intuition for the query: if you find me interesting, here is what I will communicate 
                                                                                        # interesting means the query-key dot product is high for this token  
                                                                                        # the underlying token is information private to the token
                                                                                        # the value is what gets aggregated for the purpose of one particular head
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # the lower triangular matrix used for masking
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        _,T,C = x.shape                                             # linear operations on top of tokens enable automatic and optimal feature engineering
        k = self.key(x)                                             # (batch_size, block_size, head_size)                                               
        q = self.query(x)                                           # (batch_size, block_size, head_size)           
        v = self.value(x)                                           # (batch_size, block_size, head_size)       
        
        wei = q @ k.transpose(-2,-1)                                # (batch_size, block_size, head_size) @ (batch_size, head_size, block_size) -> (batch_size, block_size, block_size)
                                                                    # the dot product between the queries in keys is done to compute the affinities between the tokens
                                                                    # by doing so, the weight matrix is data dependent and every batch item will have a different weight matrix (because every batch item is a different block)
                                                                    # if q and k are aligned their dot product will have a higher value, which means that through softmax, their affinity will take on a large value
                                                                    # hence we will learn more about that specific token as opposed to other tokens in the sequence
                                                                    # wei tells us, in a data dependent manner, how much information to aggregate from any of the tokens in the past
        
        wei = wei * C**-0.5                                         # if we do not do this the variance will be on the order of head size
                                                                    # we want the weights in wei to be fairly diffused and not take large and small values
                                                                    # when wei takes on large and small values, it will eventually saturate and produce a vector with a sharp peak 
                                                                    # this peak will correspond to the maximum element in the vector (sort of one-hot encoded)
                                                                    # hence every token will aggregate information from just one previous token (not what we want)
                                                                    # as softmax will saturate, the gradients during back propagation will be small

        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))   # :T tokens is needed for generation, for normal training mode T will cover the whole block size
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)                                     # (head_size*num_heads, n_embed) = (n_embed, n_embed)
                                                                                    # just a linear transformation of the outcome of the above multi-head attention layer
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)                         # (batch_size, block_size, num_head * head_size) = (batch_size, block_size, n_embed)
                                                                                    # we want to concatenate across the channel dimension, increasing the number of features
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),                                          # the dimensionality of input and output is n_embed, and the inner-layer has dimensionality 4*n_embed (from the paper)
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head                     # due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.                 
        self.sa = MultiHeadAttention(n_head, head_size) # multi-head attention is responsible for the communication between the tokens

        self.ffwd = FeedForward(n_embed)                # feed forward network is responsible for the computation
        self.ln1 = nn.LayerNorm(n_embed)                # this is a per token transformation which normalizes the features and makes them unit mean and unit gaussian (beta and gamma, learned parameters, change that later)
                                                        # in the paper this is after the masked multi head attention, but nowadays it is more often put prior to it
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):                               
        
        x = x + self.sa(self.ln1(x))                    # (batch_size, block_size, num_heads * head_size)
                                                        # at this point we have features per every token, produced by the self-attention mechanism
        
        x = x + self.ffwd(self.ln2(x))                  # (batch_size, block_size, num_heads * head_size) 
                                                        # here we actually combine all the features with a simple non-linear operation
                                                        # each layer has two sub-layers.
                                                        # the first is a multi-head self-attention mechanism            
                                                        # the second is a simple, position-wise fully connected feed-forward network.
                                                        # a residual connection around each                           
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)                  
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)                                               # there should be a layer normalization after the last block and right before the final linear layer
        self.lm_head = nn.Linear(n_embed, vocab_size)                                   # final linear layer decoding into the vocabulary

    def forward(self, idx, targets=None):                                               # idx is a tensor of indexes into the embedding table, representing the characters within the block
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                       # (batch_size, block_size, b_embed), these are the logits - prediction of what comes next
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))         # (block_size, b_embed)
        x = tok_emb + pos_emb                                                           # (batch_size, block_size, b_embed) + (block_size, b_embed), works because of broadcasting along the batch dimension
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                        # (batch_size, block_size, vocab_size); logits, if we added a softmax they would be probabilities 
        
        if targets == None:                                                             # this is needed for sampling (generative) mode
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                                    # reshape the logits because cross-entropy requires (number_examples, logits) shape
            targets = targets.view(B*T)                                     # reshape accordingly

            loss = F.cross_entropy(logits, targets)                         # with randomly initialized weights the initial loss should be around -ln(1/65)
                                                                            # for every example we predict a random character
                                                                            # since there is one true character, we will be right roughly 1/65% of the time 
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
                                                                            # idx is (batch_size, block_size) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                                 # we need to crop idx because
                                                                            # otherwise we will get out of scope of our positional embedding table
                                                                            # positional embedding table is defined only for block_size positions

            logits, _ = self(idx_cond)                                      # (batch_size, block_size, n_embed)
            logits = logits[:, -1, :]                                       # get the logits for the last token; (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)                               # get the probabilities from the logits; (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)              # generate batch_size new tokens; (batch_size, 1) 
            idx = torch.cat((idx, idx_next), dim=1)                         # (batch_size, block_size + 1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


optimizer = torch.optim.AdamW(params=m.parameters(), lr=learning_rate)

for iter in tqdm(range(max_iters), desc="Training", unit="iteration"):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("".join(decode(m.generate(torch.zeros((1,1), dtype=torch.long), 500)[0].tolist())))