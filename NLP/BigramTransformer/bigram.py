import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# HYPERPARAMETERS AND GLOBAL VARIABLES
batch_size = 32
block_size = 8
max_iterations = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iterations = 200
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # logits (probabilities of one character following another, hence shape is (vocab_size, vocab_size))

    def forward(self, X, targets=None):                                     # X is a tensor of indexes into the embedding table, representing the characters within the block
        logits = self.token_embedding_table(X)                              # (batch_size, block_size, vocab_size)

        if targets == None:                                                 # this is needed for sampling (generative) mode
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
        for _ in range(max_new_tokens):                         # generate for max_new_tokens timesteps

            logits, _ = self(idx)                               # (batch_size, timestep, vocab_size)
            logits = logits[:, -1, :]                           # take the logits for the last character only - (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)                   # (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample (batch_size, 1): next characters
            idx = torch.cat((idx, idx_next), dim=1)             # (batch_size, timestep+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(params=m.parameters(), lr=learning_rate)

for iter in range(max_iterations):
    
    if iter % eval_iterations == 0:                                                             # every once in a while evaluate the loss on train and validation sets
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)                                                       # remove the gradients from previous iteration
    loss.backward()                                                                             # calculate the gradients
    optimizer.step()                        

print("".join(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), 500)[0].tolist())))