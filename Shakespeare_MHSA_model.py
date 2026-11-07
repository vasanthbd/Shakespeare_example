import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
import os


# A BPE tokenizer is trained on the Shakespeare dataset
if not os.path.exists("shakespeare-tokenizer-vocab.json"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["input.txt"], vocab_size=4096, min_frequency=2)
    tokenizer.save_model(".", "shakespeare-tokenizer")
else:
    tokenizer = ByteLevelBPETokenizer(
        "shakespeare-tokenizer-vocab.json",
        "shakespeare-tokenizer-merges.txt"
    )


# Load the pre-trained tokenizer (assuming the vocab and merges files are present)
tokenizer = ByteLevelBPETokenizer(
    "shakespeare-tokenizer-vocab.json",  # vocab file
    "shakespeare-tokenizer-merges.txt"   # merges file
)



# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# obtain the ids for the data set
ids = tokenizer.encode(text).ids
data = torch.tensor(ids, dtype=torch.long)

# split the data into training and the validiation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# obtain the token from the ids
vocab_size = tokenizer.get_vocab_size()
stoi = tokenizer.get_vocab()            
itos = {i: t for t, i in stoi.items()}   

encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

batch_size = 64
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 192
n_head = 6
n_layer = 1
dropout = 0.0
# ------------

torch.manual_seed(1337)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        
        # Freeze key and value
        for param in self.key.parameters():
            param.requires_grad = False
        for param in self.value.parameters():
            param.requires_grad = False
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.decay_factor = 0.95

        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x)

        Dk = k.shape[-1]
        Dv = Dk
        λ = self.decay_factor
        
        # M_cum: Running cumulative state (B, Dv, Dk)
        M_cum = torch.zeros(B, Dv, Dk, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(T):
            # Compute outer product for current step (v_t ⊗ k_t)
            # vt: (B, Dv, 1), kt: (B, 1, Dk) -> term: (B, Dv, Dk)
            term = v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)

            # Update memory
            M_cum = λ * M_cum +  term

            # qt: (B, Dk, 1)
            qt = q[:, t].unsqueeze(-1)

            # Re-scaling by λ^t inside the loop ensures effective decay λ^{t-τ}
            yt = (M_cum) @ qt

            outputs.append(yt.squeeze(-1))

        # Stack outputs to final shape (B, T, Dv)
        out = torch.stack(outputs, dim=1)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projs = nn.ModuleList([nn.Linear(head_size, n_embd) for _ in range(num_heads)])

    def forward(self, x):
        # Each head projects separately
        head_outputs = []
        for h, proj in zip(self.heads, self.projs):
            head_out = h(x)              # (B, T, head_size)
            head_proj = proj(head_out)   # (B, T, n_embd)
            head_outputs.append(head_proj)

        # Add over heads
        out = sum(head_outputs)          # (B, T, n_embd)

        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln1 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb 
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            temperature = 0.45# Adjust between 0.1 and 2.0
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            #idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (B, 1
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



###########################################################################
model = BigramLanguageModel()
model = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Initialize optimizer and loss tracking
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
best_val_loss = float('inf')  # Start with a very high value
best_iter = 0  # Track the iteration with the best validation loss


for iter in range(5000):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log the losses to a file
        with open("loss_log.txt", "a") as f:
            f.write(f"{iter} {losses['train']:.6f} {losses['val']:.6f}\n")

        # Save the model if validation loss has improved
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_iter = iter
            torch.save(model.state_dict(), "best_model_trained.pt")
            print(f"Saved model at iteration {iter} with val loss {best_val_loss:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# After training, print the best validation loss and iteration where it was achieved
print(f"Best validation loss {best_val_loss:.4f} achieved at iteration {best_iter}")


print(decode(model.generate(context, max_new_tokens=4000)[0].tolist()))
