# controller
"""
The controller is a 2-layer LSTM that generates a sequence of tokens,
one token at a time, where each token is a hyperparameter choice for the
child network (filter height, filter width, stride height, stride width,
number of filters). This repeats for each layer of the child network.

The key property is that it is AUTO-REGRESSIVE: the token sampled at
step t is embedded and fed as input to step t+1. This means every
decision is conditioned on all previous decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE
#
# For each layer the controller predicts 5 tokens in this fixed order:
#   filter_height -> filter_width -> stride_height -> stride_width -> num_filters
#
# Each token has its own set of discrete choices.
# The controller picks an INDEX, and we look up the actual value here.
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_VALUES = [
    [1, 3, 5, 7],  # filter height  - 4 choices, indices 0,1,2,3
    [1, 3, 5, 7],  # filter width   - 4 choices
    [1, 2, 3],  # stride height  - 3 choices
    [1, 2, 3],  # stride width   - 3 choices
    [6, 12, 24, 36],  # num filters   - 4 choices
]  # these are pre-defined values.

# How many tokens are predicted per layer
TOKENS_PER_LAYER = len(TOKEN_VALUES)  # 5

# How many layers the controller designs
# (paper starts at 6 and increases via schedule - we fix it for simplicity)
NUM_LAYERS = 4

# Total number of prediction steps for one full architecture
TOTAL_TOKENS = NUM_LAYERS * TOKENS_PER_LAYER  # 4 * 5 = 20


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class Controller(nn.Module):
    """
    2-layer LSTM controller that samples one child network architecture.

    Parameters
    ----------
    lstm_hidden_size : int
        Number of hidden units in each LSTM layer = 35

    lstm_num_layers : int
        Number of stacked LSTM layers = 
    embedding_dim : int
        Size of the vector used to embed each sampled token before
        feeding it to the next LSTM step. Not specified in the paper — 8 (our assumption)
    """

    def __init__(
        self,
        lstm_hidden_size: int = 35,
        lstm_num_layers: int = 2,
        embedding_dim: int = 8,
    ):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers
        self.embedding_dim    = embedding_dim

        # ── 1. THE LSTM ───────────────────────────────────────────────────
        #
        # This is the core of the controller. It maintains memory across
        # all prediction steps via its hidden state (h, c).
        #
        # input_size = embedding_dim because at each step we feed in the
        # embedding of the PREVIOUSLY sampled token (auto-regressive).
        #
        # batch_first=True means tensors are shaped [batch, seq, features]
        # rather than [seq, batch, features]. We use batch=1 throughout
        # (one architecture at a time).
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # ── 2. SOFTMAX HEADS ──────────────────────────────────────────────
        #
        # "every prediction is carried out by a softmax
        # classifier". There are TOKENS_PER_LAYER=5 distinct
        # token types, each with its own vocabulary size (the size of possible inputs,
        # pre-decided in the paper). So we need 5 separate linear layers — one per token type.
        # At step t, we pick the head at index (t % TOKENS_PER_LAYER).
        # So step 0 uses head 0 (filter_height), step 1 uses head 1
        # (filter_width), ..., step 5 uses head 0 again (filter_height
        # for layer 2), and so on.
        # Each head maps the LSTM hidden state -> logits over that token's
        # vocabulary. Then softmax turns logits into probabilities.
        self.softmax_heads = nn.ModuleList(
            [  # ModuleList is just a more dynamic way of using sequential #
                nn.Linear(lstm_hidden_size, len(TOKEN_VALUES[i]))
                for i in range(TOKENS_PER_LAYER)
            ]
        )
        # After this, self.softmax_heads looks like:
        #   [0]: Linear(35 -> 4)   for filter_height
        #   [1]: Linear(35 -> 4)   for filter_width
        #   [2]: Linear(35 -> 3)   for stride_height
        #   [3]: Linear(35 -> 3)   for stride_width
        #   [4]: Linear(35 -> 4)   for num_filters

        # ── 3. EMBEDDINGS ─────────────────────────────────────────────────
        #
        # After sampling a token at step t, we embed it and use that
        # embedding as the INPUT to the LSTM at step t+1. This is the
        # auto-regressive connection Figure 2 shows.
        #
        # Again, one Embedding table per token type because each type
        # has a different vocabulary size.
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(TOKEN_VALUES[i]), embedding_dim)
            for i in range(TOKENS_PER_LAYER)
        ])
        # After this, self.embeddings looks like:
        #   [0]: Embedding(4, 8)   for filter_height
        #   [1]: Embedding(4, 8)   for filter_width
        #   [2]: Embedding(3, 8)   for stride_height
        #   [3]: Embedding(3, 8)   for stride_width
        #   [4]: Embedding(4, 8)   for num_filters

        # ── 4. START TOKEN ────────────────────────────────────────────────
        #
        # At step t=0 there is no previously sampled token to embed.
        # We need SOMETHING to feed the LSTM as the first input.
        # The cleanest solution is a learned parameter that acts as a
        # "start of sequence" signal — the controller learns what the
        # best initial input is.
        #
        # Shape: [1, 1, embedding_dim]
        #         ^  ^
        #         |  sequence length = 1 (we step one token at a time)
        #         batch size = 1
        self.start_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self._initialize_weights()
    
    # initialize weights. 
    def _initialize_weights(self):
        # Weights initialized uniformly between -0.08 and 0.08
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.08, b=0.08)


    # ─────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────

    def forward(self):
        """
        Sample one complete architecture auto-regressively.

        Steps through TOTAL_TOKENS prediction steps. At each step:
          1. Feed current input into LSTM -> get hidden state h_t
          2. Apply the correct softmax head -> probability distribution
          3. Sample one token index from that distribution
          4. Record log P(token) — needed for the REINFORCE loss later
          5. Embed the sampled token -> input for the next step

        Returns
        -------
        sampled_tokens : list[int]
            The flat list of sampled token indices, length TOTAL_TOKENS.
            Example: [2, 1, 0, 0, 3, 1, 2, 1, 0, 2, ...]
            These get decoded into actual layer configs by self.decode().

        log_probs : list[Tensor]
            log P(token_t | all previous tokens) for each step t.
            Each element is a scalar tensor WITH gradient.
            These are what we differentiate through for REINFORCE.

        entropies : list[Tensor]
            Entropy of the distribution at each step.
            Higher entropy = more exploratory / uncertain.
            Useful later as an exploration bonus but recorded for now.
        """

        # ── Initialise LSTM hidden state to zeros ─────────────────────
        #
        # nn.LSTM expects hidden state as a tuple (h_0, c_0) where:
        #   h_0 : [num_layers, batch, hidden_size]  — hidden state
        #   c_0 : [num_layers, batch, hidden_size]  — cell state
        #
        # We start from zeros — no prior information.
        device = self.start_token.device
        h_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size).to(device)
        c_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size).to(device)
        hidden = (h_0, c_0)

        # First input to the LSTM is the learned start token
        # shape: [1, 1, embedding_dim]
        current_input = self.start_token

        # Containers for outputs
        sampled_tokens = []
        log_probs      = []
        entropies      = []

        for t in range(TOTAL_TOKENS): # total tokens is basically the total timesteps (T).

            # Which of the 5 token types (hyper-parameters) are we predicting right now?
            # Cycles: 0,1,2,3,4,0,1,2,3,4,0,...
            token_type = t % TOKENS_PER_LAYER

            # ── Step the LSTM ──────────────────────────────────────────
            # current_input shape: [1, 1, embedding_dim]
            # lstm_out      shape: [1, 1, hidden_size]
            # hidden is updated in-place to become the new (h, c)
            lstm_out, hidden = self.lstm(current_input, hidden)

            # We only need the output vector, not the batch/seq dimensions
            # lstm_out[:, -1, :] gets the last (and only) time step
            # squeeze(0) removes the batch dimension
            # Result shape: [hidden_size]  i.e. [35]
            h_t = lstm_out[:, -1, :].squeeze(0)

            # ── Apply the softmax head for this token type ─────────────
            #
            # logits shape: [vocab_size]  e.g. [4] for filter_height
            logits = self.softmax_heads[token_type](h_t)

            # Convert logits to probabilities
            # probs shape: [vocab_size]
            probs = F.softmax(logits, dim=-1)

            # ── Sample one token from the distribution ─────────────────
            #
            # torch.multinomial draws one sample from a discrete
            # distribution defined by 'probs'.
            # num_samples=1 -> returns a tensor of shape [1]
            # .item() converts the 1-element tensor to a plain Python int
            token_idx = torch.multinomial(probs, num_samples=1).item()

            # ── Record log probability of the sampled token ────────────
            #
            # This is the key quantity for REINFORCE:
            #   log P(a_t | a_{<t} ; θ)
            #
            # We index into probs with the sampled token index.
            # The small epsilon (1e-8) prevents log(0).
            # The result is a SCALAR TENSOR THAT STILL HAS A GRADIENT —
            # this gradient path goes back through softmax -> linear head
            # -> LSTM -> all previous steps. That's how the controller
            # learns: by differentiating through these log probs.
            log_prob = torch.log(probs[token_idx] + 1e-8)

            # ── Record entropy of the distribution ────────────────────
            #
            # H = -Σ p * log(p)
            # High entropy means the controller is uncertain / exploring.
            # Low entropy means it's confident about its choices.
            entropy = -(probs * torch.log(probs + 1e-8)).sum()

            sampled_tokens.append(token_idx)
            log_probs.append(log_prob)
            entropies.append(entropy)

            # ── Embed sampled token -> input for next step ──────────────
            #
            # This is THE auto-regressive connection.
            # The token we just sampled gets embedded and becomes the
            # input to the LSTM at the next step t+1.
            #
            # torch.tensor([token_idx]) : shape [1]
            # .to(device)               : same device as model
            # self.embeddings[...](...) : shape [1, embedding_dim]
            # .unsqueeze(0)             : shape [1, 1, embedding_dim]
            #                            (batch=1, seq_len=1, emb_dim)
            #
            # We need [1, 1, embedding_dim] because that's what
            # nn.LSTM with batch_first=True expects.
            idx_tensor = torch.tensor([token_idx], device=device)
            embedding    = self.embeddings[token_type](idx_tensor)  # [1, emb_dim]
            current_input = embedding.unsqueeze(0)                  # [1, 1, emb_dim]

        return sampled_tokens, log_probs, entropies

    # ─────────────────────────────────────────────────────────────────────
    # DECODE
    # ─────────────────────────────────────────────────────────────────────

    def decode(self, tokens):
        """
        Convert the flat list of token indices into a list of layer
        config dicts that a child network builder can use directly.

        Parameters
        ----------
        tokens : list[int]  length = TOTAL_TOKENS

        Returns
        -------
        list of dicts, length = NUM_LAYERS
        Each dict:
            {
                'filter_h':    int,   e.g. 3
                'filter_w':    int,   e.g. 3
                'stride_h':    int,   e.g. 1
                'stride_w':    int,   e.g. 1
                'num_filters': int,   e.g. 24
            }
        """
        layers = []
        for i in range(0, len(tokens), TOKENS_PER_LAYER):
            chunk = tokens[i : i + TOKENS_PER_LAYER]
            layers.append({
                'filter_h':    TOKEN_VALUES[0][chunk[0]],
                'filter_w':    TOKEN_VALUES[1][chunk[1]],
                'stride_h':    TOKEN_VALUES[2][chunk[2]],
                'stride_w':    TOKEN_VALUES[3][chunk[3]],
                'num_filters': TOKEN_VALUES[4][chunk[4]],
            })
        return layers


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    controller = Controller(
        lstm_hidden_size=35,
        lstm_num_layers=2,
        embedding_dim=32,
    ).to(device)

    # Sample one architecture
    tokens, log_probs, entropies = controller()

    print(f"Raw tokens ({len(tokens)} total):")
    print(f"  {tokens}")

    print(f"\nDecoded architecture ({NUM_LAYERS} layers):")
    for i, layer in enumerate(controller.decode(tokens)):
        print(f"  Layer {i}: {layer}")

    print(f"\nLog probs (one per token step):")
    for i, lp in enumerate(log_probs):
        print(f"  step {i:2d} | log_prob={lp.item():.4f} | has_grad={lp.requires_grad}")

    print(f"\nMean entropy: {torch.stack(entropies).mean().item():.4f}")
    print(f"(Max possible entropy for 4-choice token: {torch.log(torch.tensor(4.0, device=device)).item():.4f})")
    print(f"(Max possible entropy for 3-choice token: {torch.log(torch.tensor(3.0, device=device)).item():.4f})")

    # Verify gradient flows back through log_probs
    # This simulates what REINFORCE will do: multiply log_probs by a reward
    # and backprop. If this works, the controller is correctly set up.
    fake_advantage = 0.1
    loss = -torch.stack(log_probs).sum() * fake_advantage
    loss.backward()
    print(f"\nGradient check:")
    print(f"  loss = {loss.item():.4f}")
    for name, param in controller.named_parameters():
        if param.grad is not None:
            print(f"  {name:40s} grad_norm={param.grad.norm().item():.6f}")

