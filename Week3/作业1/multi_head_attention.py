import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

def scaled_dot_product_attention(Q, K, V, mask=None):
    Dh = Q.shape[-1]
    scores = (Q @ K.transpose(0,1,3,2)) / np.sqrt(Dh)  # (B,H,S,S)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ V
    return out, attn

def make_causal_mask(seq_len):
    m = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return m[None, None, :, :]  # (1,1,S,S)

class MultiHeadAttentionNumpy:
    """
    标准 Transformer MHA：
    X -> (Wq,Wk,Wv) -> split heads -> attention -> concat -> Wo
    """
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads
        rng = np.random.default_rng(seed)

        self.Wq = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.Wk = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.Wv = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.Wo = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def _split_heads(self, X):
        # X: (B,S,D) -> (B,H,S,Dh)
        B,S,D = X.shape
        X = X.reshape(B, S, self.h, self.dh).transpose(0,2,1,3)
        return X

    def _combine_heads(self, X):
        # X: (B,H,S,Dh) -> (B,S,D)
        B,H,S,Dh = X.shape
        X = X.transpose(0,2,1,3).reshape(B, S, H*Dh)
        return X

    def forward(self, X, padding_mask=None, causal=False):
        """
        X: (B,S,D)
        padding_mask: (B,S) True=pad
        causal: True用于decoder自回归
        """
        B,S,D = X.shape
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        mask = None
        if padding_mask is not None:
            mask_pad = padding_mask[:, None, None, :]  # (B,1,1,S)
            mask = mask_pad if mask is None else (mask | mask_pad)

        if causal:
            mask_causal = make_causal_mask(S)  # (1,1,S,S)
            mask = mask_causal if mask is None else (mask | mask_causal)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B,H,S,Dh)
        out = self._combine_heads(out)  # (B,S,D)
        out = out @ self.Wo             # (B,S,D)
        return out, attn

# quick test
if __name__ == "__main__":
    B,S,D = 2,6,12
    H = 3
    X = np.random.randn(B,S,D)
    pad_mask = np.array([
        [False,False,False,False,False,False],
        [False,False,False,True, True, True ]
    ])
    mha = MultiHeadAttentionNumpy(d_model=D, num_heads=H)

    out, attn = mha.forward(X, padding_mask=pad_mask, causal=True)
    print("out:", out.shape, "attn:", attn.shape)  # (B,S,D), (B,H,S,S)