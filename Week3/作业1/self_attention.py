import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

def make_padding_mask(valid_lens, seq_len):
    """
    valid_lens: (B,) 每个样本有效长度
    return: (B, 1, 1, S)  True表示要mask掉(不可见)
    """
    B = valid_lens.shape[0]
    idx = np.arange(seq_len)[None, :]  # 形状是(1,S)
    mask = idx >= valid_lens[:, None]  # (B,S)
    return mask[:, None, None, :]      # (B,1,1,S)

def make_causal_mask(seq_len):
    """
    return: (1, 1, S, S)  True表示要mask掉(未来不可见)
    """
    m = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)  # 上三角
    return m[None, None, :, :]  # (1,1,S,S)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q,K,V: (B, H, S, Dh)
    mask:  (B or 1, 1, S, S) 或 (B,1,1,S) 两种都支持
          True表示要mask掉
    return: out (B,H,S,Dh), attn (B,H,S,S)
    """
    Dh = Q.shape[-1]
    scores = (Q @ K.transpose(0,1,3,2)) / np.sqrt(Dh)  # (B,H,S,S)

    if mask is not None:
        # mask可能是 (B,1,1,S) -> broadcast到 (B,H,S,S)
        # 或 (1,1,S,S) -> broadcast到 (B,H,S,S)
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores, axis=-1)        # (B,H,S,S)
    out  = attn @ V                        # (B,H,S,Dh)
    return out, attn

class SelfAttentionNumpy:
    """
    单头 self-attention：X -> Q,K,V -> attention -> out
    """
    def __init__(self, d_model, seed=0):
        rng = np.random.default_rng(seed)
        self.Wq = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.Wk = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.Wv = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def forward(self, X, padding_mask=None, causal=False):
        """
        X: (B,S,D)
        padding_mask: (B,S) True=pad位置（不可见）
        causal: 是否使用因果mask（decoder自回归）
        """
        B,S,D = X.shape
        Q = X @ self.Wq  # (B,S,D)
        K = X @ self.Wk
        V = X @ self.Wv

        # 统一成多头接口：H=1
        Q = Q[:, None, :, :]
        K = K[:, None, :, :]
        V = V[:, None, :, :]

        mask = None
        if padding_mask is not None:
            # padding_mask: (B,S) True=pad
            mask_pad = padding_mask[:, None, None, :]  # (B,1,1,S)
            mask = mask_pad if mask is None else (mask | mask_pad)

        if causal:
            mask_causal = make_causal_mask(S)  # (1,1,S,S)
            mask = mask_causal if mask is None else (mask | mask_causal)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        return out[:, 0, :, :], attn[:, 0, :, :]  # 去掉H维
