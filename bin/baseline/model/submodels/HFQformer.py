class Compressor(nn.Module):
    def __init__(self, embed_dim, num_heads, num_query, n_ctx):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        self.n_ctx = n_ctx
        
        self.query = nn.Parameter(torch.randn(1, num_query, embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        
        self.q_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        self.kv_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        
        self.kv_proj = nn.Identity()
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("q_pos_embeds", self.sinusoids(num_query, embed_dim))
        self.register_buffer("kv_pos_embeds", self.sinusoids(n_ctx, embed_dim))
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.q_ln.bias, 0)
        nn.init.constant_(self.q_ln.weight, 1.0)
        nn.init.constant_(self.kv_ln.bias, 0)
        nn.init.constant_(self.kv_ln.weight, 1.0)
    
    def sinusoids(self, length, channels, max_timescale=10000):
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, x: Tensor):
        q = self.q_ln(self.query.to(x.device))
        x = self.kv_ln(self.kv_proj(x))

        q = rearrange(q + self.q_pos_embeds, 'b l (h d) -> b h l d', h=self.num_heads, d=self.head_dims)
        k = rearrange(x + self.kv_pos_embeds, 'b l (h d) -> b h l d', h=self.num_heads, d=self.head_dims)
        v = rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads, d=self.head_dims)

        attn = scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, 'b h l d -> b l (h d)')
        x = self.out_proj(attn)
        return x