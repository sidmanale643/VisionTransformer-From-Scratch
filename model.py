import torch
import torch.nn as nn

class config:
    def __init__(self , image_size = 224, in_channels = 3 , patch_size = 16 , d_model = 768 , n_heads = 12, n_blocks = 12 ,  eps = 1e-6 , dropout = 0.1):
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.eps = eps
        self.dropout = dropout

class PatchEmbeddings(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patches = nn.Conv2d(in_channels = config.in_channels,
                                 out_channels = config.d_model,
                                 kernel_size =(config.patch_size ,config.patch_size),
                                 stride = (config.patch_size ,config.patch_size),
                                 padding = "valid")
        self.pos_emb = nn.Embedding(self.num_patches , config.d_model)
    
    def forward(self , img):
        patches = self.patches(img)
        patch_emb = patches.flatten(2).transpose(1,2)
        pos_ids = torch.arange(0 , self.num_patches).unsqueeze(0)
        pos_embeddings = self.pos_emb(pos_ids)
        patch_embeddings = patch_emb + pos_embeddings        
        return patch_embeddings

class MHA(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.scale = self.d_k ** 0.5
        
        self.w_q = nn.Linear(config.d_model , config.d_model)
        self.w_k = nn.Linear(config.d_model , config.d_model)
        self.w_v = nn.Linear(config.d_model , config.d_model)
        self.w_o = nn.Linear(config.d_model , config.d_model)

    def forward(self , Q , K , V):
        
        batch_size , n_seq , d_model = Q.size()
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(batch_size , n_seq , self.n_heads , self.d_k).transpose(1,2)
        K = K.view(batch_size , n_seq , self.n_heads , self.d_k).transpose(1,2)
        V = V.view(batch_size , n_seq , self.n_heads , self.d_k).transpose(1,2)
    
        attention_scores = Q @ K.transpose(-2,-1) / self.scale
        attention_weights = torch.softmax(attention_scores , dim = -1)
        attention_values = attention_weights @ V
        attention_values_concat = attention_values.transpose(1,2).contiguous().view(batch_size , n_seq , d_model)
        
        attention_out = self.w_o(attention_values_concat)
        
        return attention_out
        
class MLP(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model , 4 * config.d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model , config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self , x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
          
class ViTEncoder(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(config.d_model , eps = config.eps)
        self.mha = MHA(config)
        self.norm_2 = nn.LayerNorm(config.d_model ,eps = config.eps)
        self.ffn = MLP(config)
        
    def forward(self , x ):
        residual = x 
        x = self.norm_1(x)
        x = self.mha(x , x , x)
        x = residual + x
        
        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        out = x + residual
        
        return out
        
class ViT(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.patch_embeddings= PatchEmbeddings(config)
        self.layers = nn.ModuleList([ViTEncoder(config) for _ in range(config.n_blocks)])
        
    def forward(self , x):
        patch_embeddings = self.patch_embeddings(x)
        x = patch_embeddings
        for layer in self.layers:
            x = layer(x)
        return x
