import torch
from torch import nn
import math


from pathlib import Path

## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size//patch_size)*(img_size//patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        #Perform splitting and linear projection at the same time using a convolution
        self.proj = nn.Conv2d(
            in_channels=in_chans, 
            out_channels=embed_dim, 
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        out = self.proj(x)
        out = out.flatten(2) #Flatten the resulting grid of patches
        out = out.transpose(1,2) #Transpose patches and out_channels
        return out

class ClassEmbedding(nn.Module):
    """
    Class Embedding
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.rand(1,embed_dim)) #Parameter is a learnable tensor

    def forward(self):
        return self.cls_token

class PositionalEncoding(nn.Module):
    """
    Positional Encodings
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.rand(num_patches+1, embed_dim))

    def forward(self, x):
        return torch.add(x, self.pos_embed)
    
class MLP(nn.Module):
    def __init__(
            self, 
            input_dim,  
            hidden_layer_depth, 
            output_dim, 
            activation_function=nn.GELU, 
            dropout=0.0):
        super().__init__()
        self.act = activation_function()
        self.drop = nn.Dropout(dropout)
        self.fc_first = nn.Linear(input_dim, hidden_layer_depth)
        self.fc_last = nn.Linear(hidden_layer_depth, output_dim)
    def forward(self,x):
        x = self.fc_first(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc_last(x)
        x = self.act(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            head_dim
            ):  
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        # Initial linear layer to get the qkv for every head
        self.qkv_layer = nn.Linear(self.dim,3*self.head_dim)

        #Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        #Last linear layer to get the final output value
        self.linear = nn.Linear(self.head_dim, self.dim)

    def forward(self,x):
        B, N, C = x.shape
        QKV = self.qkv_layer(x)
        Q,K,V = torch.chunk(QKV, 3, dim = -1) #Split the tensor at last layer into three equally sized parts
        
        H = Q.shape[-1]
        scaled = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(H) #We need to specify which axes to transpose as the tensors have 4 axes. 
        attn = self.softmax(scaled) 
        x = torch.matmul(attn, V)
        return x

class TransformerEncoder(nn.Module):
    """
    Implementation of Transformer Encoder from scratch
    """
    def __init__(
            self, 
            input_dim, 
            hidden_layer_depth,
            head_dim,
            num_heads,
            norm_layer=nn.LayerNorm,
            activation_function=nn.GELU, 
            dropout=0.0,
            ):
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        #Last linear layer to get the final output value
        self.attn_linear = nn.Linear(head_dim*num_heads, input_dim)

        self.attn = nn.ModuleList(
            [Attention(input_dim, head_dim) for i in range(num_heads)]
        )
        self.norm2 = norm_layer(input_dim)
        self.mlp = MLP(input_dim,hidden_layer_depth,input_dim, activation_function, dropout)

    def forward(self,x):
        out = self.norm1(x)
        out = torch.cat([attn(out) for attn in self.attn], dim = -1)
        out = self.attn_linear(out)
        x = torch.add(x,out)
        out = self.norm2(x)
        out = self.mlp(out)
        out = torch.add(x,out)
        return out


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_encoder_layers,
            hidden_layer_depth,
            head_dim,
            num_heads,
            norm_layer,
            activation_function,
            dropout,
            num_classes,
            mlp_head_hidden_layers_depth
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans, embed_dim=embed_dim)
        self.cls_embed = ClassEmbedding(embed_dim=embed_dim)
        self.pos_embed = PositionalEncoding(self.patch_embed.num_patches, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim,
                    hidden_layer_depth,
                    head_dim,
                    num_heads,
                    norm_layer,
                    activation_function,
                    dropout
                ) 
                for i in range(num_encoder_layers)
            ]
        )

        self.mlp_head = MLP(embed_dim, mlp_head_hidden_layers_depth, num_classes, activation_function, dropout)

    def forward(self,x):
        B, nc, w, h = x.shape
        image_embeddings = self.patch_embed(x) #B x N x ED
        cls_embedding = torch.zeros(B,1, self.embed_dim).to(x.device)
        embeddings = torch.cat((cls_embedding, image_embeddings), dim = 1)
        path_pos_embed = torch.add(embeddings, self.pos_embed(embeddings)) #B x N+1 x ED

        x = path_pos_embed
        for block in self.blocks:
            x = block(x)
        out = torch.select(x,1,0).squeeze()
        out = self.mlp_head(out)
        out = out.softmax(-1)
        return out


    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        if suffix is not None:
            save_dir = Path.joinpath(save_dir, suffix)

        torch.save(self.state_dict(), save_dir)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        self.load_state_dict(torch.load(path))