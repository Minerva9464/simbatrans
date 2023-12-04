import torch
import torch.nn as nn

class EmbeddingLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_embedding=nn.Linear(in_features=1,out_features=d_model)
        #ro linear ke hover mibini khodesh bayas dare. yani aval zarb dar w mishe bad ba ye adadi jam mishe
        # in zamani farakhuni mishe ke yek shey azash besazim. va 1 vorodi migire o yek khoroji mide

    def forward(self, inp: torch.tensor) -> torch.tensor:
        # in hamon tabe __call__ hast
        """ in tabe gheymat haro migir, embedingesh ro mide

        Args:
            inp (torch.tensor): batchsize az prices. ham toye encoder ham toye decoder. Aba@d: batchsize x seqlen

        Returns:
            torch.tensor: Matrise embedding. Aba@d: Batchsize x seqlen x dmodel
        """
        inp=inp.unsqueeze(-1)
        out=self.w_embedding(inp)
        return out

# ====================================================================
# ====================================================================
    
class EmbeddingTime2Vec(nn.Module):
    def __init__(self, d_model:int, f) -> None:
        super().__init__()
        self.f=f
        self.non_periodic_coeffs=nn.Linear(in_features=1, out_features=1)
        self.periodic_coeffs=nn.Linear(1, d_model-1)
        
    def forward(self, inp: torch.tensor) -> torch.tensor:
        inp=inp.unsqueeze(-1)
        __non_periodic=self.non_periodic_coeffs(inp)
        __periodic=self.f(self.periodic_coeffs(inp))
        t2v_matrix=torch.concat((__non_periodic, __periodic), dim=-1)
        return t2v_matrix

# ====================================================================
# ====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model) -> None:
        super().__init__()
        pos_encoding=torch.zeros((max_seq_len, d_model), requires_grad=False)

        pos=torch.arange(start=0, end=max_seq_len, step=1).unsqueeze(1)
        two_i=torch.arange(start=0, end=d_model-1, step=2).unsqueeze(0)
        theta=pos/(10000**(two_i/d_model))

        pos_encoding[:,::2]=torch.sin(theta)
        pos_encoding[:,1::2]=torch.cos(theta)

        self.register_buffer('pos_enc', pos_encoding.unsqueeze(0)) # eine ine ke begi: self.pos_enc = pos_encoding. vali be soorate paydar dar RAM

    def forward(self, embedding):
        # embedding: batch_size x seqlen x dmodel
        # self.pos_enc: 1 x max_seq_len x d_model
        return embedding+self.pos_enc[:, :embedding.shape[1], :]

# ====================================================================
# ====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h) -> None:
        super().__init__()
        self.h=h
        self.d_k=d_model//h
        self.d_model=d_model

        self.wQ=nn.Linear(d_model, d_model)
        self.wK=nn.Linear(d_model, d_model)        
        self.wV=nn.Linear(d_model, d_model)        
        self.wO=nn.Linear(d_model, d_model)
        
    def generate_QKV(self, Q, K, V): #in lineare ast
        return self.wQ(Q), self.wK(K), self.wV(V)

    def split_heads(self, x:torch.tensor):
        """
        Args:
            x (torch.tensor): ye matris batch_size*seqlen*dmodel hast.
        
        Output:
            batch_size*h*seqlen*dk: in ye matris 4d. yani un 8 ta ro mindaze ro ham
        """
        batch_size=x.shape[0]
        seq_len=x.shape[1]
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2).contiguous()
    
    def concat_heads(self, x):
        #x: batch_size* h*seqlen*dk
        seq_len=x.shape[2]
        batch_size=x.shape[0]

        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q (torch.tensor): batch_size, h, seqlen, dk
            K (torch.tensor): batch_size, h, seqlen, dk
            V (torch.tensor): batch_size, h, seqlen, dk
            
        Returns:
            attention: softmax(QK^T/(d_k)**1/2)V => batch_size, h, seqlen, dk
        """
        attention_scores=torch.matmul(Q,K.transpose(2,3))/(self.d_k**.5) # batch_size x h x seq_len x seq_len
        if mask is not None:
            attention_scores+=mask[:, :, :attention_scores.shape[-1], :attention_scores.shape[-1]]
        attention_scores_softmax=torch.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_scores_softmax, V)
    
    def forward(self, Q, K, V, mask=None):
        Q, K, V=self.generate_QKV(Q, K, V) # khorojie tabe ro unpack mikonim va 3 ta khoroji dare behemon barmigardone
        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)
        
        attention=self.scaled_dot_product_attention(Q, K, V, mask) #  h ta seqlen*dk mide behemon
        concated=self.concat_heads(attention)
        return self.wO(concated)
    
# ====================================================================
# ====================================================================
class AddAndNorm(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.layer_norm=nn.LayerNorm(normalized_shape=d_model)
        # self.layer_norm.requires_grad_(False)
        
    def forward(self, x, y):
        add=x+y
        return self.layer_norm(add)

# ====================================================================
# ====================================================================   
class Dropout(nn.Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.dropout=nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

# ====================================================================
# ====================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_FF, p, network_type, kernel_size=1) -> None:
        """Choose Between two Feed Forward Networks: CNN and MLP.

        Args:
            d_model: Model Dimension
            d_FF: Hideden Layer Neurons
            p: Dropout
            network_type (String): CNN or MLP
            kernel_size (int, optional): If CNN is selected, network use this value. MUST BE ODD NUMBER. Defaults to 1.
        """
        super().__init__()
        self.network_type=network_type
        
        if network_type=='CNN':
            assert kernel_size%2==1, 'Kernel Size must be odd number'

            self.feed_forward_network = nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_FF,
                    kernel_size=kernel_size,
                    padding=(kernel_size-1)//2,
                    padding_mode='circular'
                    ),
                nn.ReLU(),
                Dropout(p),
                
                nn.Conv1d(
                    in_channels=d_FF, 
                    out_channels=d_model, 
                    kernel_size=kernel_size, 
                    padding=(kernel_size-1)//2,
                    padding_mode='circular'
                    )
                )
        else:
            self.feed_forward_network=nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_FF),
                nn.ReLU(),
                Dropout(p),
                nn.Linear(in_features=d_FF, out_features=d_model),
                )
        
    def forward(self, x):
        """
        Args:
            x: batch_size, seq_len, d_model

        Returns:
            batch_size, seq_len, d_model
        """
        if self.network_type=='CNN':
            return self.feed_forward_network(x.transpose(-1,-2)).transpose(-1,-2)
        else:
            return self.feed_forward_network(x)

# ====================================================================
# ====================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, p, d_FF, network_type, kernel_size) -> None:
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model, h)
        self.add_norm=AddAndNorm(d_model)
        self.feed_forward=FeedForward(d_model, d_FF, p, network_type=network_type, kernel_size=kernel_size)
        self.dropout=Dropout(p)
        # TODO:
        # AvgPooling after feed forward

    def forward(self, x):
        out_attention=self.self_attention.forward(x, x, x)
        out_drop_mha=self.dropout(out_attention)
        out_add_norm_self_mha=self.add_norm(x, out_drop_mha)
        out_feed_forward=self.feed_forward.forward(out_add_norm_self_mha)
        out_drop_ff=self.dropout(out_feed_forward)
        out_add_norm_ff=self.add_norm(out_drop_ff, out_add_norm_self_mha)
        return out_add_norm_ff
    
# ====================================================================
# ====================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, p, d_FF, network_type, kernel_size) -> None:
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model, h)
        self.add_norm=AddAndNorm(d_model)
        self.feed_forward=FeedForward(d_model, d_FF, p, network_type=network_type, kernel_size=kernel_size)
        self.dropout=Dropout(p)
        self.cross_attention=MultiHeadAttention(d_model, h)

    def forward(self, x, out_encoder, mask=None):
        out_self_attention=self.self_attention.forward(x, x, x, mask)
        out_drop_self_mha=self.dropout(out_self_attention)
        out_add_norm_self_mha=self.add_norm(x, out_drop_self_mha)
        out_cross_attention=self.cross_attention.forward(
            Q=out_add_norm_self_mha, K=out_encoder, V=out_encoder
            )
        out_drop_cross_mha=self.dropout(out_cross_attention)
        out_add_norm_cross_mha=self.add_norm(out_drop_cross_mha, out_add_norm_self_mha)

        out_feed_forward=self.feed_forward.forward(out_add_norm_cross_mha)
        out_drop_ff=self.dropout(out_feed_forward)
        out_add_norm_ff=self.add_norm(out_drop_ff, out_add_norm_cross_mha)
        return out_add_norm_ff
    
class Transformer(nn.Module):
    def __init__(
        self, d_model, f, h, N, d_FF, p, 
        max_seqlen_encoder, max_seqlen_decoder, 
        network_type, kernel_size
        ):
        
        super().__init__()        
        if f is None: #yani lineare /yani f ro nadade pas az formoole linear mire
            self.input_embedding=EmbeddingLinear(d_model)
            self.output_embedding=EmbeddingLinear(d_model)
        else:
            self.input_embedding=EmbeddingTime2Vec(d_model, f)
            self.output_embedding=EmbeddingTime2Vec(d_model, f)

        self.positional_encoding_encoder=PositionalEncoding(max_seqlen_encoder, d_model)
        self.positional_encoding_decoder=PositionalEncoding(max_seqlen_decoder, d_model)

        self.encoder_stack=nn.ModuleList([
            EncoderLayer(d_model, h, p, d_FF, network_type, kernel_size) for _ in range(N)
        ])
        # print(self.encoder_stack[0].feed_forward.linear_layer1.weight - self.encoder_stack[1].feed_forward.linear_layer1.weight)

        self.decoder_stack=nn.ModuleList([
            DecoderLayer(d_model, h, p, d_FF, network_type, kernel_size) for _ in range(N)
        ])
        self.mask_decoder=self.generate_mask(max_seqlen_decoder)

        self.linear_final=nn.Linear(d_model, d_model)
        self.pooling_final=nn.AvgPool1d(d_model)
        # TODO:
        # nn.MaxPool(AvgPool)
        
    def generate_mask(self, seqlen):
        neg_inf=torch.ones((1, 1, seqlen, seqlen))*-1e20
        mask=torch.triu(neg_inf, diagonal=1).int()
        return mask
    
    def forward(self, inputs: torch.tensor, outputs: torch.tensor):
        out_input_embedding=self.input_embedding(inputs)
        out_pos_encoding_encoder=self.positional_encoding_encoder(out_input_embedding)
        out_output_embedding=self.output_embedding(outputs)
        out_pos_encoding_decoder=self.positional_encoding_decoder(out_output_embedding)

        out_encoder=out_pos_encoding_encoder
        for encoder in self.encoder_stack:
            encoder=encoder(out_encoder)

        out_decoder=out_pos_encoding_decoder
        for decoder in self.decoder_stack:
            out_decoder=decoder(out_decoder, out_encoder, self.mask_decoder)

        return self.pooling_final(self.linear_final(out_decoder))
    



