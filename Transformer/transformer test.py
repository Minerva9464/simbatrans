from Transformer import *
import random

# encoder_embedding = EmbeddingLinear(512)
# #tabe __init__ farakhuni mishe
# prices = random.sample(range(1,100),10)
# input_embedding = encoder_embedding(inp=prices)
# # inja tabe __call__ bayad farakhuni beshe vali chon az nn.madule ersbari kardim tabe forward farakhuni mishe
# print(input_embedding.size())

# print(encoder_embedding.w_embedding.weight)

d_model=512
seq_len=20
d_FF=2048
h=8
p=0.1
embedding_time2vec = EmbeddingTime2Vec(d_model=d_model, f=torch.sin)
# print(torch.sin)
# print(print)
prices = random.sample(range(1,100), seq_len)
input_embedding_t2v = embedding_time2vec(inp=prices) #shey ro be sorat tabe neveshtim,pas bayad tabe __call__ inja forward farakhuni beshe. ke mesle hameja meghdar return ro mide
print(input_embedding_t2v)
print(input_embedding_t2v.shape)

print('===================================================================')


pos_enc= PositionalEncoding(len(prices), d_model)
out_pos_enc=pos_enc(input_embedding_t2v)
print(out_pos_enc)
print(out_pos_enc.shape)

print('===================================================================')


mha_encoder=MultiHeadAttention(d_model, 8)
mha_encoder_output=mha_encoder.forward(Q=out_pos_enc, K=out_pos_enc, V=out_pos_enc)
print(mha_encoder_output)

print('===================================================================')

add_and_norm=AddAndNorm(d_model)
out_add_norm=add_and_norm(out_pos_enc, mha_encoder_output)
print(out_add_norm)
print(out_add_norm.shape)
print(torch.std_mean(out_add_norm[0,:]))

print('===================================================================')

feed_forward=FeedForward(d_model, d_FF)
out_feed_forward=feed_forward(out_add_norm)
print(out_feed_forward)
print(out_feed_forward.shape)

print('===================================================================')

encoder=EncoderLayer(d_model, h, p, d_FF)
out_encoder=encoder(out_pos_enc)
print(out_encoder)
print(out_encoder.shape)
print(torch.std_mean(out_encoder[0,:]))

print('===================================================================')




