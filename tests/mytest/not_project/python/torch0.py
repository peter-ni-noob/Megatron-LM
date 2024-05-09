import torch


def padding_mask(sentence, pad_idx):
    mask = (sentence != pad_idx).int().unsqueeze(-2)  # [B, 1, L]
    return mask


def subsequent_mask(sentence):
    batch_size, seq_len = sentence.size()
    mask = 1 - torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


def test():
    # 以最简化的形式测试Transformer的两种mask
    sentence = torch.LongTensor([[1, 2, 5, 8, 3, 0]])  # batch_size=1, seq_len=3，padding_idx=0
    embedding = torch.nn.Embedding(num_embeddings=50000, embedding_dim=300, padding_idx=0)
    query = embedding(sentence)
    key = embedding(sentence)

    scores = torch.matmul(query, key.transpose(-2, -1))
    print("\nscores = \n", scores)

    mask_p = padding_mask(sentence, 0)
    mask_s = subsequent_mask(sentence)
    print("mask_p = \n", mask_p)
    print("mask_s = \n", mask_s)

    mask_encoder = mask_p
    mask_decoder = mask_p & mask_s  # 结合 padding mask 和 Subsequent mask
    print("mask_encoder = \n", mask_encoder)
    print("mask_decoder = \n", mask_decoder)

    scores_encoder = scores.masked_fill(mask_encoder == 0, -1e9)  # 对于scores，在mask==0的位置填充-1e9
    scores_decoder = scores.masked_fill(mask_decoder == 0, -1e9)  # 对于scores，在mask==0的位置填充-1e9

    print("scores_encoder = \n", scores_encoder)
    print("scores_decoder = \n", scores_decoder)

test()
