import json
import torch
from scgpt.tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained, set_seed
import numpy as np
import scgpt
from scgpt.tasks import GeneEmbedding
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from einops import rearrange


class scGPTModel():

    def __init__(self, model_file, vocab_file, model_config_file, device="cuda", use_fast_transformer=True) -> None:
        
        self.model_file = model_file
        self.vocab_file = vocab_file
        self.model_config_file = model_config_file
        self.device = device
        self.use_fast_transformer = use_fast_transformer

        self.model = None

        ### Model config parameters ###
        ### Updated by the load_model_configs() method ###
        self.model_configs = None
        self.vocab = None
        self.embed_size = None
        self.nheads = None
        self.d_hid = None
        self.ntokens = None
        self.nlayers = None
        self.nlayers_cls:int = 3
        self.n_cls:int = 1
        self.dropout:float=0.5
        self.pad_token:str = "<pad>"
        self.pad_value:int = 0
        self.do_mvc:bool = False
        self.do_dab:bool = False
        self.use_batch_labels:bool = False
        self.num_batch_labels:Optional[int] = None,
        self.domain_spec_batchnorm: Union[bool, str] = False
        self.input_emb_style: str = "continuous"
        self.n_input_bins: Optional[int] = None
        self.cell_emb_style: str = "cls"
        self.mvc_decoder_style: str = "inner product"
        self.ecs_threshold: float = 0.3
        self.explicit_zero_prob: bool = False
        self.use_fast_transformer: bool = False
        self.fast_transformer_backend: str = "flash"
        self.pre_norm: bool = False


        ### Gene Embedding Variables ###
        self.model_GeneEmbedding= None
        self.model_gene_ids = None
        self.model_gene_embeddings = None
        self.model_gene_id_token_dict:dict = None
        self.gene_order_dict = None

        
    
    def process_init_model(self):
        self.load_model_configs()
        self.load_vocab()
        self.init_model()

    def load_vocab(self, special_tokens:list=None, pad_token=None):
        if special_tokens is None: 
            pad_token = pad_token if pad_token is not None else self.pad_token
            special_tokens = [pad_token, "<cls>", "<eoc>"]
        # Load vocabulary
        self.vocab = GeneVocab.from_file(self.vocab_file)
        
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        
        self.vocab.set_default_index(self.vocab[pad_token])
        self.ntokens = len(self.vocab)  # size of vocabulary



    def load_model_configs(self):
        # Load model configs
        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)


    def init_model(self):
        set_seed(self.model_configs.get("seed", 42))

        # Initialize model
        self.model = TransformerModel(
            ntoken=self.ntokens,
            d_model=self.embed_size,
            nhead=self.nheads,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            nlayers_cls=self.nlayers_cls,
            n_cls=self.n_cls,
            vocab=self.vocab,
            dropout=self.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=self.do_mvc,
            do_dab=self.do_dab,
            use_batch_labels=self.use_batch_labels,
            num_batch_labels=self.num_batch_labels,
            domain_spec_batchnorm=self.domain_spec_batchnorm,
            input_emb_style=self.input_emb_style,
            n_input_bins=self.n_input_bins,
            cell_emb_style=self.cell_emb_style,
            mvc_decoder_style=self.mvc_decoder_style,
            ecs_threshold=self.ecs_threshold,
            explicit_zero_prob=self.explicit_zero_prob,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend=self.fast_transformer_backend,
            pre_norm=self.pre_norm,
        )
  
        try:
            # Load pretrained weights
            load_pretrained(self.model, torch.load(self.model_file, map_location=self.device), verbose=False)
        except:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)   

        self.model.to(self.device)
        self.model.eval()
    
    def init_gene_embedding(self):
        self.get_model_gene_embeddings()

    def get_model(self):
        return self.model

    def get_vocab(self):
        return self.vocab

    def get_config(self):
        return self.model_configs

    def get_gene_vocab(self, genes:list):
        return np.array(self.vocab(genes), dtype=int)
    
    def get_cell_embeddings(self, adata, gene_col, max_length=1200, batch_size=64, use_batch_labels=False): ##use_batch_labels requires adata.obs["batch_id"]
        genes = adata.var.index.tolist() if gene_col == "index" else adata.var[gene_col].tolist() 
        gene_ids = self.get_gene_vocab(genes)

        return scgpt.tasks.get_batch_cell_embeddings(
            adata,
            cell_embedding_mode="cls",
            model=self.get_model(),
            vocab=self.get_vocab(),
            max_length=max_length,
            batch_size=batch_size,
            model_configs=self.get_config(),
            gene_ids=gene_ids,
            use_batch_labels=use_batch_labels,
        )
    
    def get_pretrained_genes(self):
        ### Returns genes (tokens) that are already in the vocab file
        if  self.model_gene_ids is None:
            self.model_gene_ids = self.get_model_genes()
        return self.model_gene_ids 
    
    def get_gene2idx(self):
        if self.model_gene_id_token_dict is None:
            self.model_gene_id_token_dict = self.vocab.get_stoi()
        return self.model_gene_id_token_dict
    
    def get_model_genes(self):
        return np.array(list(self.get_gene2idx().keys()))
        
    def get_model_gene_tokens(self):
        return np.array(list(self.get_gene2idx().values()))
    
    def get_model_gene_embeddings(self):
        if self.model_gene_embeddings is None:
            self.model_gene_embeddings = self.model.encoder(torch.tensor(self.get_model_gene_tokens(), dtype=torch.long).to(self.device)).detach().cpu().numpy()
        return self.model_gene_embeddings
    
    def _get_gene_embedding_vectors(self, embedding_mappings:dict):
        self.model_GeneEmbedding = GeneEmbedding(embedding_mappings)
        return self.model_GeneEmbedding
    
    @staticmethod
    def _get_gdata(embed:GeneEmbedding, louvain_res=1):
        return embed.get_adata(resolution=louvain_res)
    
    @staticmethod
    def _get_metagenes(embed:GeneEmbedding, gdata):
        # Retrieve the gene clusters
        return embed.get_metagenes(gdata)
    
    def get_gene_clusters(self, embedding_mappings:dict, louvain_res=1):
        embed = self._get_gene_embedding_vectors(embedding_mappings)
        gdata = self._get_gdata(embed, louvain_res)
        return self._get_metagenes(embed, gdata)
    
    def calc_similarity_score(embed:GeneEmbedding, gene, subset=None):
        return embed.compute_similarities(gene=gene, subset=subset)
    
    def get_cosine_similarity_matrix(gene_embed_matrix:np.array):
        from sklearn.metrics.pairwise import cosine_similarity
        # Compute all pairwise cosine similarities at once
        return cosine_similarity(gene_embed_matrix)
    
    def get_gene_order_dict(self):
        if self.gene_order_dict is None:
            self.gene_order_dict = {gene : order_idx for order_idx, gene in enumerate(self.get_gene2idx())}
        return self.gene_order_dict
    
    def get_gene_embedding(self, gene):
        return self.model_gene_embeddings[self.get_gene_order_dict()[gene]]
    
    def get_gene_embedding_matrix(self, genes:list):
        # Get embeddings matrix (genes as rows)
        return np.array([self.get_gene_embedding(gene) for gene in genes])

    def get_attention_scores(self):
        ### this one requires heavy GPU memory ~15-20GB
        self.model.eval()
        batch_normaliser = self.get_batch_normaliser()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            M = all_gene_ids.size(1)
            N = all_gene_ids.size(0)
            device = next(self.model.parameters()).device
            for i in tqdm(range(0, N, batch_size)):
                batch_size = all_gene_ids[i : i + batch_size].size(0)
                outputs = np.zeros((batch_size, M, M), dtype=np.float32)
                # Replicate the operations in model forward pass
                src_embs = self.model.encoder(torch.tensor(all_gene_ids[i : i + batch_size], dtype=torch.long).to(device))
                val_embs = self.model.value_encoder(torch.tensor(all_values[i : i + batch_size], dtype=torch.float).to(device))
                total_embs = src_embs + val_embs

                total_embs = batch_normaliser(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
                # Send total_embs to attention layers for attention operations
                # Retrieve the output from second to last layer
                for layer in self.model.transformer_encoder.layers[:-2]:
                    total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask[i : i + batch_size].to(device))
                # Send total_embs to the last layer in flash-attn
                # https://github.com/HazyResearch/flash-attention/blob/1b18f1b7a133c20904c096b8b222a0916e1b3d37/flash_attn/flash_attention.py#L90
                qkv = self.model.transformer_encoder.layers[-1].self_attn.Wqkv(total_embs)
                # Retrieve q, k, and v from flast-attn wrapper
                qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.nheads)
                q = qkv[:, :, 0, :, :]
                k = qkv[:, :, 1, :, :]
                v = qkv[:, :, 2, :, :]
                # https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
                # q = [batch, gene, n_heads, n_hid]
                # k = [batch, gene, n_heads, n_hid]
                # attn_scores = [batch, n_heads, gene, gene]
                attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
                # Rank normalization by row
                attn_scores = attn_scores.reshape((-1, M))
                order = torch.argsort(attn_scores, dim=1)
                rank = torch.argsort(order, dim=1)
                attn_scores = rank.reshape((-1, self.nheads, M, M))/M
                # Rank normalization by column
                attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
                order = torch.argsort(attn_scores, dim=1)
                rank = torch.argsort(order, dim=1)
                attn_scores = (rank.reshape((-1, self.nheads, M, M))/M).permute(0, 1, 3, 2)

                # Average 8 attention heads
                attn_scores = attn_scores.mean(1)
                
                outputs = attn_scores.detach().cpu().numpy()
                
                for index in range(batch_size):
                    # Keep track of sum per condition
                    c = condition_ids[i : i + batch_size][index]
                    if c not in dict_sum_condition:
                        dict_sum_condition[c] = np.zeros((M, M), dtype=np.float32)
                    else:
                        dict_sum_condition[c] += outputs[index, :, :]
    
    def get_batch_normaliser(self):
        #### Batch normalisation 
        ## https://github.com/bowang-lab/scGPT/blob/0cd3c73779e93e999789d52b4412e6c23baaa02b/scgpt/model/model.py
        if self.domain_spec_batchnorm is True or self.domain_spec_batchnorm == "dsbn":
            use_affine = True if self.domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            normaliser = scgpt.model.dsbn.DomainSpecificBatchNorm1d(
                self.embed_size, self.num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif self.domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            normaliser = torch.nn.BatchNorm1d(self.embed_size, eps=6.1e-5)
        else:
            raise ValueError(f"Unrecognized domain_spec_batchnorm was called as\n {self.domain_spec_batchnorm}")
        
        return normaliser.to(self.device)
    
    def get_attention_score(self, q, k, M, get_average=False):
        # attn_scores = [batch, n_heads, gene, gene]
        attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        # Rank normalization by row
        attn_scores = attn_scores.reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = rank.reshape((-1, self.nheads, M, M))/M
        # Rank normalization by column
        attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = (rank.reshape((-1, self.nheads, M, M))/M).permute(0, 1, 3, 2)

        return attn_scores.mean(1).detach().cpu().numpy() if get_average else attn_scores.detach().cpu().numpy()