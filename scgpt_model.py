import json
import torch
from scgpt.tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
import numpy as np
import scgpt
from scgpt.tasks import GeneEmbedding


class scGPTModel():

    def __init__(self, model_file, vocab_file, model_config_file, device="cuda", use_fast_transformer=True) -> None:
        
        self.model_file = model_file
        self.vocab_file = vocab_file
        self.model_config_file = model_config_file
        self.device = device
        self.use_fast_transformer = use_fast_transformer

        self.model = None
        self.vocab = None
        self.model_configs = None

        self.model_gene_ids = None
        self.model_gene_embeddings = None
        self.model_gene_id_token_dict:dict = None
    
    def process_init_model(self):
        self.load_vocab()
        self.load_model_configs()
        self.init_model()

    def load_vocab(self, special_tokens:list=None, pad_token="<pad>"):
        if special_tokens is None: 
            special_tokens = [pad_token, "<cls>", "<eoc>"]
        # Load vocabulary
        self.vocab = GeneVocab.from_file(self.vocab_file)
        
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        
        self.vocab.set_default_index(self.vocab[pad_token])


    def load_model_configs(self):
        # Load model configs
        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)


    def init_model(self):
        # Initialize model
        self.model = TransformerModel(
            seed=42,
            ntoken=len(self.vocab),
            d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"],
            d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"],
            nlayers_cls=self.model_configs["n_layers_cls"],
            n_cls=self.model_configs.get("n_cls", 1),
            vocab=self.vocab,
            dropout=self.model_configs["dropout"],
            pad_token=self.model_configs["pad_token"],
            pad_value=self.model_configs["pad_value"],
            do_mvc=self.model_configs.get("do_mvc", True),
            do_dab=self.model_configs.get("do_dab",False),
            use_batch_labels=self.model_configs.get("use_batch_labels",False),
            domain_spec_batchnorm=self.model_configs.get("domain_spec_batchnorm",False),
            explicit_zero_prob=self.model_configs.get('explicit_zero_prob', False),
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=self.model_configs.get("pre_norm",False),
        )
        # Load pretrained weights
        load_pretrained(self.model, torch.load(self.model_file, map_location=self.device), verbose=False)
        self.model.to(self.device)
        self.model.eval()

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
            self.model_gene_ids = np.array(list(self.get_gene2idx.values()))
        return self.model_gene_ids 
    
    def get_gene2idx(self):
        if self.model_gene_id_token_dict is None:
            self.model_gene_id_token_dict = self.vocab.get_stoi()
        return self.model_gene_id_token_dict
    
    def get_model_gene_embeddings(self):
        if self.model_gene_embeddings is None:
            self.model_gene_embeddings = self.model.encoder(torch.tensor(self.model_gene_ids, dtype=torch.long).to(self.device)).detach().cpu().numpy()
        return self.model_gene_embeddings
    
    @staticmethod
    def _get_gene_embedding_vectors(embedding_mappings:dict):
        return GeneEmbedding(embedding_mappings)
    
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
    
    def get_gene_order_index(self):
        return {gene : order_idx for order_idx, gene in enumerate(self.get_gene2idx())}
    
    def get_gene_embedding(self, gene):
        return self.model_gene_embeddings[self.get_gene_order_index()[gene]]