import os
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from typing import Dict

class AttentionGRNProcessor:
    """
    This class gets pre-trained/fine-tuned scGPTs attention scores for specific groups in anndata obj
    TF centric approach --> asks TFs to the attention heads and saves memory by dropping non-TF rows.
    """

    def __init__(self, model, tf_names: list = None):
        self.model = model ## Initiated scGPTModel instance
        self.tf_names = tf_names if tf_names is not None else []

    def process(self, adata, group_key, gene_key, output_dir: str, layer_key='X_binned', threshold_weight: float = 0.0, ):
        """
        Main orchestrator pipeline.
        """
        print("1. Preprocessing AnnData and Tokenizing...")
        tokenized_data, padding_mask, condition_ids, query_ids, group_mapping, group_counts, genes, tfs = self.preprocess_adata(
            adata=adata,
            gene_key=gene_key,
            group_key=group_key,
            layer_key=layer_key
        )
        
        print("2. Querying Model for Attention Scores...")
        dict_sum_condition = self.query_attention_score(
            tokenized_data["genes"], 
            tokenized_data["values"], 
            padding_mask, 
            condition_ids,
            query_ids
        )
        
        print("3. Processing GRNs and saving to disk...")
        for grn in self.process_grn(
            tfs,
            genes,
            dict_sum_condition, 
            group_mapping, 
            group_counts, 
            threshold_weight
            ):
            self.save_grn(grn, file_name=grn['group_name'][0], output_dir=output_dir)
        
        print(f"Pipeline complete. Files saved to {output_dir}")

    def preprocess_adata(self, adata, group_key, gene_key, layer_key):
        """
        Gets genes, asks for gene ids from vocab, tokenizes gene_ids,
        and gets/prepares condition ids.
        """
        # 1. Get genes and identify TFs
        genes = self._get_genes_from_adata(adata, gene_key)

        if self.tf_names:
            tf_mask = np.isin(genes, self.tf_names)
            tf_indices = np.where(tf_mask)[0]
            valid_tfs = genes[tf_indices]
            print(f"Tracking {len(valid_tfs)} TFs out of {len(genes)} total genes.")
        else:
            valid_tfs = genes  # Default to all genes if no TFs specified
            print(f"No TFs provided. Tracking all {len(genes)} genes.")
    

        # 2. Get Gene IDs via Model Vocab
        gene_ids = self._get_gene_ids(genes)
        
        # 3. Extract Counts (handle sparse matrices)        
        counts = (
            adata.layers[layer_key].toarray()
            if issparse(adata.layers[layer_key])
            else adata.layers[layer_key]
        )

        # 4. Tokenize
        tokenized_data = self.model.tokenize_gene_ids(gene_ids, counts)
        assert (tokenized_data["genes"] == tokenized_data["genes"][0]).all(), \
            "Gene ordering is not consistent across cells — include_zero_gene=True required"
        
        #print(tokenized_data)
        #print(tokenized_data["genes"])
        #print(tokenized_data["genes"][0])

        # Derive TF positions directly from tokenized sequence (CLS offset handled automatically)
        pad_id = self.model.vocab[self.model.pad_token]
        tf_token_ids_raw = self.model.get_gene_vocab(valid_tfs)
        in_vocab = tf_token_ids_raw != pad_id

        if in_vocab.sum() < len(valid_tfs):
            print(f"Warning: {(~in_vocab).sum()} TFs not found in model vocab and will be skipped.")

        valid_tfs = valid_tfs[in_vocab]
        tf_token_ids   = torch.tensor(tf_token_ids_raw[in_vocab])
        tf_indices_tokenized = torch.where(
            torch.isin(tokenized_data["genes"][0], tf_token_ids)
        )[0]
        
        print(len(tf_indices_tokenized))
        
        padding_mask = tokenized_data["genes"].eq(self.model.vocab[self.model.pad_token])

        # 5. Prepare Condition IDs (Convert categorical strings to integer codes)
        groups = adata.obs[group_key].astype('category')
        condition_ids = groups.cat.codes.values
        
        # Store mapping and counts for later GRN processing
        group_mapping = {}
        group_counts = {}
        for code, category in enumerate(groups.cat.categories):
            group_mapping[code] = category
            group_counts[code] = np.sum(condition_ids == code)

        return tokenized_data, padding_mask, condition_ids, tf_indices_tokenized, group_mapping, group_counts, genes, valid_tfs

    def query_attention_score(self, tokenized_genes, tokenized_values, padding_mask, condition_ids, query_ids=None):
        return self.model.get_attention_scores(
            all_gene_ids=tokenized_genes,
            all_values=tokenized_values,
            src_key_padding_mask=padding_mask,
            condition_ids=condition_ids,
            query_ids=query_ids   # integer positions, CLS-corrected
            )
    
    def process_grn(self, tfs, genes, dict_sum_condition: Dict[int, np.array], group_mapping, group_counts, threshold_weight: float = 0.0):
        """
        Calculates mean attention, formats into TF->TG links, and saves to TSV.
        """
        for code, sum_matrix in dict_sum_condition.items():
            group_name = group_mapping[code] #self.group_mapping.get(code, f"Group_{code}")
            num_cells = group_counts[code] #self.group_counts.get(code, 1)
            
            # Calculate Mean Attention Matrix (Shape: n_TFs x n_All_Genes)
            mean_matrix = sum_matrix / num_cells
            
            # Convert to DataFrame
            df = pd.DataFrame(mean_matrix, index=tfs, columns=genes)
            
            # Melt into long format: TF | Target | Weight
            df_long = df.reset_index().melt(id_vars='index', var_name='Target_Gene', value_name='weight')
            df_long.rename(columns={'index': 'TF'}, inplace=True)
            
            ## Filter out self-loops
            ##df_long = df_long[df_long['TF'] != df_long['Target_Gene']]
            
            # Filter by threshold to save space
            if threshold_weight > 0:
                df_long = df_long[abs(df_long['weight']) >= threshold_weight]
            
            df_long['group_name'] = group_name
            df_long['num_cells'] = num_cells
            
            yield df_long
                
            
    
    @staticmethod
    def save_grn(grn:pd.DataFrame, output_dir:str, file_name:str):
        os.makedirs(output_dir, exist_ok=True)
        # Save to TSV
        safe_name = str(file_name).replace(" ", "_").replace("/", "_")
        file_path = os.path.join(output_dir, f"{safe_name}_GRN.tsv")
        grn.to_csv(file_path, sep='\t', index=False)
        print(f" -> Saved {len(grn)} links for {file_name}")


    @staticmethod
    def _get_genes_from_adata(adata, gene_key: str):
        if gene_key == "index" or gene_key is None:
            return adata.var.index.values
        return adata.var[gene_key].values
    
    @staticmethod
    def _get_groups_from_adata(adata, group_key: str):
        ## Groups can be different conditions (perturbations) or celltypes/cell subtypes
        return adata.obs[group_key].unique()

    def _get_gene_ids(self, gene_list: list):
        ## Queries model to get gene tokens (using your scGPTModel wrapper method)
        return self.model.get_gene_vocab(gene_list)