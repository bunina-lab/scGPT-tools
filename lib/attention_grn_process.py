import hashlib
import json
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

    def process(self, adata, group_key, gene_key, output_dir: str, layer_key='X_binned', threshold_weight: float = 0.0, batch_size=16):
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
            query_ids,
            batch_size
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
            if grn.empty:
                print(" -> No links passed the threshold; skipping")
                continue
            # .iloc, not [0]: after threshold filtering the index label 0 may be gone
            self.save_grn(grn, file_name=grn['group_name'].iloc[0], output_dir=output_dir)
        
        print(f"Pipeline complete. Files saved to {output_dir}")

    def process_perturbed_grn(self, adata, group_key, gene_key, output_dir: str,
                              perturb_genes, level: str = "bin:0",
                              layer_key: str = "X_binned", batch_size: int = 16,
                              threshold_weight: float = 0.0, compress: bool = False):
        """Infer the GRN with one gene's expression overwritten in silico.

        Identical to process() except that the tokenized value matrix has one
        column replaced before the attention pass. The model stays frozen and in
        eval mode throughout -- no training happens, no gradients are taken.

        Runs over ALL cells and writes one GRN per cell type per gene, in the
        same schema process() produces, so the output is directly comparable to
        an ordinary full-run GRN. One forward pass per gene.
        """
        from lib.insilico_perturb import (
            apply_perturbation, gene_token_position, level_slug,
        )

        print("1. Preprocessing AnnData and Tokenizing...")
        tokenized_data, padding_mask, condition_ids, query_ids, group_mapping, \
            group_counts, genes, tfs = self.preprocess_adata(
                adata=adata, gene_key=gene_key, group_key=group_key,
                layer_key=layer_key,
            )

        max_bin = float(tokenized_data["values"].max())
        missing = [g for g in perturb_genes if g not in set(genes)]
        if missing:
            raise KeyError(f"Genes not in the adata var set: {missing}")

        slug = level_slug(level)
        for n, gene in enumerate(perturb_genes, start=1):
            position = gene_token_position(genes, gene)
            values, frac = apply_perturbation(
                tokenized_data["values"], position, level, max_bin,
            )
            print(f"[{n}/{len(perturb_genes)}] {gene} -> {level} "
                  f"at token {position}: {frac:.1%} of {adata.n_obs} cells changed")
            if frac == 0.0 and level != "sham":
                print(f"  WARNING: {gene} is already at the target in every cell, "
                      f"so this GRN will be identical to the unperturbed one.")

            print("2. Querying Model for Attention Scores...")
            dict_sum_condition = self.query_attention_score(
                tokenized_data["genes"], values, padding_mask,
                condition_ids, query_ids, batch_size,
            )

            print("3. Processing GRNs and saving to disk...")
            for grn in self.process_grn(
                tfs, genes, dict_sum_condition, group_mapping,
                group_counts, threshold_weight,
            ):
                if grn.empty:
                    print(" -> No links passed the threshold; skipping")
                    continue
                group_name = grn['group_name'].iloc[0]
                grn['perturbed_gene'] = gene
                grn['perturb_level'] = level
                file_name = f"{gene}_{slug}_{group_name}"
                self.save_grn(grn, output_dir=output_dir, file_name=file_name,
                              compress=compress)
                self._write_manifest(
                    output_dir=output_dir, file_name=file_name, gene=gene,
                    level=level, frac=frac, group_name=group_name,
                    num_cells=int(grn['num_cells'].iloc[0]), genes=genes, tfs=tfs,
                    layer_key=layer_key,
                )

        print(f"Perturbation run complete. Files saved to {output_dir}")

    def _write_manifest(self, output_dir, file_name, gene, level, frac,
                        group_name, num_cells, genes, tfs, layer_key):
        """Provenance sidecar for a GRN file.

        Rank weights are only comparable between runs sharing an identical gene
        set, TF set, cell set and normalisation version, so a later delta step
        needs these recorded to refuse a mismatch instead of silently returning
        a plausible but wrong answer.
        """
        safe_name = str(file_name).replace(" ", "_").replace("/", "_")
        path = os.path.join(output_dir, f"{safe_name}_GRN.manifest.json")
        with open(path, "w") as fh:
            json.dump({
                "perturbed_gene": gene,
                "perturb_level": level,
                "cells_changed_fraction": frac,
                "group_name": str(group_name),
                "num_cells": num_cells,
                "n_tfs": int(len(tfs)),
                "n_genes": int(len(genes)),
                "gene_set_sha1": hashlib.sha1(
                    "\n".join(map(str, genes)).encode()).hexdigest(),
                "tf_set_sha1": hashlib.sha1(
                    "\n".join(map(str, tfs)).encode()).hexdigest(),
                "model_file": str(getattr(self.model, "model_file", None)),
                "layer_key": layer_key,
                "normalisation_version": 2,   # 2 = column rank divided by n_q
            }, fh, indent=2)

    def preprocess_adata(self, adata, group_key, gene_key, layer_key):
        """
        Gets genes, asks for gene ids from vocab, tokenizes gene_ids,
        and gets/prepares condition ids.
        """
        # 1. Get genes and identify TFs
        genes = self._get_genes_from_adata(adata, gene_key)

        if self.tf_names:
            ##print(self.tf_names.intersection(genes))
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
        
        #print(len(tf_indices_tokenized))
        
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

    def query_attention_score(self, tokenized_genes, tokenized_values, padding_mask, condition_ids, query_ids=None, batch_size=16):
        return self.model.get_attention_scores(
            all_gene_ids=tokenized_genes,
            all_values=tokenized_values,
            src_key_padding_mask=padding_mask,
            condition_ids=condition_ids,
            query_ids=query_ids,   # integer positions, CLS-corrected
            batch_size=batch_size
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
    def save_grn(grn:pd.DataFrame, output_dir:str, file_name:str, compress:bool=False):
        os.makedirs(output_dir, exist_ok=True)
        # Save to TSV. pandas compresses automatically on a .gz suffix; a dense
        # run is ~100 MB per cell type, so gzip is worth it for wide screens.
        safe_name = str(file_name).replace(" ", "_").replace("/", "_")
        ext = "tsv.gz" if compress else "tsv"
        file_path = os.path.join(output_dir, f"{safe_name}_GRN.{ext}")
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