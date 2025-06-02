from scgpt_model import scGPTModel
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List
from anndata import AnnData

class AnnotateCells:

    def __init__(self, 
                 model:scGPTModel, 
                 ref_adata, 
                 ref_cell_column, 
                 ref_gene_column="index", 
                 embedding_layer="X_scGPT",
                 k_nearest_neigbours=20,
                 ) -> None:
        self.model = model ## Model to embed the cells 
        
        self.ref_adata = ref_adata  ## Reference data to ask about the cells
        self.ref_gene_column = ref_gene_column
        self.ref_cell_column = ref_cell_column

        self.embedding_layer = embedding_layer
        self.k_nearest_neigbours = k_nearest_neigbours

        self.nn = None
    

    def init_annotator(self):
        if self.embedding_layer not in self.ref_adata.obsm:
            print("Embedding reference data")
            self.ref_adata.obsm[self.embedding_layer] = self._get_cell_embedding(self.ref_adata, self.ref_gene_column)
        
        self.nn = NearestNeighbors(n_neighbors=self.k_nearest_neigbours, metric='l2')
        self.nn.fit(self._get_embed_layer(self.ref_adata, self.embedding_layer)) 

        return self


    def _get_cell_embedding(self, adata, gene_col):
        return self.model.get_cell_embeddings(adata=adata, gene_col=gene_col)
    
    @staticmethod
    def _get_embed_layer(adata, embed_layer_key):
        return adata.obsm[embed_layer_key]
    
    def predict_cell_knn(self, q_adata):
        if self.nn is None:
            raise ValueError("Nearest neighbors model not initialized. Please call init_annotator() first.")
        
        distances, indices = self.nn.kneighbors(self._get_embed_layer(q_adata, self.embedding_layer))
        
        # Prepare lists for labels and confidence scores
        cell_labels = []
        confidence_values = []  # Store as a list instead of dictionary
        
        for i in range(len(q_adata)):
            # Get labels of nearest neighbors
            neighbor_labels = self.ref_adata.obs[self.ref_cell_column].iloc[indices[i]].values
            
            # Majority voting
            label_counts = pd.Series(neighbor_labels).value_counts()
            predicted_label = label_counts.index[0]
            confidence = label_counts.iloc[0] / self.k_nearest_neigbours  # Proportion of top vote
            
            cell_labels.append(predicted_label)
            confidence_values.append(confidence)  # Add to list in same order as cells
        
        # Add labels to your query AnnData object
        q_adata.obs['celltype'] = cell_labels
        
        # Add confidence scores - now properly aligned with the AnnData object's index
        q_adata.obs['label_confidence'] = confidence_values
        
        return q_adata
    
    def annotate_cells(self, q_adata, gene_column="index"):
        if self.embedding_layer not in q_adata.obsm:
            print("Embedding query data")
            q_adata.obsm[self.embedding_layer] = self._get_cell_embedding(q_adata, gene_column)
        
        return self.predict_cell_knn(q_adata)

        