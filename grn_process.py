from scgpt_model import scGPTModel
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm

class ProcessGRN:
    def __init__(self, model:scGPTModel) -> None:
        self.model = model

        self.gene_embed_mapping = None
        self.gene_clusters = None

    def process(self, adata:AnnData, output_dir):
        clusters = self.process_gene_clusters(self.get_common_features(adata))
        similarity_clusters = self.get_high_similarity_clusters(clusters)
        for cluster_id, similarity_df in similarity_clusters.items():
            self.process_network_graph(
                similarity_df=similarity_df,
                save=os.path.join(output_dir, f"{cluster_id}_GRN.png")
                )

    def get_common_features(self, adata, gene_col="index"):
        adata_genes = set(adata.var.index) if gene_col == "index" else set(adata.var.index[gene_col])
        model_genes = set(self.model.get_pretrained_genes())
        common_features = model_genes.intersection(adata_genes)
        print(f'Retrieved gene embeddings for {len(common_features)} genes from {len(adata_genes)} of data genes')
        return common_features
    
    def process_gene_clusters(self,features, louvain_res=20, minimum_feature_count=10)->dict:
        self.get_gene_embeddings(features)
        clusters = self.get_embedded_gene_clusters(self.gene_embed_mapping, louvain_res=louvain_res)
        self.gene_clusters = self.filter_gene_clusters(clusters, minimum_feature_count)
        return self.gene_clusters


    def get_model_gene_embedding(self, gene):
        return self.model.get_gene_embedding(gene)
    
    def get_gene_embeddings(self,features)->dict:
        if self.gene_embed_mapping is None:
            self.gene_embed_mapping = {gene : self.get_model_gene_embedding(gene) for gene in features}
        return self.gene_embed_mapping
    
    def get_embedded_gene_clusters(self, gene_embed_mapping:dict, louvain_res=20):
        return self.model.get_gene_clusters(gene_embed_mapping, louvain_res)


    @staticmethod
    def filter_gene_clusters(gene_clusters, min_feature_count=9):
        # Obtain the set of gene programs from clusters with #genes >= ...
        mgs = dict()
        for mg, genes in gene_clusters.items():
            if len(genes) >= min_feature_count:
                mgs[mg] = genes
        return mgs
    
    def get_similarity_df(self, gene_clusters, max_similarity_threshold=0.99):
        # Compute cosine similarities among genes in this gene program
        df_GP = pd.DataFrame(columns=['Gene', 'Similarity', 'Gene1'])
        for i in tqdm.tqdm(gene_clusters):
            df = self.model.calc_similarity_score(i, gene_clusters)
            df['Gene1'] = i
            df_GP = pd.concat([df_GP, df], ignore_index=True)
        return df_GP[df_GP['Similarity']<max_similarity_threshold].sort_values(by='Gene') # Filter out edges from each gene to itself

    def get_high_similarity_clusters(self, gene_clusters:dict, minimum_similarity=0.4):
        high_sims = {} ## leiden_index:df

        for leiden_cluster_num, gene_prog in gene_clusters.items():
            df_GP_sim = self.get_similarity_df(gene_prog)
            if df_GP_sim["Similarity"].max() > minimum_similarity:
                high_sims[leiden_cluster_num] = df_GP_sim
                print(f'Cluster:{leiden_cluster_num}\nMax value{df_GP_sim["Similarity"].max()}')

        return high_sims
    

    def process_network_graph(self, similarity_df, bold_thresh=0.4, save=None):
        graph = self.construct_network_graph(similarity_df=similarity_df)
        self.plot_network_graph(graph, bold_thresh, save=save)


    @staticmethod
    def construct_network_graph(similarity_df:pd.DataFrame):
        # Creates a graph from the cosine similarity network
        input_node_weights = [(row['Gene'], row['Gene1'], round(row['Similarity'], 2)) for i, row in similarity_df.iterrows()]
        G = nx.Graph()
        G.add_weighted_edges_from(input_node_weights)
        return G
    
    @staticmethod
    def plot_network_graph(G, thresh=0.4, save=None):
        # Plot the cosine similarity network; strong edges (> select threshold) are highlighted
        plt.figure(figsize=(10, 10))
        widths = nx.get_edge_attributes(G, 'weight')

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > thresh]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= thresh]

        pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)

        width_large = {}
        width_small = {}
        for i, v in enumerate(list(widths.values())):
            if v > thresh:
                width_large[list(widths.keys())[i]] = v*10
            else:
                width_small[list(widths.keys())[i]] = max(v, 0)*10

        nx.draw_networkx_edges(G, pos,
                            edgelist = width_small.keys(),
                            width=list(width_small.values()),
                            edge_color='lightblue',
                            alpha=0.8)
        nx.draw_networkx_edges(G, pos,
                            edgelist = width_large.keys(),
                            width = list(width_large.values()),
                            alpha = 0.5,
                            edge_color = "blue",
                            )
        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
        d = nx.get_edge_attributes(G, "weight")
        edge_labels = {k: d[k] for k in elarge}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        #plt.show()
        plt.savefig(save)
