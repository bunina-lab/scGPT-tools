from lib.scgpt_model import scGPTModel
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
from anndata import AnnData
import scanpy as sc
import numpy as np
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class GRNProcessor:
    def __init__(self, model:scGPTModel, cluster_resolution=10.0, minimum_feature_count=9, minimum_similarity_threshold=0.4) -> None:
        
        self.model = model
        self.minimum_similarity_threshold =minimum_similarity_threshold
        self.cluster_resolution = cluster_resolution
        self.minimum_feature_count = minimum_feature_count

        self.gene_embed_mapping = None

    def process(self, adata:AnnData, gene_column, celltype_column, output_dir):
        ## Get clusters on the genes/features that are common
        common_features = self.get_common_features(adata, gene_col=gene_column)
        clusters = self.process_gene_clusters(common_features, louvain_res=self.cluster_resolution, minimum_feature_count=self.minimum_feature_count)
        similarity_clusters = self.get_high_similarity_clusters(clusters, self.minimum_similarity_threshold)
        if len(similarity_clusters) == 0:
            raise ValueError("No interaction passed the similarity threshold. Try lowering it")
        
        ##NEED to change the index here
        high_mgs = dict(filter(lambda i:i[0] in similarity_clusters.keys(), clusters.items()))
        self.process_metagenes_heatmap(
            metagenes=high_mgs,
            adata=adata,
            ct_column=celltype_column,
            save=os.path.join(output_dir,"High_Sim_Heatmap.png")
        )
        for cluster_id, similarity_df in similarity_clusters.items():
            self.process_network_graph(
                similarity_df=similarity_df,
                save=os.path.join(output_dir, f"{cluster_id}_GRN.png"),
                bold_thresh=self.minimum_similarity_threshold
                )

    def get_common_features(self, adata, gene_col="index"):
        adata_genes = set(adata.var.index) if gene_col == "index" else set(adata.var[gene_col])
        model_genes = set(self.model.get_pretrained_genes())
        common_features = model_genes.intersection(adata_genes)
        if len(common_features) == 0:
            raise ValueError("No common feature between model vocab and adata genes")
        print(f'Retrieved gene embeddings for {len(common_features)} genes from {len(adata_genes)} of data genes')
        return common_features
    
    def process_gene_clusters(self,features, louvain_res=20, minimum_feature_count=10)->dict:
        gene_embedding_mappings = self.get_gene_embeddings(features)
        clusters = self.get_embedded_gene_clusters(gene_embedding_mappings, louvain_res=louvain_res)
        return self.filter_gene_clusters(clusters, minimum_feature_count)
    
    def process_metagenes_heatmap(self,metagenes, adata, ct_column, save):
        scores_adata = self.calc_score_metagenes(adata, metagenes)
        self.plot_metagenes_scores(scores_adata, metagenes, column=ct_column, plot=save)

    def get_gene_embedding(self, gene):
        return self.model.get_gene_embedding(gene)
    
    def get_gene_embeddings(self,features:list)->dict:
        if self.gene_embed_mapping is None:
            self.gene_embed_mapping = {gene : self.model.get_gene_embedding(gene) for gene in features}
        return self.gene_embed_mapping
    
    def get_embedded_gene_clusters(self, gene_embed_mapping:dict, louvain_res=20)->dict:
        return self.model.get_gene_clusters(gene_embed_mapping, louvain_res)

    @staticmethod
    def filter_gene_clusters(gene_clusters, min_feature_count=9):
        # Obtain the set of gene programs from clusters with #genes >= ...
        mgs = dict()
        for mg, genes in gene_clusters.items():
            if len(genes) >= min_feature_count:
                mgs[mg] = genes
        return mgs
    
    @staticmethod
    def get_similarity_df(gene_program:list, embeddings_dict:dict, max_similarity_threshold=0.99):
        """
        Efficiently compute pairwise cosine similarities using vectorized operations.
        
        Args:
            gene_program: List of gene names
            embeddings_dict: Dictionary mapping gene names to embedding vectors
        """
        # Get embeddings matrix (genes as rows)
        embeddings_matrix = np.array([embeddings_dict[gene] for gene in gene_program])
        
        # Compute all pairwise cosine similarities at once
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Convert to DataFrame format
        results = []
        n_genes = len(gene_program)
        
        for i in range(n_genes):
            for j in range(i + 1, n_genes):  # Only upper triangle to avoid duplicates
                similarity = similarity_matrix[i, j]
                if similarity < max_similarity_threshold:  # Filter out self-similarities
                    results.append({
                        'Gene': gene_program[j],
                        'Gene1': gene_program[i], 
                        'Similarity': similarity
                    })
        
        return pd.DataFrame(results).sort_values(by='Gene')
    
    
    def get_high_similarity_clusters(self, gene_clusters:dict, minimum_similarity=0.4):
        high_sims = {} ## leiden_index:df

        for leiden_cluster_num, gene_prog in gene_clusters.items():
            df_GP_sim = self.get_similarity_df(gene_prog, self.gene_embed_mapping)
            if df_GP_sim["Similarity"].max() > minimum_similarity:
                high_sims[leiden_cluster_num] = df_GP_sim
                #print(f'Cluster:{leiden_cluster_num}\nMax value{df_GP_sim["Similarity"].max()}')

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
        plt.close()

    @staticmethod
    def plot_metagenes_scores(adata, metagenes, column, plot=None):
        sns.set_theme(font_scale=0.9)
        plt.figure(figsize=(9, 12))
        matrix = []
        meta_genes = []
        cfnum = 1
        cfams = dict()
        for cluster, vector in metagenes.items():
            row = []
            cts = []
            for ct in set(adata.obs[column]):
                sub = adata[adata.obs[column] == ct]
                val = np.mean(sub.obs[str(cluster) + "_SCORE"].tolist())
                row.append(val)
                cts.append(ct)
            matrix.append(row)
            label = f"{str(cluster)}-{';'.join(vector[:3])}"
            if len(set(vector)) > 3:
                label += "*"
            meta_genes.append(label)
            cfams[cluster] = label
            cfnum += 1
        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, index=meta_genes, columns=cts)
        plt.figure()
        sns.clustermap(
            df,
            figsize=(9, 12),
            dendrogram_ratio=0.22,
            cmap="mako",
            yticklabels=True,
            standard_scale=5,
        )
        plt.tight_layout()
        if plot:
            plt.savefig(plot)
            plt.close()

    @staticmethod
    def calc_score_metagenes(adata, metagenes):
        """
        Score metagenes and normalize scores using MinMaxScaler.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data matrix
        metagenes : dict
            Dictionary mapping metagene names to gene lists
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        tmp_adata = adata.copy()
        
        for pathway, genes in metagenes.items():
            score_name = f"{pathway}_SCORE"
            
            try:
                # Score genes using scanpy
                sc.tl.score_genes(tmp_adata, score_name=score_name, gene_list=genes)
                
                # Get scores as numpy array directly
                scores = tmp_adata.obs[score_name].values.reshape(-1, 1)
                
                # Fit and transform in place, then flatten
                tmp_adata.obs[score_name] = scaler.fit_transform(scores).flatten()
                
            except Exception as e:
                # Set all scores to 0.0 for failed scoring
                print("Error occured:\n{}".format(e))
                tmp_adata.obs[score_name] = 0.0

        return tmp_adata