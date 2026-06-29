#!/usr/bin/env python3

def execute(args):
    import anndata
    from lib.grn_process import GRNProcessor
    import os
    from utils import load_model
    import json
    
    print("Loading scGPT model...")
    model = load_model(
        model_path=args.model_path,
        model_args_path=args.model_args,
        model_vocab_path=args.model_vocab,
        init_gene_embedding=True
    )
    
    print(args.use_attention)

    ## Read tf list
    with open(args.tf_path, "r") as fh:
        tf_set = list(set([tf.replace('\n', '').strip() for tf in fh]))

    # Initialize Processor
    if args.use_attention:
        from lib.attention_grn_process import AttentionGRNProcessor
        annotator = AttentionGRNProcessor(
            model=model,
            tf_names=tf_set
        )
    else:
        from lib.grn_process import GRNProcessor
        annotator = GRNProcessor(
            model=model,
            minimum_similarity_threshold=args.min_similarity_threshold,
            cluster_resolution=args.cluster_resolution,
            minimum_feature_count=args.min_feature_count,
            tf_names=list(tf_set)
            )
    
    
    # Process and annotate cells
    print("Infering GRNs...")
    for adata_path in args.query_h5ad:
        out_dir_path = os.path.join(args.output_dir, os.path.basename(adata_path).split(".")[0].replace(" ", "_").strip()) 
        os.makedirs(out_dir_path, exist_ok=True)
        if args.use_attention:
            annotator.process(
                adata=anndata.read_h5ad(adata_path),
                layer_key=args.layer_key,
                gene_key=args.gene_column,
                group_key=args.cell_column,
                output_dir=out_dir_path,
                threshold_weight=args.min_similarity_threshold,
                batch_size = args.batch_size
                
            )
        else:
            annotator.process(
                adata=anndata.read_h5ad(adata_path), 
                gene_column=args.gene_column, 
                celltype_column=args.cell_column, 
                output_dir=out_dir_path
                )
    
    # Save parameters
    with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        json.dump(vars(args), f)

    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Infer Gene Regulatory Network using scGPT model from annotation data')
    
    # Required arguments
    parser.add_argument('--query-h5ad', required=True, nargs='+', help='Path(s) to query h5ad file(s)')
    parser.add_argument('--model-path', required=True, help='Path to scGPT model')
    parser.add_argument('--output-dir', required=True, help='Path to save plots and output files')
    parser.add_argument('--tf-path', required=True, help='Path to transcription file to get TF list')
    
    # Optional arguments
    parser.add_argument('--model-args', required=False, default=None, help='Path to scGPT args.json file')

    parser.add_argument('--model-vocab', required=False, default=None, help='Path to scGPT vocab.json file')

    parser.add_argument('--cell-column', default='celltype', 
                        help='Column name in query anndata containing cell type annotations')
    parser.add_argument('--gene-column', default='index',
                        help='Column name in query data containing gene names')
    parser.add_argument('--layer-key', default='X_binned', help='Anndata Layer key to be used for attention')
    
    parser.add_argument('--min-similarity-threshold', type=float, default=0.0, 
                        help='Threshold that filters out the clusters below this value')
    
    parser.add_argument('--cluster-resolution', type=float, default=4.5, 
                        help='Higher this value more clusters. Lower the number for bigger GRNs')
    
    parser.add_argument('--min-feature-count', type=int, default=9, 
                        help='Minimum number of features that should be included in a GRN')

    parser.add_argument('--use-attention', action='store_true', help='Rather to use attention layer of the transformer')

    parser.add_argument('--batch-size',type=int, default=16, help='Batch size for attention layer')
    

    execute(parser.parse_args()) 