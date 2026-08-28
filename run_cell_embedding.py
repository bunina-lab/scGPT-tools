#!/usr/bin/env python3

def execute(args):
    import anndata
    from lib.get_cell_annotations import AnnotateCells   
    import os
    import json
    import scanpy as sc

    if not os.path.exists(args.output_dir):
        print(f"Path \n {args.output_dir}\n does not exists. Creating new folder")
        os.makedirs(args.output_dir)
    
    # Check if model files exist
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if args.model_args is not None and not os.path.isfile(args.model_args):
        raise FileNotFoundError(f"Model args file not found: {args.model_args}")
    if args.model_vocab is not None and not os.path.isfile(args.model_vocab):
        raise FileNotFoundError(f"Model vocab file not found: {args.model_vocab}")

    # Check if all query files exist
    for adata_path in args.query_h5ad:
        if not os.path.isfile(adata_path):
            raise FileNotFoundError(f"Query h5ad file not found: {adata_path}")



    from utils import load_model
    print("Loading scGPT model...")
    model = load_model(
        model_path=args.model_path,
        model_args_path=args.model_args,
        model_vocab_path=args.model_vocab
    )

    # Initialize annotator
    annotator = AnnotateCells(
        model=model,
        ref_adata=None,
        ref_cell_column=None,
        ref_gene_column=None,
        embedding_layer=args.embedding_layer,
        k_nearest_neigbours=None
    )
    
    # Process and annotate cells
    print("Getting embeddings for cells...")
    for adata_path in args.query_h5ad:
        q_adata = anndata.read_h5ad(adata_path)
        q_adata.obsm[args.embedding_layer] = annotator._get_cell_embedding(
            q_adata, 
            args.gene_column
            )
       
        q_adata.write_h5ad(os.path.join(args.output_dir, f"cell_embedded_{os.path.basename(adata_path)}"))
    
    # Save parameters
    with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        json.dump(vars(args), f)

    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Annotate cells using scGPT model and reference data')
    
    # Required arguments
    parser.add_argument('--query-h5ad', required=True, nargs='+', help='Path(s) to query h5ad file(s)')
    parser.add_argument('--model-path', required=True, help='Path to scGPT model')
    parser.add_argument('--output-dir', required=True, help='Path to save annotated h5ad file')
    
    # Optional arguments
    parser.add_argument('--model-args', required=False, default=None, help='Path to scGPT args.json file')
    
    parser.add_argument('--model-vocab', required=False, default=None, help='Path to scGPT vocab.json file')

    parser.add_argument('--gene-column', default='index',
                        help='Column name in query data containing gene names')

    parser.add_argument('--embedding-layer', default='X_scGPT',
                        help='Name of the embedding layer to use')

    execute(parser.parse_args()) 
