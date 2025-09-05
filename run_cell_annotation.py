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
    
    # Check if reference file exists
    if not os.path.isfile(args.reference_h5ad):
        raise FileNotFoundError(f"Reference h5ad file not found: {args.reference_h5ad}")

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

     # Load Reference data
    print("Loading reference data...")
    ref_adata = anndata.read_h5ad(args.reference_h5ad)
    if args.ref_cell_column not in ref_adata.obs.columns:
        raise KeyError(f"Reference cell key not found in columns:\n{args.ref_cell_column}")
    if args.ref_gene_column not in ref_adata.var.columns:
        raise KeyError(f"Reference gene key not found in columns:\n{args.ref_gene_column}")
    
    if args.sampling_frac:
        ref_adata = sc.pp.sample(ref_adata, fraction=args.sampling_frac, copy=True)

    save_ref = args.embedding_layer not in ref_adata.obsm.keys()

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
        ref_adata=ref_adata,
        ref_cell_column=args.ref_cell_column,
        ref_gene_column=args.ref_gene_column,
        embedding_layer=args.embedding_layer,
        k_nearest_neigbours=args.k_neighbors
    )
    annotator.init_annotator()
    
    # Process and annotate cells
    print("Annotating cells...")
    for adata_path in args.query_h5ad:
        annotated_adata = annotator.annotate_cells(anndata.read_h5ad(adata_path), args.q_gene_column)
        annotated_adata.write_h5ad(os.path.join(args.output_dir, f"cell_annotated_{os.path.basename(adata_path)}"))
    
    ##Saves the reference data with the embedded layer automatically for future use
    if save_ref:
        print(f"Saving reference data with embedded layer {args.embedding_layer}.\nNext time use this file to avoid re-embedding the reference data.")
        if args.sampling_frac:
            print(f"Remember, this referenfe is downsampled to {args.sampling_frac} of the original data.")
        ref_adata.write_h5ad(os.path.join(args.output_dir, "ref_adata_embedded.h5ad"))
    # Save parameters
    with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        json.dump(vars(args), f)

    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Annotate cells using scGPT model and reference data')
    
    # Required arguments
    parser.add_argument('--query-h5ad', required=True, nargs='+', help='Path(s) to query h5ad file(s)')
    parser.add_argument('--reference-h5ad', required=True, help='Path to reference h5ad file')
    parser.add_argument('--model-path', required=True, help='Path to scGPT model')
    parser.add_argument('--output-dir', required=True, help='Path to save annotated h5ad file')
    
    # Optional arguments
    parser.add_argument('--model-args', required=False, default=None, help='Path to scGPT args.json file')
    
    parser.add_argument('--model-vocab', required=False, default=None, help='Path to scGPT vocab.json file')

    parser.add_argument('--ref-cell-column', default='celltype', 
                        help='Column name in reference data containing cell type annotations')
    parser.add_argument('--q-gene-column', default='index',
                        help='Column name in query data containing gene names')
    parser.add_argument('--ref-gene-column', default='index',
                        help='Column name in reference data containing gene names')
    parser.add_argument('--embedding-layer', default='X_scGPT',
                        help='Name of the embedding layer to use')
    parser.add_argument('--k-neighbors', type=int, default=20,
                        help='Number of nearest neighbors to use for annotation')
    parser.add_argument('--sampling-frac', type=float, default=0.15, help='Sampling fraction of the reference data to be used for cell annotation.')
    

    execute(parser.parse_args()) 
