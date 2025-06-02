#!/usr/bin/env python3

def execute(args):
    import anndata
    from scgpt_model import scGPTModel
    from get_cell_annotations import AnnotateCells   
    import os
    import json
    import scanpy as sc


    
    print("Loading scGPT model...")
    model_dir = os.path.abspath(os.path.dirname(args.model_path))
    model_args = args.model_args if args.model_args is not None else os.path.join(model_dir, "args.json")
    model_vocab = args.model_vocab if args.model_vocab is not None else os.path.join(model_dir, "vocab.json")
    model = scGPTModel(model_file=args.model_path, vocab_file=model_vocab, model_config_file=model_args)
    model.process_init_model()
    

     # Load Reference data
    print("Loading reference data...")
    ref_adata = anndata.read_h5ad(args.reference_h5ad)
    if args.sampling_frac:
        ref_adata = sc.pp.sample(ref_adata, fraction=args.sampling_frac, copy=True)

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
        q_adata = anndata.read_h5ad(adata_path)
        annotated_adata = annotator.annotate_cells(q_adata, args.q_gene_column)
        annotated_adata.write_h5ad(os.path.join(args.output_dir, f"cell_annotated_{os.path.basename(adata_path)}"))
    
    # Save parameters
    with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        json.dump(args, f)

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
    parser.add_argument('--k-neighbors', type=int, default=33,
                        help='Number of nearest neighbors to use for annotation')
    parser.add_argument('--sampling-frac', type=float, default=0.15, help='Sampling fraction of the reference data to be used for cell annotation.')
    

    execute(parser.parse_args()) 