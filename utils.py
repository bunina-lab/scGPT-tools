import os
from lib.scgpt_model import scGPTModel

def load_model(model_path, model_args_path=None, model_vocab_path=None, init_gene_embedding=False):
    model_dir = os.path.abspath(os.path.dirname(model_path))
    model_args = model_args_path if model_args_path is not None else os.path.join(model_dir, "args.json")
    model_vocab = model_vocab_path if model_vocab_path is not None else os.path.join(model_dir, "vocab.json")
    model = scGPTModel(model_file=model_path, vocab_file=model_vocab, model_config_file=model_args)
    model.process_init_model()
    if init_gene_embedding:
        model.init_gene_embedding()# Init model's gene embedding
    return model