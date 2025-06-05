# Use NVIDIA CUDA runtime as base
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget bzip2 ca-certificates git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and set up Python 3.10
ENV MINICONDA_DIR=/opt/conda
ENV PATH=$MINICONDA_DIR/bin:$PATH

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $MINICONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda install -y python=3.10 mamba && \
    conda clean -afy

# Create and activate conda env, install PyTorch, scgpt, scvi, and deps
RUN mamba install -y \
      -c pytorch \
      "pytorch=1.13.0=py3.10_cuda117_cudnn8_0" \
      "torchtext=0.14.0" && \
    pip install --no-cache-dir \
      packaging ninja \
      "flash-attn<1.0.5" \
      "numpy<2" \
      scgpt scvi && \
    rm -rf /root/.cache/pip


# When tries to install scgpt, it uninstalls torch and torchtext
RUN pip uninstall torch && pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117/
RUN pip uninstall torchtext && pip install torchtext==0.14.0 --extra-index-url https://download.pytorch.org/whl/cu117/
## For finetuning processes
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# It also requires gseapy package
RUN pip install gseapy

## Also notice, the source codes of the package should change
RUN chmod 777 ${MINICONDA_DIR}/miniconda3/lib/python3.10/site-packages/scgpt/preprocess.py
RUN sed -i "s/layer_data\.A/layer_data.toarray()/g" ${MINICONDA_DIR}/miniconda3/lib/python3.10/site-packages/scgpt/preprocess.py

###According to the issue here https://github.com/bowang-lab/scGPT/compare/main...avysogorets:scGPT:copy-layers
###scgpt overrides the previous layers  
RUN sed -i '1s/^/import deepcopy /' ${MINICONDA_DIR}/miniconda3/lib/python3.10/site-packages/scgpt/preprocess.py
RUN sed -i 's/_get_obs_rep(adata, layer=key_to_process)/deepcopy(_get_obs_rep(adata, layer=key_to_process))/g' ${MINICONDA_DIR}/miniconda3/lib/python3.10/site-packages/scgpt/preprocess.py

# (Optional) install Jupyter
RUN pip install --no-cache-dir jupyter

# Default working dir
WORKDIR /workspace

# Expose notebook port
EXPOSE 8888

# Launch Jupyter by default; override CMD to run a different entrypoint
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
