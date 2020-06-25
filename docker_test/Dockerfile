FROM continuumio/miniconda3
ARG SUITE2P_TAG=0.7.5

RUN echo "dockerfile RUN step 1: setting conda init" && \
    CONDA_BASE=$(conda info --base) && \
    . $CONDA_BASE/etc/profile.d/conda.sh && \

    echo "dockerfile RUN step 2: installing Suite2P" && \
    apt-get install -y libgl1-mesa-glx && \
    git clone https://github.com/MouseLand/suite2p && \
    cd suite2p && \
    git checkout ${SUITE2P_TAG} && \
    conda env create -f environment.yml && \
    conda activate suite2p && \
    pip install . && \

    echo "dockerfile RUN step 3: cleaning up" && \
    conda clean --all && \
    cd ../ && \
    rm -rf suite2p && \

    echo "dockerfile RUN complete"

ENTRYPOINT ["conda", "run", "-n", "suite2p"]
