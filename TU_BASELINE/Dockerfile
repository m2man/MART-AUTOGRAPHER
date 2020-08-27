# Select base image (host OS)
FROM jupyter/scipy-notebook:latest

# Name environment and choose python 3.x version
ARG conda_env=python37
ARG py_ver=3.7
ARG working_dir=MART

# Set working directory inside the container
WORKDIR /${working_dir}

# Copy dependencies installation file to the container
COPY ./requirements.txt .

# Install additional libraries from conda
RUN conda create --quiet --yes -p $CONDA_DIR/envs/$conda_env python=$py_ver ipython ipykernel && \
    conda clean --all -f -y

# Install dependencies and create Python 3.x environment and link it to Jupyter
USER root
RUN $CONDA_DIR/envs/${conda_env}/bin/python -m ipykernel install --user --name=${conda_env} && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /${working_dir} && \
    fix-permissions /home/$NB_USER

RUN conda install --yes --file requirements.txt
USER $NB_USER


# Set environment path
ENV PATH $CONDA_DIR/envs/${conda_env}/bin:$PATH

# Copy source code to the container
COPY . .

# Expose jupyter public port
EXPOSE 8888

# ENTRYPOINT ["tiny", "-g", "--"]
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]