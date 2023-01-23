FROM airlab404/dl:cuda11.1_pytorch-1.8.1_conda_py38
#FROM cudrano/continual:cuda11.1_pytorch-1.8.1_avalanche0.2.1_conda_py38
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN /bin/bash -c "conda update conda"
# Install extras
COPY requirements.yml /requirements.yml
RUN /bin/bash -c ". activate py38 && conda env update -f=/requirements.yml"

# Enable jupyter
#RUN mkdir /.local
RUN chmod -R 777 /.local

# Enable avalanche
RUN mkdir /.avalanche
RUN chmod -R 777 /.avalanche

# Enable wandb
# RUN mkdir /.config
RUN chmod -R 777 /.config
RUN touch /.netrc
RUN chmod 777 /.netrc
#RUN mkdir /.cache
RUN chmod -R 777 /.cache

ARG UNAME=cudrano
ARG UID=1031
ARG GID=1031
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME

CMD ["bash"]

WORKDIR /exp
