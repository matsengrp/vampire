FROM continuumio/anaconda3

RUN conda update -y conda
COPY . /vampire
WORKDIR /vampire

# Install conda dependencies.
RUN /opt/conda/bin/conda env create -f install/environment.yml
RUN /opt/conda/bin/conda env update -n vampire -f install/environment-R.yml
RUN /opt/conda/bin/conda env create -f install/environment-olga.yml

# Install R dependencies.
RUN git submodule update --init --recursive
RUN /opt/conda/bin/conda run -n vampire ./install/install_R_packages.sh
RUN /opt/conda/bin/conda run -n vampire R --vanilla --slave -e 'install.packages("BiocManager",repos = "http://cran.us.r-project.org"); BiocManager::install("Biostrings")'

# Here we check to make sure that all of the sumrep dependencies are installed.
# The following uses a silly hack to get the sumrep path. If it doesn't work just plug the path in.
RUN SUMREP_PATH=$(git submodule | grep sumrep | cut -d ' ' -f3); /opt/conda/bin/conda run -n vampire R --vanilla --slave -e "library(devtools); devtools::load_all('$SUMREP_PATH')"
