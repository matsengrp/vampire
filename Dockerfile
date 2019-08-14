FROM continuumio/anaconda3:2019.07

RUN /opt/conda/bin/conda update -y conda
RUN mkdir /vampire
COPY Dockerfile /vampire/
COPY install/ /vampire/install/
WORKDIR /vampire

# Install conda dependencies.
RUN /opt/conda/bin/conda env create -f install/environment.yml
RUN /opt/conda/bin/conda env update -n vampire -f install/environment-R.yml
RUN /opt/conda/bin/conda env create -f install/environment-olga.yml

# Install R dependencies.
RUN /opt/conda/bin/conda run -n vampire ./install/install_R_packages.sh
RUN /opt/conda/bin/conda run -n vampire R --vanilla --slave -e 'install.packages("BiocManager",repos = "http://cran.us.r-project.org"); BiocManager::install("Biostrings")'
