#install Requirements
pip install -r assets/requirements/requirements.txt

#install MPI
conda install -c conda-forge mpi4py mpich

#Rest of packages
pip install colorama
pip install jsonschema
pip install -r assets/requirements/requirements_custom.txt

#Custom Operator for deformabalbe vision encoder
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

#Run Demo
pip install gradio
pip install timm
pip install nltk
pip install Pillow==9.5.0
pip install transformers
pip install kornia