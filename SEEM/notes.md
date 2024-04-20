#install MPI
conda install -c conda-forge mpi4py mpich
pip install colorama


#install Requirements
# May need to remove MPI4PY (Conda install above)
pip install -r assets/requirements/requirements.txt

#Rest of packages

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