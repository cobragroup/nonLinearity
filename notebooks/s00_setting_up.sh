conda create -n nonlin matplotlib numpy scipy pandas seaborn allendsk ipykernel
conda activate nonlin
python -m ipykernel install --user --name=nonlin
cd ..
make