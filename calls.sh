set -v
#[arquivo de origem] [valor minimo] [valor maximo] [pasta destino] [coluna no csv]
python -W ignore fitter_disc.py data/discrete.csv 1 5 destDisc qtd;
#[arquivo de origem] [pasta destino] [coluna no csv]
python -W ignore fitter.py data/continuous.csv destCont duration;
