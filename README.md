# BD-Thesis
## Deep Learning per il riconoscimento di entita' nominate (NER)

<p align="center">
<img width="35%" src="https://i.imgur.com/Wym8abw.png">
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-2.7-blue.svg">
<img src="https://img.shields.io/badge/keras-2.0.8-green.svg">
<img src="https://img.shields.io/badge/license-MIT-blue.svg">
</p>

### Replicare l'esperimento
1) Pre-elaborazione dei dati di addestramento FIGER (Gold)
```
sh preprocess.sh
```

2) Creazione file di configurazione da placeholder (config/config.default.json). Rinominare il file togliendo il ".default" per usarlo con la configurazione standard.


3) Installare le dipendenze tramite il gestore di pacchetti python
```
pip install keras tensorflow scipy sklearn
```

4) Utilizzare Tensorflow con GPU (se possibile) per velocizzare la fase di training. Virtual Environment (venv) Ã¨ consigliato. https://www.tensorflow.org/install/install_linux#installing_with_virtualenv


5) Per iniziare la fase di addestramento (training), visualizzare il modello, salvare i pesi (weight) una volta completato il training per poter riutilizzare il modello successivamente gia' addestrato e predire/valutare tramite i dati di test:
```
python ner.py -P -S -SW
```


### CLI (Command Line Interface)

```
usage: ner.py [-h] [--load-model-weights LOAD_MODEL_WEIGHTS] [--model-summary]
              [--save-model-weights] [--predict-and-evaluate]

optional arguments:
  -h, --help            show this help message and exit
  --load-model-weights LOAD_MODEL_WEIGHTS, -LW LOAD_MODEL_WEIGHTS
                        Load model weights from a (previously saved) .h5 file
  --model-summary, -S   Print model summary after compilation
  --save-model-weights, -SW
                        Save model weights after training (into a .h5 file)
  --predict-and-evaluate, -P
                        Get predictions from the test dataset and its F1-score
```


### About
Progetto realizzato per la prova finale/tirocinio del corso di Laurea di Informatica (Computer Science), facolta' di Ing. dell'Informazione, Informatica e Statistica, Universita' Sapienza (Roma). Anno accademico 2016/17.
