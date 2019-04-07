## Dependencies
* Python 3.6
* pytorch 0.4.1
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.24.1
* jsonlines

Other versions may also work, but the ones listed are the ones I've used

## Training a new model
Add the base directory to your python path

```export PYTHONPATH=${PYTHONPATH}:/path/to/caml-mimic/```

Edit `constants.py` and add the paths to the `saved_models` and `mimicdata` folders.

Copy in the `mimicdata` folder the files `D_ICD_DIAGNOSES.csv` and `D_ICD_PROCEDURES.csv` from your MIMIC-III database copy.

To train a new model use the script `learn/training.py`. Execute `python training.py -h` for a full list of input arguments and flags.

Use the following files as input for the model
* `notes_labeled_train.ndjson` training split of the dataset
* `notes_labeled_dev.ndjson` development split of the dataset
* `notes_labeled_test.ndjson` validation split of the dataset
* `glove.840B.300d.txt` GloVe pre-trained word vectors (available at https://nlp.stanford.edu/data/glove.840B.300d.zip)
* `vocab.csv` vocabulary of the training corpus

To train a new model run the following command

```python3 /path/to/learn/training.py /path/to/notes_labeled_train.ndjson /path/to/vocab.csv full hier_conv_attn 200 --filter-size-sents 3 --num-filter-maps-words 200 --num-filter-maps-sents 100 --dropout 0.5 --dropout-sents 0.5 --patience 10 --lr 0.0001 --criterion prec_at_8_fine --embed-file /path/to/glove.840B.300d.txt --gpu --embed-size 300 --embed-trainable --batch-size 8 --layer-norm --embed-desc```
