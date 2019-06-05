## Dependencies
* Python 3.6
* pytorch 0.4.1
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.24.1
* jsonlines

Other versions may also work, but the ones listed are the ones I've used

## Training a new model
Create a `mimicdata` folder that holds the files `D_ICD_DIAGNOSES.csv` and `D_ICD_PROCEDURES.csv` from your MIMIC-III database copy and a ```saved_models``` folder that will hold your trained models.

To train a new model use the script `training.py`. Execute `python training.py -h` for a full list of input arguments and flags.

Use the following files as input for the model
* `notes_labeled_train.ndjson` training split of the dataset
* `notes_labeled_dev.ndjson` development split of the dataset
* `notes_labeled_test.ndjson` validation split of the dataset
* `glove.840B.300d.txt` GloVe pre-trained word vectors (available at https://nlp.stanford.edu/data/glove.840B.300d.zip)
* `vocab.csv` vocabulary of the training corpus

To train a new model run the following command

```python3 training.py /path/to/notes_labeled_train.ndjson /path/to/vocab.csv full hier_conv_attn 300,125,125,125 --filter-size 3 --dilation 1,2,4 --dropout 0.2,0.2,0.2 --n-epochs 200 --patience 10 --lr 0.0001 --criterion f1_micro_fine --batch-size 8 --max-len 5200 --embed-file /path/to/glove.840B.300d.txt --embed-desc --models-dir /path/to/saved_models --data-dir /path/to/mimicdata```

To test a pre-trained model run the following command

```python3 training.py /path/to/notes_labeled_train.ndjson /path/to/vocab.csv full hier_conv_attn 300,125,125,125 --filter-size 3 --dilation 1,2,4 --dropout 0.2,0.2,0.2 --n-epochs 200 --patience 10 --lr 0.0001 --criterion f1_micro_fine --batch-size 8 --max-len 5200 --embed-file /path/to/glove.840B.300d.txt --embed-desc --models-dir /path/to/saved_models --data-dir /path/to/mimicdata --test-model /path/to/model.pth```
