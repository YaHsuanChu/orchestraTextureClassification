# Orchestral Texture Classification with Convolution

We have investigated the classification of different textural elements in orchestral symbolic music data. A simple convolutional neural network (CNN) is utilized to perform the classification task in a track-wise and bar-wise manner. Preliminary results are reported, and different training parameters, including the use of contextual data and the combination of tracks, are also discussed. 
- Piano roll dataset can be found at `/dataset`
- Modify different model settings in `train.py`, then train and validate the model


## Structure of the code and files


| file / folder | description|
| -------- | -------- |
|`train.py`| trains and evaluate the model <br /> **input and training settings can be modified in this file**|
| `make_piano_rolls.py` | Parse *.musicxml* files and convert to piano rolls as numpy arrays |
| `organize_annotations.py`| Parse *.orch* files and store the labels as numpy arrays |
| `/data_processing/pianoRoll.py` | Object defined to handle, access and manipulate piano rolls |
| `/data_processing/PianoRollsDataset.py`  | Object extend to Pytorch Dataset object to manage the selection of training examples by bar|
| `/src/` | contains code for parsing *.orch* files  |
| `/model/ ` | store trained models |
| `/dataset/` | including annotations, *.musicxml*, *MIDI*, and converted pianorolls (*.npy* files) |
| `/result/` | perpformance metrics saved here |


## Dataset
- We use the dataset provided by *Le, Dinh-Viet-Toan, et al. "A Corpus Describing Orchestral Texture in First Movements of Classical and Early-Romantic Symphonies." Proceedings of the 9th International Conference on Digital Libraries for Musicology. 2022.* [[Paper]](https://dl.acm.org/doi/10.1145/3543882.3543884) [[Gitlab Repo]](https://gitlab.com/algomus.fr/orchestration)
- We parse *.musicxml* files and convert into pianorolls

### Files and folders in `/dataset`


| folder | content |
| -------- | -------- |
| `/annotaions/` |  labels of each pieces (numpy arrays) are stored in *.npy* files, the original *.orch* files are in `/{nameOfAuthor}` folders |
| `/score_xml/`  | *.musicxml* files : original digital scores    |
| `/scores_midi/` | *MIDI* files converted from *.musicxml* files are sourced |
| `/score_pianoroll/` | piano roll (numpy array) are stored as  *.npy* files   |
