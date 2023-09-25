# Orchestral Texture Classification with Convolution

We have investigated the classification of different textural elements in orchestral symbolic music data. A simple convolutional neural network (CNN) is utilized to perform the classification task in a track-wise and bar-wise manner. Preliminary results are reported, and different training parameters, including the use of contextual data and the combination of tracks, are also discussed. 

- Piano roll dataset can be found at `/dataset`
- Modify different model settings in `train.py` and run this file to see the result 

## Dataset
- We use the dataset provided in Le, Dinh-Viet-Toan, et al. "A Corpus Describing Orchestral Texture in First Movements of Classical and Early-Romantic Symphonies." Proceedings of the 9th International Conference on Digital Libraries for Musicology. 2022. [Gitlab repo](https://gitlab.com/algomus.fr/orchestration)
- Manual data cleaning to preduce piano rolls to adress some exceptional cases in the *.musicxml* file

## Structure of this github repo


