# make_piano_rolls.py
'''Load midi files to convert to piano rolls and save as .npz files'''

import pypianoroll 
import pandas as pd
import os
from src.parser import get_raw_segments
from src.classes import ScoreAnnot
from src.annot_types import string_to_instrument

midi_path = './dataset/scores_midi'
pianoroll_path = './dataset/scores_pianoroll'
metadata_file = './dataset/new_metadata.csv'
#metadata_file = './dataset/orch_metadata.csv'

meta_df = pd.read_csv(metadata_file) 
meta_df = meta_df.set_index(pd.Index(range(len(meta_df))))

# create path to store midi saving path
file_series = meta_df['score']
midi_file_path = []
pianoroll_file_path = []
for path in file_series:
    file_name = path.split('/')[-1].split('.')[0]
    midi_file_path.append( os.path.join(midi_path, file_name+'.mid') )
    pianoroll_file_path.append( os.path.join(pianoroll_path+'/'+file_name+'.npz') ) #Warning: what is file name to store an object

meta_df['midi'] = pd.Series(midi_file_path)
meta_df['pianoroll'] = pd.Series(pianoroll_file_path)
#meta_df.to_csv(new_metadata_file)
print(meta_df) #18 files left

'''
For each piece:
    1. load the annotaion file to get the instrument list (in order to rename the tracks)
    2. load MIDI files with pypianoroll package to convert to piano rolls
    3. change names of each tracks
    4. save a track as .npz files
'''

for piece in range(meta_df.shape[0]):
    midi_file = meta_df['midi'][piece]
    annotation = meta_df['annotation'][piece]
    score = ScoreAnnot(get_raw_segments(annotation))
    score_inst_list = score.get_inst_list()
    print(score_inst_list)
    
    if os.path.isfile(midi_file):
        
        # load MIDI file into pypianoroll object 
        multitrack = pypianoroll.read(midi_file)
        multitrack = multitrack.pad(multitrack.resolution*meta_df['n_end_beat'][piece]) 
        print( len(multitrack.tracks) )
        inst_list = [ track.name for track in multitrack.tracks ]
        print(inst_list)

        # change the name of the tracks
        for i in range(len(multitrack.tracks)):
            multitrack.tracks[i].name = score_inst_list[i]
            
        # save to npz file
        pypianoroll.save(meta_df['pianoroll'][piece], multitrack)
        print('file_saved: ', meta_df['pianoroll'][piece])
        
    else:
        print(f'ERROR: {midi_file} does not exist!')