# organize_annotations.py
'''
This file parse .orch files and save as .npy files
- .orch files is the annotation format provided in the orchestral texture dataset
- code in /src/ are used for parsing the annotation .orch files
'''

from src.parser import get_raw_segments
from src.classes import ScoreAnnot 
from src.annot_types import string_to_instrument
import pandas as pd
import numpy as np

meta_file = './dataset/new_metadata.csv'
df_meta = pd.read_csv(meta_file)

# create path for each piece to store pianoroll data
path = './dataset/annotations/'
for piece in range(df_meta.shape[0]):
    piece_name = df_meta['score'][piece].split('/')[-1].split('.')[0]
    df_meta['annot_npy'][piece] = path+piece_name+'.npy'
df_meta.to_csv(meta_file)

def make_label_array(score: ScoreAnnot)->np.array:
    n_inst = len(score.get_inst_list())
    inst_list = score.get_inst_list()
    segments = score.get_list_segments()

    end = 0
    arr = []
    #iterate through each segment
    for seg in segments:
        layer_list = seg.get_list_layers()
        role_dict = dict( (string_to_instrument(inst), None) for inst in inst_list ) # In hob103, there will be 2 VIOLIN1, but since their roles are identical, it will be fine
        #print( 'Before editting: role_dict = ', role_dict )

        # fill in the role of each instrumnet
        for layer in layer_list:
            role_one_hot = np.array( layer.get_role().get_role_three_bool() )
            for inst in layer.get_ensemble().get_list_components():
                role_dict[inst] = role_one_hot
        #print( 'After editting: role_dict = ', role_dict )

        # make the label numpy array
        a_measure = [] #shape = (n_inst, 3)
        for inst in inst_list:
            inst = string_to_instrument(inst)
            if role_dict[inst] is not None:
                a_measure.append( role_dict[inst] )
            else: # instrument not in any layers -> not making sound
                a_measure.append( np.zeros(3, dtype=bool) ) # MAYBE WE CAN HANDLE IT DIFFERENTLY, NOW IT IS IDENTICAL TO role='None'
        a_measure_arr = np.array(a_measure)

        # if there is a gap (empty bars)
        if seg.get_measure_beg() != end+1:
            #print(f'There is a gap at {end}')
            for i in range(end+1, seg.get_measure_beg()):
                arr.append( np.zeros((n_inst, 3),dtype=bool ) ) #append empty labels 
        for i in range( seg.get_measure_beg(), seg.get_measure_end()+1 ):
            arr.append( a_measure_arr )
        end = seg.get_measure_end()
        
    # after iterating through all segments
    piece_label = np.array( arr ) # shape = ( n_measures, n_inst, 3 )
    print(piece_label.shape)
    return piece_label

for piece in range(df_meta.shape[0]):
    file_path = df_meta['annotation'][piece]
    l = get_raw_segments(file_path)
    score = ScoreAnnot(l)
    piece_label = make_label_array(score)
    np.save( df_meta['annot_npy'][piece], piece_label )