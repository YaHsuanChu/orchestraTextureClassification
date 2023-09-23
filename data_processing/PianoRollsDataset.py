#PianoRollsDataset.py
'''prepare and load piano roll data'''

'''These are default settings, if you want to change different configuration parameters
please modify the variables in train.py'''
DEFAULT_TEST_DATA = [0, 4, 9]
DEFAULT_CONTEXT_MEASURES = 0
DEFAULT_OTHER_INST = False
DEFAULT_INPUT_CHANNEL = 1
DEFAULT_BLEND_MODE = 'SUM' #'SUM': sum all tracks => 2 channels, 'COMB': choose k-1 other tracks => k channels  
DEFAULT_K = 5

import torch
import pandas as pd
import cv2
import numpy as np
from .pianoRoll import pianoRoll
import random

class PianoRollsDataset(torch.utils.data.Dataset):
    def __init__(self, meta_csv_file, test=False, test_piece=DEFAULT_TEST_DATA, context=DEFAULT_CONTEXT_MEASURES,\
                other_inst=DEFAULT_OTHER_INST, blend=DEFAULT_BLEND_MODE, k=DEFAULT_K):
        ''' configuration '''
        self.meta_df = pd.read_csv(meta_csv_file)
        self.test_data = test_piece
        self.context_measures = context
        self.other_inst = other_inst
        self.blend_mode = blend
        self.k = k
        
        ''' define input channels '''
        if not other_inst: self.n_input_channels=1
        else:
            if self.blend_mode=='SUM': self.n_input_channels=2
            elif self.blend_mode=='COMB': self.n_input_channels=k
        
        self.empty = np.zeros( (self.n_input_channels, 96, 128), dtype=np.float32 ) #for padding
        
        ''' keep desired pieces only '''
        if test:                                            
            self.meta_df = self.meta_df.iloc[self.test_data]
            print('There are {} pieces in test set'.format(self.meta_df.shape[0]))
        else:
            self.meta_df = self.meta_df.drop(self.test_data, axis=0)
            print('There are {} pieces in train set'.format(self.meta_df.shape[0]))
        self.meta_df.reindex( range(self.meta_df.shape[0]), axis=0 )
        #print(self.meta_df.loc[:,['piece', 'n_measures', 'n_end_beat', 'time_signature']])

        ''' read into pianoRoll Objects '''
        self.pianoRoll_list = []
        for piece in range(self.meta_df.shape[0]):
            self.pianoRoll_list.append( pianoRoll(npz_file=self.meta_df.iloc[piece]['pianoroll'],\
                                                    n_bars_in_annotation=self.meta_df.iloc[piece]['n_measures'],\
                                                    time_signature=self.meta_df.iloc[piece]['time_signature']) )
        ''' read label arrays '''
        label_list = []
        for piece in range(self.meta_df.shape[0]):
            label_list.append( np.load(self.meta_df.iloc[piece]['annot_npy']) )

        ''' select eligible data, and store as id=[piece_index, measure, instrment_index] '''
        ''' for each piece:
                for each bar:
                    for each instrument:
                        if label is not [False, Fasle, False]
                        x.append()
                        label.append()
        '''       
        self.accessor = [] # id to access a data, not loading data here because of memory capacity limit
        self.label = []
        count_empty = 0
        for piece in range(self.meta_df.shape[0]):
            pianoroll = self.pianoRoll_list[piece]
            for measure in range(pianoroll.n_bars_in_annotation):
                for inst in range(pianoroll.n_tracks):
                    if np.any(label_list[piece][measure][inst]): #has at least one role
                        if np.any(pianoroll.get_pianoroll_by_inst_and_measure(inst, measure+1)): # if no notes played ,discard
                            self.accessor.append(np.array([piece, measure, inst]))
                            self.label.append(label_list[piece][measure][inst])
                        else: #an empty bar 
                            count_empty += 1
                                               
        
        self.accessor = np.array(self.accessor)
        self.label = torch.from_numpy(np.array(self.label)).float()
        #print( 'There are {} empty bars with labels in this dataset'.format(count_empty) )
        #print( 'self.accessor.shape = ', self.accessor.shape )
        #print( 'self.label.shape = ', self.label.shape )
        print( 'There are {} (inst, measure) data in this dataset'.format(len(self.label)) )
        #self.label_statistics(self.label.numpy())

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_id = self.accessor[index]
        piece, predicting_measure, inst = data_id[0], data_id[1], data_id[2] 
        pianoroll = self.pianoRoll_list[piece]
        x = []
        for getting_measure in range( predicting_measure-self.context_measures, predicting_measure+self.context_measures+1 ):
            if getting_measure>=0 and getting_measure<pianoroll.n_bars_in_annotation:
                target_inst = self.transform( pianoroll.get_pianoroll_by_inst_and_measure(inst, getting_measure+1) )
                if self.other_inst and self.blend_mode=='SUM':
                    all_inst = self.transform( pianoroll.get_pianoroll_blended_by_measure(getting_measure+1) )
                    stacked = np.stack( [target_inst, all_inst], axis=0 )
                    x.append(stacked)
                elif self.other_inst and self.blend_mode=='COMB':
                    n = pianoroll.n_tracks
                    pool = list(range(n))
                    pool.pop(inst)
                    chosen_k_minus_1 = random.sample(pool, self.k-1) #choose k-1 tracks from n-1 other tracks
                    other_tracks = [ self.transform( pianoroll.get_pianoroll_by_inst_and_measure(\
                                            other_inst_index, getting_measure+1) ) for other_inst_index in chosen_k_minus_1 ]
                    all_tracks = [target_inst]+other_tracks
                    stacked = np.stack(all_tracks)
                    x.append(stacked)
                else: # other_inst=False <=> input single track
                    x.append( np.expand_dims(target_inst, axis=0 ) )
            else: #pad zero arrays
                x.append(self.empty)

        if len(x)==1: #no context
            x = x[0]
        else:
            x = np.concatenate(x, axis=-2)
        #print(x.shape)
        return x, self.label[index]
    
    def transform(self, one_bar_pianoroll):
        '''resize and normalize 
        input:
            one_bar_pianoroll, shape = (48, 128) or (72, 128) or (96, 128)
        return:
            one_bar_pianoroll, shape = (96, 128), value between 0~1, dtype=np.float32
        '''
        one_bar_pianoroll = cv2.resize( one_bar_pianoroll, (128, 96), interpolation=cv2.INTER_AREA ) #NOTE.pass (n_cols, n_rows) as arg in cv2
        one_bar_pianoroll = one_bar_pianoroll.astype(np.float32)/127.0 # normalize
        return one_bar_pianoroll
    
    def label_statistics(self, label):
        '''label shohuld be of shape (n_data, 3)
        invetigate the distrubution of data'''
        
        print('Labels statistics:')
        n_data = label.shape[0]
        role_counts = np.sum(label, axis=0)
        total_labels = np.sum(role_counts)
        label_count_per_data = np.sum(label, axis=1)
        n_label_distribution = np.array([np.sum(label_count_per_data==n_label) for n_label in range(4)])
        
        n_mel_rhythm = np.sum( np.sum( label==[1, 1, 0], axis=1 )==3 )
        n_rhythm_harm = np.sum( np.sum( label==[0, 1, 1], axis=1 )==3 )

        print('number of [mel, rhythm, harm] labels = ', role_counts, ', : ', role_counts.astype(np.float32)/total_labels )
        print('data with [0, 1, 2, 3] labels = ', n_label_distribution, ', : ', n_label_distribution.astype(np.float32)/n_data)
        print('number of mel+rhythm = ', n_mel_rhythm)
        print('number of rhythm+hram = ', n_rhythm_harm)
