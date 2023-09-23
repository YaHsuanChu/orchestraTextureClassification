#pianoRoll.py
'''extend the object from the library pypianoroll, storing and manipulation of a single piece'''
import numpy as np
import pypianoroll


class pianoRoll:
    def __init__(self, npz_file=None, n_bars_in_annotation=None, time_signature=None):
        if npz_file is not None:
            self.multitrack = pypianoroll.load(npz_file)
        self.n_tracks = len(self.multitrack.tracks)
        self.inst_list = [track.name for track in self.multitrack.tracks]
        
        self.resolution = self.multitrack.resolution
        self.active_length = self.multitrack.get_length()
        self.max_length = self.multitrack.get_max_length()
        self.n_bars_in_annotation = n_bars_in_annotation 
        self.time_signature= self.parse_time_signature(time_signature)
        self.check_theoretical_total_length()

    def parse_time_signature(self, string):
        '''parse time_signature information from metadata.csv and save to a list
        input format: 
            "beg_measure:end_measure=number_of_quarter_beats_per_bar, ..."
            example:
                "1:10=4,11:20=3" means:
                { measure 1 to 10: 4/4 or 2/2,
                  measure 11 20 20: 3/4 or 6/8 }
        return:
            a list of 3-tuple
            [ [begin of a segment, end of a segment, number of quarter beats per bar],
              [ ...                                               ],
              ...                                                   ]
            example:
                [ [1,10,4],
                  [11,20,3] ] a list shape (2, 3)'''

        time_signature_list = []
        raw_segs = string.split(',')
        for seg in raw_segs:
            n_quarter_beat = int(seg.split('=')[-1])
            interval = seg.split('=')[0]
            beg = int(interval.split(':')[0])
            end = int(interval.split(':')[1])
            #print('measure {:<4d} to {:<4d} with time signature {}'.format(beg, end, n_quarter_beat))
            time_signature_list.append( [beg, end, n_quarter_beat] )
        
        return time_signature_list
        
    def measure_to_time_step(self, measure):
        '''return the equivelant time step of the begining of a measure
        example.
            if measure = 1, return 0
            if measure = 10, and time signature = 4/4, return 9*4*resolution '''
        
        if measure > self.n_bars_in_annotation+1 or measure < 1:
            raise Exception('''Error: Measure number out of range !!!''')
        else:
            steps = 0
            for (beg, end, n_q_beat) in self.time_signature:
                if measure > end: # contain the whole segment
                    steps += (end-beg+1)*n_q_beat*self.resolution
                elif measure > beg: # break inside the segment
                    steps += (measure-beg)*n_q_beat*self.resolution
                else: # do not contain the segment
                    pass
            return steps

    def measure_interval_to_step_interval(self, beg_measure, end_measure):
        '''calculate the desired time steps range [beg_measure, end_measure]
        which is equivelant to [beg_time_step, end_time_step)
        return: 
            (beg_time_step, end_time_step)
        example.
            input: [beg_measure, end_measure] = [1, 10] and time_signature=4/4 for this segment
            return: (0, 40*resolution) '''
        
        return (self.measure_to_time_step(beg_measure), self.measure_to_time_step(end_measure+1))

    def get_trimmed_multitrack_by_measure(self, beg_measure, end_measure):
        '''the tracks of all instruments in range [beg_measure, end_measure]
        return an pypianoroll.Multitrack object with all tracks trimed'''
        
        beg, end = self.measure_interval_to_step_interval(beg_measure, end_measure)
        return self.multitrack.trim(start=beg, end=end)

    def get_trimmed_multitrack_by_time_step(self, beg_step, end_step):
        '''the tracks of all instrument in range [beg_step, end_step]
        return an pypianoroll.Multitrack object with all tracks trimmed'''

        if beg_step < 0 or beg_step >= self.max_length or end_step < 0 or end_step >= self.max_length:
            raise Exception('Error: time step index out of range!!!')
        elif end_step < beg_step:
            raise Exception('Error: beg_step > end_step')

        return self.multitrack.trim(start=beg_step, end=end_step)
            

    def get_pianoroll_by_inst_and_measure(self, inst_index, measure):
        if measure<1 or measure>self.n_bars_in_annotation:
            raise Exception('Error:  measure number out of range !!!')
        if inst_index<0 or inst_index>=self.n_tracks:
            raise Exception('Error: instrument index out of range !!!')
        beg_step, end_step = self.measure_interval_to_step_interval( measure, measure )

        return self.multitrack.tracks[inst_index].pianoroll[beg_step:end_step] 
    
    def get_pianoroll_blended_by_measure(self, measure):
        '''stack all the tracks and sum them up'''
        if measure<1 or measure>self.n_bars_in_annotation:
            raise Exception('Error:  measure number out of range !!!')
        beg_step, end_step = self.measure_interval_to_step_interval( measure, measure )
        
        stacked = []
        for track in self.multitrack.tracks:
            stacked.append( track.pianoroll[beg_step:end_step] )
        stacked = np.stack(stacked, axis=0).astype(np.float32)
        stacked = np.sum(stacked, axis=0)/(self.n_tracks)

        return stacked

    ###############################################################################
    ########## The code below is for manual data cleaning and debugging ###########
    ###############################################################################
    def check_theoretical_total_length(self):
        '''calculate the theiretical length of the pianoroll using time signature imformation
        and compare to the real length of a pianoroll by get_max_length()'''
        
        theoretical_time_steps = 0
        for seg in self.time_signature:
            beg = seg[0]
            end = seg[1]
            n_q_beat = seg[2]
            theoretical_time_steps += (end-beg+1)*n_q_beat*self.resolution
        #print('active length = {:<6d}, max length = {:<6d}, theoretical length = {:<6d}, difference = {:<4d}'.format(\
                #self.active_length, self.max_length, theoretical_time_steps, self.max_length-theoretical_time_steps))
        if self.max_length != theoretical_time_steps:
            raise Exception('Error: max length = {:<6d}, theoretical length = {:<6d}, difference = {:<4d}'.format(\
                                self.max_length, theoretical_time_steps, self.max_length-theoretical_time_steps))
    
    def check_piece_length_by_downbeat_count(self):
        '''see if downbeats number match the total number of bars in a piece'''

        count_annot = self.n_bars_in_annotation
        count_pianoroll = self.multitrack.count_downbeat()
        if count_annot != count_pianoroll:
            print('number of down beats: {:<5d}, number of bars in annotation: {:<5d}, do not match!!!'.format(count_pianoroll, count_annot))
        else:
            print('Consistent number of bars')



