"""
Processing functions of ScoreAnnot
- Complex score -> score split bar-by-bar
- Complex score -> only BlendLayer

Error checking functions
"""
from .classes import *
from .parser import *
import argparse
import collections
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import time

def timer(func):
    def wrapper(*args, **kwargs):
        beg_time = time.time()
        val = func(*args, **kwargs)
        tot_time = time.time() - beg_time
        print(f"{func.__name__} : ", tot_time)
        return val
    return wrapper

def find_corresponding_segment(score, bar_beg):
    """Finds the corresponding segment of the score with the bar number

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot in which the segement is located
    bar_beg : int
        Beginning bar inside the segment we look for

    Returns
    -------
    Segment
        The corresponding segment if it exists, new Segment else
    """
    for segment in score.list_segments:
        if segment.measure_beg <= bar_beg <= segment.measure_end:
            return segment
    
    # Discard rest bars
    # raise IndexError(f"# Bar : {bar_beg}")

    # Accept rest bars
    # print(f"WARNING : bar {bar_beg} is missing")
    return Segment(measure_beg=bar_beg, measure_end=bar_beg, list_layers=[])



def update_list_components(list_components, segment_beg):
    """Returns a new list of instruments with the ensemble of AddedInstruments as SimpleInstruments if it actually plays

    Parameters
    ----------
    list_components : list
        List of Instruments with AddedInstruments
    segment_beg : int
        Beginning of the segment

    Returns
    -------
    list
        List of instruments without AddedInstruments
    """
    new_list_components = []
    for instr in list_components:
        if isinstance(instr, AddedInstruments):
            # If it is an AddedInstrument inside the bar limits : append it
            if instr.measure_beg <= segment_beg <= instr.measure_end: 
                # On part du principe que y a pas de AddedInstrument dans un AddedInstrument
                new_list_components += (instr.added_ensemble.list_components) 
            # If it is an AddedInstrument outside the bar limits : do nothing
            else: 
                continue
        else: # If it is a simpleInstrument : append it
            new_list_components.append(instr)
    return new_list_components


def remove_non_existing_instruments(segment):
    """Removes instruments that do not play (i.e. AddedInstruments that play later) in a segment

    Parameters
    ----------
    segment : Segment
        Segment in which we want to remove instruments
    """
    segment_beg = segment.measure_beg

    new_list_layers = []

    for layer in segment.get_list_layers():
        if isinstance(layer, BlendLayer):
            new_list_components = update_list_components(layer.ensemble.list_components, segment_beg)
            layer.ensemble.list_components = new_list_components

        elif isinstance(layer, QRLayer):
            for phrase in layer.list_phrases:
                for internal_layer in phrase.list_qr_internal_layers:
                    new_list_components = update_list_components(internal_layer.internal_ensemble.list_components, segment_beg)
                    internal_layer.internal_ensemble.list_components = new_list_components


        if len(new_list_components) == 0: # If the layer was only composed of an AddedInstrument
            continue
        else:
            new_list_layers.append(layer)
                
    segment.list_layers = new_list_layers


def split_to_bar_layers(score, qr_split=True):
    """Splits a score into segments of length 1 bar

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot object to split
    qr_split : bool
        Splits QRLayers into BlendLayers if True
    
    Returns
    -------
        ScoreAnnot object with only BlendLayers and Segments of length 1
    """
    # @timer
    def split_QR(score):
        dict_qrs = {
            "QR" : [],
            "bar_beg" : [],
            "bar_end" : []
        }
        list_already_done_QR = [] # small optimization : compare str instead of LayerObjects
        
        def add_to_dict_qrs(layer, begin_qr, end_qr):
            dict_qrs['QR'].append(layer)
            dict_qrs['bar_beg'].append(begin_qr)
            dict_qrs['bar_end'].append(end_qr)
            list_already_done_QR.append(make_qr_layer(layer))
        
        n_qrs = 0
        # Retrieve all QR layers and their boundaries
        for i_seg, segment in enumerate(score.get_list_segments()):
            for i_lay, layer in enumerate(segment.get_list_layers()):
                if isinstance(layer, QRLayer):
                    str_layer = make_qr_layer(layer)
                    begin_qr = segment.get_measure_beg()
                    for i_next, next_segment in enumerate(score.get_list_segments()[i_seg+1:]):
                        
                        if next_segment == score.get_list_segments()[-1]: # Si le dernier layer est un QR
                            # if layer not in dict_qrs['QR']:
                            if str_layer not in list_already_done_QR:
                                add_to_dict_qrs(layer, begin_qr, score.get_list_segments()[-1].get_measure_end())
                                n_qrs += 1
                        
                        if layer not in next_segment.get_list_layers(): # Autrement, on regarde le next layer s'il y est toujours
                            end_qr = next_segment.get_measure_end() - 1
                            # if layer not in dict_qrs['QR']:
                            if str_layer not in list_already_done_QR:
                                add_to_dict_qrs(layer, begin_qr, end_qr)
                                n_qrs += 1
                            else:
                                indexes_layer = [i for i, x in enumerate(dict_qrs['QR']) if x == layer]
                                end_already_added = [dict_qrs['bar_end'][i] for i in indexes_layer]
                                if end_qr not in end_already_added:
                                    add_to_dict_qrs(layer, begin_qr, end_qr)
                                    n_qrs += 1
                            break
                
        
        # Split QR in blend layers and store them in a list (size n_qrs) of list (size periodicity)
        for i in range(n_qrs):
            this_qr = dict_qrs['QR'][i]
            this_bar_beg = dict_qrs['bar_beg'][i]
            this_bar_end = dict_qrs['bar_end'][i]


            # print(this_bar_beg, this_bar_end, make_qr_layer(this_qr))
            list_list_blend_layers = []
            for phrase in this_qr.list_phrases:
                for internal_layer in phrase.get_list_qr_internal_layers():
                    list_blends_temporal = []
                    this_ensemble = internal_layer.get_internal_ensemble()
                    for function in internal_layer.get_internal_temporal_functions():
                        new_blend_layer = BlendLayer(measure_beg=this_bar_beg, measure_end=this_bar_end)
                        new_blend_layer.set_function(function)
                        new_blend_layer.set_ensemble(this_ensemble)

                        list_blends_temporal.append(new_blend_layer)
                    periodicity = len(list_blends_temporal)
                    list_list_blend_layers.append(list_blends_temporal)
            
            # for l in list_list_blend_layers:
            #     for bl in l:
            #         print(make_blend_layer(bl))
            #     print()

            # Append newly created blend layers to the existing score
            
            # Case n < 1
            if not(this_qr.get_length_phrase().is_integer()):
                # print(f"QR with non-integer length : [{this_bar_beg}-{this_bar_end}] {make_qr_layer(this_qr)}")
                periodicity_bars = 1            
                count_bars = 0
                count_phrase = 0
                for i, n_bar in enumerate(range(this_bar_beg, this_bar_end+1)):
                    segment_to_put = find_corresponding_segment(score, n_bar)
                    to_put = count_phrase % periodicity
                    count_phrase += 1
                    for list_blends in list_list_blend_layers:
                        blend_to_put = copy.deepcopy(list_blends[to_put]) # On n'est jamais assez surs...
                        if blend_to_put.get_relation() != Relation.REST:
                            segment_to_put.append_layer(blend_to_put)
                        else:
                            for i in range(periodicity): # On ajoute un des blend suivant
                                next_blend_to_put = list_blends[(to_put + i) % periodicity]
                                if next_blend_to_put.get_relation() != Relation.REST:
                                    segment_to_put.append_layer(next_blend_to_put)
                                    break
                    count_bars += 1
                    
                    # Remove QR layers
                    new_list_layers = []
                    for layer in segment_to_put.get_list_layers():
                        if not(isinstance(layer, QRLayer)):
                            new_list_layers.append(copy.deepcopy(layer))
                    segment_to_put.set_list_layers(new_list_layers)
                    # print(make_segment(segment_to_put))
            
            # Case n is integer
            else:
                periodicity_bars = this_qr.get_length_phrase()            
                count_bars = 0
                count_phrase = 0
                for i, n_bar in enumerate(range(this_bar_beg, this_bar_end+1)):
                    segment_to_put = find_corresponding_segment(score, n_bar)
                    if count_bars % periodicity_bars == 0:
                        to_put = count_phrase % periodicity
                        count_phrase += 1
                    for list_blends in list_list_blend_layers:
                        blend_to_put = copy.deepcopy(list_blends[to_put]) # On n'est jamais assez surs...
                        if blend_to_put.get_relation() != Relation.REST:
                            segment_to_put.append_layer(blend_to_put)
                    count_bars += 1
                    
                    # Remove QR layers
                    new_list_layers = []
                    for layer in segment_to_put.get_list_layers():
                        if not(isinstance(layer, QRLayer)):
                            new_list_layers.append(copy.deepcopy(layer))
                    segment_to_put.set_list_layers(new_list_layers)
                    # print(make_segment(segment_to_put))
                        
            
    
    beginning_bar = score.list_segments[0].measure_beg
    total_bars = score.list_segments[-1].measure_end
    new_score = ScoreAnnot()
    for bar_i in range(beginning_bar, total_bars+1):
        # Split into segments of length 1 bar
        new_segment = Segment(
            measure_beg = bar_i,
            measure_end = bar_i)
        corresonding_segment = find_corresponding_segment(score, bar_i)
        new_segment.list_layers = copy.deepcopy(corresonding_segment.get_list_layers())
        new_segment.label = copy.deepcopy(corresonding_segment.get_label())

        # Process segments to remove instruments that are actually not in the segment
        remove_non_existing_instruments(new_segment)

        # Append to the new score
        new_score.list_segments.append(new_segment)
    
    if qr_split:
        split_QR(new_score)

    return new_score




def merge_bar_layers(score):
    """ Merges identical segments of a score in which segments are 1-bar long

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot object with possible duplicate consecutive segments

    Returns
    -------
        ScoreAnnot object with only BlendLayers

    """

    new_score = ScoreAnnot()
    list_new_segments = []
    current_measure_beg = score.list_segments[0].measure_beg
    for seg_i in range(len(score.list_segments)-1):
        current_layers = score.list_segments[seg_i].get_list_layers()
        next_layers = score.list_segments[seg_i+1].get_list_layers()
        if (current_layers != next_layers): # check if the two segment are the same (apart from the bar limits)
            new_segment = copy.deepcopy(score.list_segments[seg_i])
            new_segment.set_measure_limits(current_measure_beg, score.list_segments[seg_i].measure_end)
            new_segment.set_label(score.list_segments[seg_i].label)
            list_new_segments.append(new_segment)
            current_measure_beg = score.list_segments[seg_i+1].measure_end
    # Add the last segment
    new_segment = copy.deepcopy(score.list_segments[seg_i+1])
    new_segment.set_measure_limits(current_measure_beg, score.list_segments[len(score.list_segments)-1].measure_end)
    list_new_segments.append(new_segment)

    new_score.list_segments = list_new_segments
    return new_score



def split_to_blend_layers(score, qr_split=True, DISPLAY=True):
    """ Splits a score into segments containing only BlendLayers

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot object with possible AddedInstruments
    qr_split : bool
        Splits QRLayers into BlendLayers if True
    DISPLAY : bool
        Displays the score if True

    Returns
    -------
        ScoreAnnot object with only BlendLayers
    """
    if DISPLAY:
        print("*" * 50 )
    new_score = split_to_bar_layers(score, qr_split)
    new_score_merged = merge_bar_layers(new_score)
    if DISPLAY:
        print("*" * 50 + "\n")
        print(score_annot_to_orchnot(score, with_label=True))
        print("*" * 50 + "\n")
        print(score_annot_to_orchnot(new_score, with_label=True))
        print("*" * 50 + "\n")
        print(score_annot_to_orchnot(new_score_merged, with_label=True))
    return new_score_merged




def make_simple_ensembles(score_blend):
    """
    Remove all the divisi within a same Layer
    """
    for i_segment, seg in enumerate(score_blend.list_segments):
        for i_layer, layer in enumerate(seg.list_layers):
            this_ensemble = layer.ensemble.list_components
            simple_ensemble = []
            for instr in this_ensemble:
                instr.set_divisi(-1)
                if instr not in simple_ensemble:
                    simple_ensemble.append(instr)
            layer.ensemble.list_components = simple_ensemble


def alter_score_annot(score_blend, action):
    if action == "add_cb":
        for i_segment, seg in enumerate(score_blend.list_segments):
            for i_layer, layer in enumerate(seg.list_layers):
                this_ensemble = layer.ensemble.list_components
                instr_names = [instr.name for instr in this_ensemble]
                if ("CELLO" in instr_names) and not("CONTREBASS" in instr_names):
                    layer.ensemble.list_components.append(
                        string_to_instrument("Cb")
                    )


def set_label_to_score(filename="", score=None):
    """Sets annotated labels to segments of the score

    Parameters
    ----------
    filename : str
        Name of the file
    score : ScoreAnnot
        ScoreAnnot to be labeled

    Returns
    -------
        ScoreAnnot with labels

    """
    
    def retrive_relevant_lines(raw_lines):
        """Gets code lines + labelled lines    
        """
        relevant_lines = []
        for l in raw_lines:
            if not(l.isspace()) and (l[0:2] != "# "):
                relevant_lines.append(repr(l)[1:-3])
        return relevant_lines

    def get_corresponding_label(segment, labels):
        """ Gets the corresponding label of a segment
        """
        seg_beg = segment.get_measure_beg()
        seg_end = segment.get_measure_end()
        for i, label in enumerate(list(labels.keys())[:-1]):
            label_beg = labels[label]
            label_end = list(labels.values())[i+1] - 1
            if (label_beg <= seg_beg) and (seg_end <= label_end):
                return label
        raise IndexError(f'No label found for {filename} at bar {seg_beg}')

    # Make score
    if filename != "" and score is None:
        score = to_score(filename)
    
    # Retrieve labels
    annot_file = open(filename, 'r')
    raw_lines = annot_file.readlines()
    annot_file.close()

    relevant_lines = retrive_relevant_lines(raw_lines)

    labels = dict()
    for i, l in enumerate(relevant_lines):
        if l[0:2] == "#!":
            label = l.split(" ")[0].split('!')[1].replace('\\','')
            next_line = relevant_lines[i+1]

            bar_limits = re.search(r'\[[0-9-]+\]', next_line).group()
            bar_limits_tuple = re.split('\[|\]|-', bar_limits)
            measure_beg = int(bar_limits_tuple[1])
            labels[label] = int(measure_beg)

    labels['End'] = score.get_length() + 1
    
    # Set label
    for segment in score.get_list_segments():
        corresponding_label = get_corresponding_label(segment, labels)
        segment.set_label(corresponding_label)

    return score



# ================================================================
# ======== ERRORS CHECKING =======================================
# ================================================================


def check_identifiers(raw_segments):
    """
    Checks that all identifiers are used and defined
    """
    n_idents_errors = 0
    for segment in raw_segments:
        if "<" not in segment:
            continue
        elements = re.search(r"<(.*)>(.*)", segment)
        bar_limits = re.search(r'\[[0-9-]+\]', segment).group()
        layers_ident = elements.group(1).replace('|','')
        layers_roles = elements.group(2).strip()

        layers_roles = re.sub('\{.*\}', '', layers_roles)
        layers_roles = re.sub('\(.*\)', '', layers_roles)
        
        set_idents_left = set([c for c in layers_ident if c.isalpha()])
        
        list_layers_roles = layers_roles.split(' ')
        list_idents_right = []
        for layer in list_layers_roles:
            if len(layer) > 0:
                if layer[0] == "~": list_idents_right.append(layer[1])
                else: list_idents_right.append(layer[0])
        set_idents_right = set(list_idents_right)
        
        if set_idents_left != set_idents_right:
            idents_not_assigned = set_idents_left - set_idents_right
            idents_not_used = set_idents_right - set_idents_left
            if len(idents_not_assigned) > 0:
                print(f"WARNING: {bar_limits} Identifier on the LEFT not right: {idents_not_assigned}")
                n_idents_errors += 1
            if len(idents_not_used) > 0:
                print(f"WARNING: {bar_limits} Identifier on the RIGHT not left: {idents_not_used}")
                n_idents_errors += 1

    return n_idents_errors




def check_syntax(filename):
    """
    Check the syntax of an .orchnot file to a score object

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    """
    raw_segments = get_raw_segments(filename)
    n_idents_errors = check_identifiers(raw_segments)
    score = ScoreAnnot(raw_segments, checking_syntax=True)
    if score.n_errors == 0:
        print("================================")

        print("==== IDENTIFIERS errors :", n_idents_errors, "====")
        print("================================")

        count_warning = {
            "tilde_errors" : 0, 
            "duplicate_instruments" : 0, 
            "added_instrument" : 0,
            "remaining_qr" : 0,
            }
        
        set_internal_bar_limits(score, checking_syntax=True, count_warning=count_warning)
        print("=== TILDE errors :", count_warning['tilde_errors'], "===")
        print("================================")




def check_added_instrument(score, count_warning):
    """
    Checks if AddedInstruments are at the right place
    (They begin and end inside a segment)
    
    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot to be checked
    count_warning : dict
        Dictionnary to store errors

    """
    def check_list_components(list_components, measure_beg, measure_end, segment):
        for instr in list_components:
            if isinstance(instr, AddedInstruments):
                if not((measure_beg <= instr.get_measure_beg()) and (instr.get_measure_beg() <= instr.get_measure_end()) and (instr.get_measure_end() <= measure_end)):
                    print(f"WARNING : AddedInstrument between [{instr.get_measure_beg()}-{instr.get_measure_end()}] in segment : [{measure_beg}-{measure_end}]")
                    print(make_segment(segment))
                    count_warning['added_instrument'] += 1

    for segment in score.get_list_segments():
        measure_beg = segment.get_measure_beg()
        measure_end = segment.get_measure_end()
        for layer in segment.get_list_layers():
            if isinstance(layer, BlendLayer):
                check_list_components(layer.get_ensemble().get_list_components(), measure_beg, measure_end, segment)
            if isinstance(layer, QRLayer):
                for phrase in layer.list_phrases:
                    for internal_layer in phrase.list_qr_internal_layers:
                        check_list_components(internal_layer.get_internal_ensemble().get_list_components(), measure_beg, measure_end, segment)


def check_composition_layers(score, count_warning):
    """
    Checks that instruments are not duplicate within a segment
    
    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot to be checked
    count_warning : dict
        Dictionnary to store errors

    """

    def get_all_instruments_playing_in_segment(segment):
        list_instruments_blend = []
        list_instrument_qr = []
        for layer in segment.get_list_layers():
            if isinstance(layer, BlendLayer):
                for instr in layer.get_ensemble().get_list_components():
                    list_instruments_blend.append(str(instr))
            if isinstance(layer, QRLayer):
                for phrase in layer.list_phrases:
                    for internal_layer in phrase.list_qr_internal_layers:
                        for instr in internal_layer.get_internal_ensemble().get_list_components():
                            list_instrument_qr.append(str(instr))
                    
        return list_instruments_blend, list_instrument_qr

    score_bar_split = split_to_bar_layers(score)
    for segment in score_bar_split.get_list_segments():
        list_instruments_blend, list_instrument_qr = get_all_instruments_playing_in_segment(segment)
        duplicate_instruments_blend = [item for item, count in collections.Counter(list_instruments_blend).items() if count > 1]
        if len(duplicate_instruments_blend) > 0:
            print(f"WARNING : {duplicate_instruments_blend} are duplicate in BlendLayers at bar {segment.get_measure_beg()}")
            count_warning['duplicate_instruments'] += 1

        duplicate_instruments_qr = [item for item, count in collections.Counter(list_instrument_qr).items() if count > 1]
        if len(duplicate_instruments_qr) > 0:
            print(f"INFO : {duplicate_instruments_qr} are duplicate in CRLayers at bar {segment.get_measure_beg()}")

        instr_in_qr_and_blend = []
        for instr in list_instruments_blend:
            if instr in list_instrument_qr:
                instr_in_qr_and_blend.append(instr)
        if len(instr_in_qr_and_blend) > 0:
            print(f"WARNING : {instr_in_qr_and_blend} are duplicate in BlendLayer/CRLayers at bar {segment.get_measure_beg()}")
            count_warning['duplicate_instruments'] += 1

    return score_bar_split


# ==============================================================
# ============== PARALLEL SYNTAX PARSING  ======================
# ==============================================================


def preprocess_raw_segments(list_lines):
    """Splits parallel bar indications into sequential bar indications

    Parameters
    ----------
    list_lines : list
        List of raw lines

    Returns
    -------
    list
        List of raw lines without parallel bar indications
    """
    new_list_lines = []
    for str_segment in list_lines:
        index_close_bracket = str_segment.index("]")
        bar_indications = str_segment[0:index_close_bracket+1]
        layers = str_segment[index_close_bracket+1:].replace('<QR>','')
        new_str_segment = f"{bar_indications}{layers}"
        if "," in bar_indications:
            list_indications = re.split('\[|\,|\]', bar_indications)[1:-1]
            for indication in list_indications:
                beg, end = beg_end(indication)
                new_str_segment = f"[{beg}-{end}]{layers}"
                new_list_lines.append(new_str_segment)
            continue

        new_list_lines.append(new_str_segment)

    return new_list_lines



def set_score_from_raw_segments(score, processed_raw_segments):
    """Put sequential lines into corresponding segments

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot in which segments are put
    processed_raw_segments : list
        List of sequential lines
    """
    for str_segment in processed_raw_segments:
        parallel_segment = Segment(str_segment)
        seg_beg = parallel_segment.get_measure_beg()
        seg_end = parallel_segment.get_measure_end()
        list_layers = parallel_segment.get_list_layers()
        for n_bar_parallel in range(seg_beg, seg_end+1):
            corresponding_segment = find_corresponding_segment(score, n_bar_parallel)
            corresponding_segment.concatenate_list_layers(list_layers)
        

def to_score_parallel(filename, check_syntax=False):
    """Makes a ScoreAnnot from an .orchnot in parallel syntax

    Parameters
    ----------
    filename : str
        Name of the file
    check_syntax : bool
        Displays errors if True

    Returns
    -------
    ScoreAnnot
        Corresponding ScoreAnnot
    """
    raw_segments = get_raw_segments(filename)
    processed_raw_segments = preprocess_raw_segments(raw_segments)
    # print_list(processed_raw_segments)

    # Get first and last bars
    bar_limits_end = re.search(r'\[[0-9-]+\]', processed_raw_segments[-1]).group()
    _, last_bar = beg_end(bar_limits_end[1:-1])
    bar_limits_begin = re.search(r'\[[0-9-]+\]', processed_raw_segments[0]).group()
    first_bar, _ = beg_end(bar_limits_begin[1:-1])
    
    # Makes a score of 1-long segments
    score = ScoreAnnot()
    list_segments = []
    for n_bar in range(first_bar, last_bar+1):
        list_segments.append(Segment(
            measure_beg=n_bar,
            measure_end=n_bar
        ))
    score.set_list_segments(list_segments)

    # Read file and put it at the corresponding segment
    set_score_from_raw_segments(score, processed_raw_segments)

    # print(score_annot_to_orchnot(score))

    # Remove useless instruments
    list_new_segments = []
    for segment in score.get_list_segments():
        new_segment = copy.deepcopy(segment)
        remove_non_existing_instruments(new_segment)
        list_new_segments.append(new_segment)

    score.set_list_segments(list_new_segments)


    if check_syntax:
        print("================================")
        count_warning = {
            "tilde_errors" : 0, 
            "duplicate_instruments" : 0, 
            "added_instrument" : 0
            }
        
        set_internal_bar_limits(score, checking_syntax=True, count_warning=count_warning)
        print("=== TILDE errors :", count_warning['tilde_errors'], "===")
        print("================================")

        check_added_instrument(score, count_warning)
        print("=== ADDED-INSTR errors :", count_warning['added_instrument'], "===")
        print("================================")

        check_composition_layers(score, count_warning)
        print("=== DUPLICATE errors :", count_warning['duplicate_instruments'], "===")
        print("================================")

        print()

    return score
    




# ================================================================
# ======== COMPARE TWO SCORES =======================================
# ================================================================


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


LIST_POSSIBLE_INSTRUMENTS = [k for k in DECYPHER_MAIN_INSTRUMENT]



def make_score_matrix_partial(THIS_POSSIBLE_ROLES):
    """Makes the score matrix with aggregated roles
    """
    score_matrix = np.eye(len(THIS_POSSIBLE_ROLES))

    def set_score(score, role1, role2):
        i_role1 = THIS_POSSIBLE_ROLES.index(role1)
        i_role2 = THIS_POSSIBLE_ROLES.index(role2)
        score_matrix[i_role1, i_role2] = score
        score_matrix[i_role2, i_role1] = score
        
    # set_score(0.5, 'MAIN_MEL', 'DECORATIVE_MEL')
    # set_score(0.5, 'RHYTHM', 'DECORATIVE_MEL')
    # set_score(0.5, 'RHYTHM', 'SPARSE')
    # set_score(0.5, 'HARM', 'SPARSE')

    # Différences acceptables: rhythm+mel et rhythm+harm avec l'un des rôles principaux
    if True:
        set_score(1.0, 'MAIN_MEL', 'DECORATIVE_MEL')
        set_score(1.0, 'RHYTHM', 'DECORATIVE_MEL')
        set_score(1.0, 'RHYTHM', 'SPARSE')
        set_score(1.0, 'HARM', 'SPARSE')

    # Vraies différences: quand on dit deux choses incompatibles
    if False:
        set_score(1.0, 'None', 'MAIN_MEL')
        set_score(1.0, 'None', 'RHYTHM')
        set_score(1.0, 'None', 'HARM')

    return score_matrix




def make_score_matrix_full(THIS_POSSIBLE_ROLES):
    """Makes the score matrix with full roles
    """
    score_matrix = np.eye(len(THIS_POSSIBLE_ROLES))

    def set_score(score, role1, role2):
        i_role1 = THIS_POSSIBLE_ROLES.index(role1)
        i_role2 = THIS_POSSIBLE_ROLES.index(role2)
        score_matrix[i_role1, i_role2] = score
        score_matrix[i_role2, i_role1] = score


    # ======= SMOOTH SCORE MATRIX =======
    # Within rhythm family
    for r in ["REPEAT_NOTE", "BATTERIE", "OSCILLATION", "ARPEGGIO", "SCALE"]:
        set_score(0.8, r, 'RHYTHM')

    for r in ["BATTERIE", "OSCILLATION"]:
        set_score(0.7, r, 'REPEAT_NOTE')
        
    set_score(0.7, 'ARPEGGIO', 'SCALE')
    set_score(0.7, 'RHYTHM', 'DECORATIVE_MEL')
    set_score(0.7, 'RHYTHM', 'SPARSE')
        
    # Mel
    set_score(0.7, 'MAIN_MEL', 'DECORATIVE_MEL')
    for r in ["RHYTHM", "REPEAT_NOTE", "BATTERIE", "OSCILLATION", "ARPEGGIO", "SCALE"]:
        set_score(0.5, r, 'MAIN_MEL')

    # None
    for r in THIS_POSSIBLE_ROLES:
        if r != 'None':
            set_score(0.3, r, 'None')
        
    # Repeat_note
    set_score(0.5, 'REPEAT_NOTE', 'HARM')
    set_score(0.3, 'REPEAT_NOTE', 'DECORATIVE_MEL')
    # Osc
    set_score(0.5, 'OSCILLATION', 'DECORATIVE_MEL')
    # Batt
    set_score(0.7, 'BATTERIE', 'OSCILLATION')
    set_score(0.5, 'BATTERIE', 'ARPEGGIO')
    set_score(0.3, 'BATTERIE', 'HARM')
    set_score(0.7, 'BATTERIE', 'DECORATIVE_MEL')
    # Arp
    set_score(0.5, 'ARPEGGIO', 'OSCILLATION')
    set_score(0.7, 'ARPEGGIO', 'DECORATIVE_MEL')
    # Scale
    set_score(0.5, 'SCALE', 'DECORATIVE_MEL')
    # Harm
    set_score(0.3, 'HARM', 'DECORATIVE_MEL')
    set_score(0.3, 'HARM', 'SPARSE')
    # Sparse
    set_score(0.3, 'SPARSE', 'DECORATIVE_MEL')

    return score_matrix






def get_score(score_matrix, role1, role2, THIS_POSSIBLE_ROLES):
    """Gets the score from the score matrix
    """
    i_role1 = THIS_POSSIBLE_ROLES.index(role1)
    i_role2 = THIS_POSSIBLE_ROLES.index(role2)
    return score_matrix[i_role1, i_role2]

def compare_segment_role(segment1, segment2, SCORE_MATRIX, aggr_roles, THIS_POSSIBLE_ROLES, dict_role_to_score):
    """Compares two segments

    Parameters
    ----------
    segment1 : Segment
        Segment from the first annotator
    segment2 : Segment
        Segment from the second annotator
    SCORE_MATRIX : nupy.array
        Score matrix
    aggr_roles : bool
        Roles are aggregated or not
    THIS_POSSIBLE_ROLES : list
        List of roles
    dict_role_to_score : dict
        Dictionnary with roles as keys and list of score as values

    Returns
    -------
    float
        Similarity between the two annotated segments
    dict
        Score between each instrument
    dict
        If the instrument is present in one of the annotation
    """
    instr_to_role1 = dict((k, "") for k in LIST_POSSIBLE_INSTRUMENTS)
    instr_to_role2 = dict((k, "") for k in LIST_POSSIBLE_INSTRUMENTS)

    def update_instr_to_role(segment, instr_to_role):
        for layer in segment.get_list_layers():
            if isinstance(layer, QRLayer):
                raise TypeError("QR are not compatible yet")
            elif isinstance(layer, BlendLayer):
                real_role = layer.get_role().get_only_role()

                if aggr_roles:
                    if real_role in ["REPEAT_NOTE", "BATTERIE", "OSCILLATION", "ARPEGGIO", "SCALE"]:
                        this_role = "RHYTHM"
                    else:
                        this_role = real_role
                else:
                    this_role = real_role
                
                for instrument in layer.get_ensemble().get_list_components():
                    instr_to_role[CYPHER_INSTRUMENT[str(instrument.get_only_instr())]] = this_role

    
    update_instr_to_role(segment1, instr_to_role1)
    update_instr_to_role(segment2, instr_to_role2)

    score = 0
    max_score = 0
    score_dict = dict((k, 0) for k in LIST_POSSIBLE_INSTRUMENTS)
    is_present_dict = dict((k, 0) for k in LIST_POSSIBLE_INSTRUMENTS)
    instruments_ok = 0

    for instrument in LIST_POSSIBLE_INSTRUMENTS:
        role1 = instr_to_role1[instrument]
        role2 = instr_to_role2[instrument]
        this_score = get_score(SCORE_MATRIX, role1, role2, THIS_POSSIBLE_ROLES)
         # Checker qu'on a bien annoté TOUS LES DEUX que les instruments jouaient : and
         # Checker QU'AU MOINS L'UN DES DEUX a annoté que les instruments jouaient : or
        if (role1 != "") or (role2 != ""):
            score_dict[instrument] = this_score
            is_present_dict[instrument] = 1
            score += this_score
            max_score += 1
            if this_score < 1.0:
                print('  ' + bcolors.FAIL + '%-4s' % instrument + bcolors.ENDC + '%14s / %-14s  %.2f' % (role1, role2, this_score))
            else:
                instruments_ok += 1

        dict_role_to_score[role1].append(this_score)
        dict_role_to_score[role2].append(this_score)

    if instruments_ok:
        print ('%d instruments: 1.0' % instruments_ok)


    return score/max_score, score_dict, is_present_dict



def update_association_matrix(list_components, association_matrix):
    for instrument1 in list_components:
        for instrument2 in list_components:
            index_instr1 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[str(instrument1.get_only_instr())])
            index_instr2 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[str(instrument2.get_only_instr())])
            association_matrix[index_instr1, index_instr2] = 1
            association_matrix[index_instr2, index_instr1] = 1

def make_association_matrix(segment):
    association_matrix = np.zeros((len(LIST_POSSIBLE_INSTRUMENTS), len(LIST_POSSIBLE_INSTRUMENTS)))
    for layer in segment.get_list_layers():
        if isinstance(layer, QRLayer):
            raise TypeError("QR are not compatible yet")
        elif isinstance(layer, BlendLayer):
            update_association_matrix(layer.get_ensemble().get_list_components(), association_matrix)
    return association_matrix


def compute_distance_association(segment1, segment2):
    """Compares the distribution of instruments among layers :
    Makes two association matrices (binary matrices, where M[i,j] = 1 if i,j in same layer)
    And computes L2 norm

    Parameters
    ----------
    segment1 : Segment
        Segment from the first annotator
    segment2 : Segment
        Segment from the second annotator

    Returns
    -------
    float
        Distance between the two association matrixes
    """

    association_matrix1 = make_association_matrix(segment1)
    association_matrix2 = make_association_matrix(segment2)

    # Simple distance
    dist_assoc_matrix = np.linalg.norm(association_matrix1 - association_matrix2)

    return dist_assoc_matrix


def compute_n_instruments_agree(segment1, segment2):
    
    association_matrix1 = make_association_matrix(segment1)
    association_matrix2 = make_association_matrix(segment2)

    n_agree = 0
    n_not_off = 0
    for i, (instr1, instr2) in enumerate(zip(association_matrix1, association_matrix2)):
        if (instr1.sum() != 0) and (instr2.sum() != 0): # both annotators didnt annotate it at OFF
            n_agree += np.array_equal(instr1, instr2)
            n_not_off += 1

    return n_agree / n_not_off





def color_score(score):
    """Beautify display of score
    """
    color_display = bcolors.OKGREEN if score > 0.8 else bcolors.OKBLUE if score > 0.5 else bcolors.FAIL
    return color_display + "%.2f" % score + bcolors.ENDC


def color_distance(distance):
    color_display = bcolors.OKGREEN if distance == 0 else bcolors.OKBLUE if distance < 4  else bcolors.FAIL
    return color_display + "%.2f" % distance + bcolors.ENDC



def compare_scores(score1, score2, aggr_roles):
    """Compares two scores at the bar-level

    Parameters
    ----------
    score1 : ScoreAnnot
        Score from the first annotator
    score2 : ScoreAnnot
        Score from the second annotator
    aggr_roles : bool
        If the roles are aggregated or not
    """
    first_bar1 = score1.get_list_segments()[0].get_measure_beg()
    first_bar2 = score2.get_list_segments()[0].get_measure_beg()
    last_bar1 = score1.get_list_segments()[-1].get_measure_end()
    last_bar2 = score2.get_list_segments()[-1].get_measure_end()
    if (first_bar1 != first_bar2) or (last_bar1 != last_bar2):
        print(f"Annot1 : {first_bar1}-{last_bar1} ; Annot2 : {first_bar2}-{last_bar2}")
        raise ValueError("First and last bars are not equal")

    if aggr_roles:
        THIS_POSSIBLE_ROLES = ["MAIN_MEL", "RHYTHM", "HARM", "DECORATIVE_MEL", "SPARSE", "None", "QR", ""]
        SCORE_MATRIX = make_score_matrix_partial(THIS_POSSIBLE_ROLES)
    else:
        THIS_POSSIBLE_ROLES = [k for k in CYPHER_ROLE] + ["QR"] + [""]
        SCORE_MATRIX = make_score_matrix_full(THIS_POSSIBLE_ROLES)

    # SCORE_MATRIX = np.eye(len(THIS_POSSIBLE_ROLES))
    
    tot_bars = 0
    tot_score = 0
    list_distance_association = []
    list_ratio_instr_association_agree = []
    list_score_dict = []
    list_is_present_dict = []
    dict_role_to_score = dict((k, []) for k in THIS_POSSIBLE_ROLES)

    for n_bar, (segment1, segment2) in enumerate(zip(score1.get_list_segments(), score2.get_list_segments())):
        print("Annot1 : " + make_segment(segment1))
        print("Annot2 : " + make_segment(segment2))
        score_similarity, score_dict, is_present_dict = compare_segment_role(segment1, segment2, SCORE_MATRIX, aggr_roles, THIS_POSSIBLE_ROLES, dict_role_to_score)
        tot_score += score_similarity
        list_score_dict.append(score_dict)
        list_is_present_dict.append(is_present_dict)
        print("Score role similarity : " + color_score(score_similarity))

        ## Exact same partition
        distance_association = compute_distance_association(segment1, segment2)
        list_distance_association.append(distance_association)
        print("Distance association matrix : " + color_distance(distance_association))

        ## Instruments in same layer
        ratio_instr_assoc_agree = compute_n_instruments_agree(segment1, segment2)
        list_ratio_instr_association_agree.append(ratio_instr_assoc_agree)
        print("Ratio agreement combination instruments : " + color_score(ratio_instr_assoc_agree))

        print()
        tot_bars += 1
    

    list_distance_association = np.array(list_distance_association)

    print("----------------")
    print("TOTAL SCORE :", color_score(tot_score/tot_bars))
    print("EXACT SAME PARTITION :", color_score(np.mean(list_distance_association == 0)))
    print("INSTRUMENTS SAME LAYER :", color_score(np.mean(list_ratio_instr_association_agree)))

    # Compute average score by instrument
    print("----------------")
    for instr in LIST_POSSIBLE_INSTRUMENTS:
        ocurrences = np.sum([k[instr] for k in list_is_present_dict])
        sum_score = np.sum([k[instr] for k in list_score_dict])
        average_score = (sum_score / ocurrences)
        print(color_score(average_score), f"{instr} ({ocurrences})")

    # Compute average score by role : ca a un sens ?
    print("----------------")
    for k, v in dict_role_to_score.items():
        if k == "":
            k = "Off"
        if len(v) > 0:
            print(color_score(np.nanmean(v)) + " " + k)
        else:
            print("xxxx", k)

    plt.hist(list_distance_association)
    plt.title('Histogram Distance between association matrices')
    plt.show()



def compare_orchnots(filename1, filename2, aggr_roles=True):
    """Compares two sequential syntaxed .orchnot

    Parameters
    ----------
    filename1 : str
        Name of the .orchnot file from the first annotator
    filename2 : str
        Name of the .orchnot file from the second annotator
    aggr_roles : bool
        If the roles are aggregated or not
    """
    score1 = set_label_to_score(filename=filename1)
    score2 = set_label_to_score(filename=filename2)
    score_split1 = split_to_bar_layers(score1, qr_split=True)
    score_split2 = split_to_bar_layers(score2, qr_split=True)
    compare_scores(score_split1, score_split2, aggr_roles)



def compare_diff_syntax(filename_seq, filename_par, aggr_roles=True):
    """Compares a sequential syntaxed and a parallel syntaxed .orchnot files

    Parameters
    ----------
    filename_seq : str
        Name of the sequential syntaxed .orchnot
    filename_par : str
        Name of the parallel syntaxed .orchnot
    aggr_roles : bool
        If the roles are aggregated or not
    """
    score1 = set_label_to_score(filename=filename_seq)
    score_split1 = split_to_bar_layers(score1, qr_split=True)
    score_split2 = to_score_parallel(filename=filename_par)
    compare_scores(score_split1, score_split2, aggr_roles)



def test():
    score1 = set_label_to_score(filename="data/annotations/mozart/mozart-symph35-k385-mvt1.orch")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=False, action='store_true')
    parser.add_argument("--splitBlend", required=False)
    parser.add_argument("--checkSyntax", required=False)
    parser.add_argument("--compare", required=False, nargs='+')
    parser.add_argument("--compareSeqPar", required=False, nargs='+')

    args = parser.parse_args()

    if args.splitBlend is not None:
        filename = args.splitBlend
        score_annot = to_score(filename)
        split_to_blend_layers(score_annot)
    elif args.checkSyntax is not None:
        filename = args.checkSyntax
        check_syntax(filename)
    elif args.compare is not None:
        compare_orchnots(args.compare[0], args.compare[1])
    elif args.compareSeqPar is not None:
        compare_diff_syntax(args.compareSeqPar[0], args.compareSeqPar[1])
    elif args.test is not None:
        test()

    