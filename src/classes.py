"""
Declaration of classes
- ScoreAnnot
- Segment
- Layer
- QRLayer(Layer)
- Phrase
- QRInternalLayer
- BlendLayer(Layer)
- Ensemble
- AddedInstruments
- Function
"""

from .annot_types import *
import re
import json
import sys
import numpy as np
from .metalayers import *

INDENT_LENGTH = 2

def print_list(l):
    for i in l:
        print(i)


def beg_end(limits):
    '''Parse bar limits'''

    try:
        m = re.match(r'^\d+-\d+$', limits)
        if m:
            bar_limits = m.group()
            bar_limits_tuple = re.split('-', bar_limits)
            return int(bar_limits_tuple[0]), int(bar_limits_tuple[1])

        m = re.match(r'^\d+$', limits)
        if m:
            one = int(m.group())
            return one, one

        raise SyntaxError()

    except Exception as e:
        raise SyntaxError("No suitable bar limit '%s' " % limits + str(e))





class ObjectEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) in [Instrument, Role, Relation]:
            return str(o)
        elif type(o) in [Function, Ensemble]:
            return o.__dict__
        else:
            return {str(type(o).__name__) : o.__dict__}

def read_inst_list(line):
    """The functions takes a line from the .orch file containing the instruments and returns a python list with the instruments

    Args:
        line (string): a string starting wit "InstList:"

    Returns:
        list: a list of strings with instrument names
    """
    inst_list = []
    families = line.replace("InstList:",'').strip().split('|')
    for fam in families:
        inst_list.extend(fam.split(':')[1].split('.'))

    return inst_list



class ScoreAnnot:
    def __init__(self, list_lines=None, checking_syntax=False):
        self.list_segments = []
        self.list_metalayers = []
        self.inst_list = None
        previous_metalayers_list = []

        if list_lines is not None:
            if not(checking_syntax):
                for l in list_lines:
                    try:
                        if l[:8] == 'InstList': # If the current line is a list of instrument (new syntax)
                            self.inst_list = read_inst_list(l)
                        else: # The line is not a list of instrument, it can be in either syntaxes
                            self.list_segments.append(Segment(l,inst_list=self.inst_list))
                    except Exception as e:
                        raise SyntaxError(f"*{repr(l)}*")
            
            else:
                print("Checking syntax...")
                n_ok = 0
                n_errors = 0
                for l in list_lines:
                    try:
                        if l[:8] == 'InstList': # If the current line is a list of instrument (new syntax)
                            inst_list = read_inst_list(l)
                        else: # The line is not a list of instrument, it can be in either syntaxes
                            self.list_segments.append(Segment(l,inst_list=self.inst_list))
                        n_ok += 1
                    except Exception as e:
                        self.list_segments.append({})
                        n_errors += 1
                        print(f"*{repr(l)}* \n\t {e}")
                print(f"=== {n_errors} lines with syntax errors, {n_ok} lines ok ===")
                self.n_errors = n_errors
            for l in list_lines: # Check the metalayers after we have all segments
                if "{" in l: # If the line contains a specifier (there is a CR or TE)
                    current_metalayers_list = extract_metalayers(l,inst_list = self.inst_list,list_segments = self.list_segments)
                    metalayers_list_to_be_added = current_metalayers_list
                    for ml in metalayers_list_to_be_added: # Check that the current metalayers are not already in the list from previous bars
                        if ml in previous_metalayers_list:
                            metalayers_list_to_be_added.remove(ml)
                    self.list_metalayers.extend(metalayers_list_to_be_added)
                    previous_metalayers_list = current_metalayers_list

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def get_inst_list(self):
        return self.inst_list

    def get_list_segments(self):
        return self.list_segments

    def get_list_metalayers(self):
        return self.list_metalayers

    def set_list_segments(self, list_segments):
        self.list_segments = list_segments

    def get_length(self):
        return self.list_segments[-1].get_measure_end()

    def get_n_annotated_layers(self):
        n_layers = 0
        for seg in self.get_list_segments():
            n_layers += len(seg.get_list_layers())
        return n_layers

    def get_n_annotated_layers_by_bar(self):
        n_layers = 0
        for seg in self.get_list_segments():
            n_layers += seg.get_n_annotated_layers_by_bar()
        return n_layers


# ===================================================================
# =================== CONVERSTION NEW > OLD SYNTAX ==================
# ===================================================================  



def convert_single_layer_new_to_old(layer_new,inst_list,layers_codes):
    """Converts a layer in the new syntax to the "old" syntax
    """

    [letter, label] = layer_new.split(':')
    layer_inst_list = []
    # Find all occurencies of the letter and add corresponding instrument to the string
    start = 0
    indx = 0
    # Deal with divisi case
    div_indx = find_divisi(layers_codes) # Listo of indeces to store the position of the divisi
    while start < len(layers_codes) and indx != -1:
        indx = layers_codes.find(letter, start)
        if indx in range(len(layers_codes)):
            if div_indx == []:
                layer_inst_list.append(inst_list[indx])
            else:
                (section_index,subsection_code) = divisi_handler(div_indx,indx)
                layer_inst_list.append(inst_list[section_index] + subsection_code)
            
            start = indx
        start = start + 1
    
    layer_new = '({' + letter + '}' + label + ':' + '.'.join(layer_inst_list) + ')'

    return layer_new

def find_divisi(layers_codes):
    div_ind = []
    start = 0
    indx = 0
    while start < len(layers_codes) and indx != -1:
        indx = layers_codes.find('(',start)
        if indx in range(len(layers_codes)):
            div_ind.append(indx)
            start = indx
        start = start + 1

    return div_ind

# inst_list = ['Ob','Fg','Hrn','Trp','Timp','Vln1','Vln2','Vla','Vc']
# str_layers_new = '<(b0)i|h0|0|aaii> b:mel h:harm-u ~i:sparse-u ~a:mel-u (CR 0.25:(ai)b)'

def divisi_handler(div_indx,indx): # To correct the index for the instrument list
    a = np.array([x < indx for x in div_indx])
    b = np.array([x + 3 > indx for x in div_indx])
    if not any(a & b):
        section_index = indx - 3*sum(a)
        subsection_code = ''
    elif len(div_indx) == 1:
        section_index = div_indx[0]
        if indx == div_indx[0] + 1:
            subsection_code = '1'
        else:
            subsection_code = '2'
    else:
        section_index = div_indx[:sum(a)][-1] - 3*(sum(a)-1)
        if indx == div_indx[:sum(a)][-1] + 1:
            subsection_code = '1'
        else:
            subsection_code = '2'
    return (section_index,subsection_code)


def layers_str_conv_new_to_old(str_layers_new,inst_list):
    """
    This function converts a string containing the layers in the new format to the old format.
    In practice, <...|..> .:. (.) ==> (.) / (.)

    Args:
        str_layers_new (string): layers representation in the new format
        inst_list (list): list of the instruments
    """

    layers_codes_str = str_layers_new.split()[0] # String containing the layers letter codes, <aad|dd>
    layers_list_old = str_layers_new.split()[1:] # List of strings containing the descriptions of the layers
    layers_list_new = []

    # Clean the layers codes (drop <,>,| to btain a strings with only letters and 0s)
    layers_codes_str = layers_codes_str.replace('<','').replace('>','').replace('|','')

    for l in layers_list_old:
        if l[0] == '(' or l[0] == '{': # stop the conversion when we arrive at the higher level relations
            break
        elif l[0] == '~':
            layers_list_new.append('~' + convert_single_layer_new_to_old(l[1:], inst_list, layers_codes_str))
        else:
            layers_list_new.append(convert_single_layer_new_to_old(l, inst_list, layers_codes_str))
    
    layers_str_new = ' / '.join(layers_list_new)
    
    return layers_str_new


# ===================================================================  



class Segment:
    """
    Temporal segment of the score
    In practice, stuff like "[beg-end] (.) / (.)" 
    Adapted also for stuff like "[beg-end] <...|.|..> .:. (.)"

    Attributes
    ----------
    measure_beg : int
        Beginning bar number
    measure_end : int
        Ending bar number
    list_layers : [Layer]
        List of layers
    """
    def __init__(self, str_segment=None, measure_beg=None, measure_end=None, list_layers=None, label="", inst_list=None):

        self.label = label
        self.measure_beg = 0
        self.measure_end = 0
        self.list_layers = []
        
        if measure_beg is not None:
            self.measure_beg = measure_beg
        if measure_end is not None:
            self.measure_end = measure_end
        
        if str_segment is not None:

            if '<QR>' in str_segment:
                str_segment = str_segment.replace('<QR>', '')
                sys.stderr.write('Ignored <QR>\n')


            # Retrieve bar limits
            bar_limits_beg_end = re.search(r'\[[0-9-]+\]', str_segment)

            if bar_limits_beg_end:
                bar_limits = bar_limits_beg_end.group()
                self.measure_beg, self.measure_end = beg_end(bar_limits[1:-1])
            else:
                raise SyntaxError("No suitable bar limit")

            # Get layers
            str_all_layers = str_segment.replace(bar_limits + " ", '').strip() # remove bars indication

            # If the layers are expressed through the new syntax, convert the string to the old syntax
            # if the case call layers_str_conv_new_to_old(str_layers_new,inst_list)
            if str_all_layers[0] == '<':
                str_all_layers = layers_str_conv_new_to_old(str_all_layers, inst_list)


            # Split and convert all layers
            str_layers = re.split(r'\s\/\s', str_all_layers)
            self.list_layers = []
            for str_lay in str_layers:
                if "CR" in str_lay:
                    current_layer = QRLayer(str_lay, self.measure_beg, self.measure_end)
                else:
                    current_layer = BlendLayer(str_blend=str_lay, measure_beg=self.measure_beg, measure_end=self.measure_end)
                self.list_layers.append(current_layer)

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def get_list_layers(self):
        return self.list_layers

    def append_layer(self, layer):
        self.list_layers.append(layer)

    def concatenate_list_layers(self, list_layers):
        self.list_layers += list_layers

    def set_list_layers(self, list_layers):
        self.list_layers = list_layers

    def get_n_annotated_layers_by_bar(self):
        return len(self.list_layers) * self.get_length()
    
    def get_length(self):
        return self.measure_end - self.measure_beg + 1

    def get_measure_beg(self):
        return self.measure_beg

    def get_measure_end(self):
        return self.measure_end

    def get_measure_belong(self, measure):
        measure = int(measure)
        return (measure >= self.measure_beg and measure <= self.measure_end)

    def set_measure_limits(self, measure_beg, measure_end):
        self.measure_beg = measure_beg
        self.measure_end = measure_end

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label




class Layer:
    """
    One layer inside a segment
    In practice, stuff like "(.)"

    Attributes
    ----------
    internal_bar_beg : int
        Real beginning of the layer (due to ~)
    internal_bar_end : int
        Real ending of the layer (due to ~)
    """
    def __init__(self, internal_bar_beg, internal_bar_end, identifier):
        self.internal_bar_beg = int(internal_bar_beg) # For ~
        self.internal_bar_end = int(internal_bar_end) # For ~
        self.same_function = False
        self.identifier = identifier

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def get_internal_bar_beg(self):
        return self.internal_bar_beg
    
    def get_internal_bar_end(self):
        return self.internal_bar_end

    def get_length(self):
        return self.internal_bar_end - self.internal_bar_beg + 1

    def get_identifier(self):
        return self.identifier






class BlendLayer(Layer):
    """
    A normal blending layer
    In practice, stuff like (.:.)

    Attributes
    ----------
    function : Function
        Function of the layer
    same_function : bool
        Has a tilde at the beginning
    ensemble : Ensemble
        Instruments involved in the layer
    """
    def __init__(self, str_blend=None, measure_beg=None, measure_end=None, identifier=None):
        super().__init__(measure_beg, measure_end, identifier)
        self.function = None
        self.ensemble = None

        if str_blend is not None:
            str_blend_split = re.split("\(|:|\)", str_blend)
            self.same_function = (str_blend_split[0] == "~")
            str_function = str_blend_split[1]
            str_ensemble = str_blend_split[2]

            if re.match(r"\{[a-z]\}.*", str_function):
                regex_identifier = re.search(r"\{([a-z])\}(.*)", str_function)
                self.identifier = regex_identifier.group(1)
                str_function = regex_identifier.group(2)

            self.function = Function(str_function)
            self.ensemble = Ensemble(str_ensemble)

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        return super().__eq__(value)

    def __hash__(self):
        return hash(str(self))

    def get_role(self):
        return self.function.get_role()

    def get_relation(self):
        return self.function.get_relation()

    def get_ensemble(self):
        return self.ensemble

    def set_ensemble(self, ensemble):
        self.ensemble = ensemble
    
    def set_function(self, function):
        self.function = function



class Ensemble:
    """
    An musical instruments ensemble that can also include additional instruments

    Attributes
    ----------
    list_components : [Instrument|AddedInstruments]
        List of instruments involved in the ensemble + additional instruments if needed
    """
    def __init__(self, str_ensemble):
        # print(f"*{str_ensemble}*")
        self.list_components = []
        current_component = ""
        is_an_addition = False
        for i, c in enumerate(str_ensemble):
            current_component += c
            if c == "+":
                is_an_addition = True
            elif not is_an_addition:
                if c == "." and current_component != ".":
                    self.list_components.append(string_to_instrument(current_component[:-1]))
                    current_component = ""
                elif current_component == ".":
                    current_component = ""
            elif is_an_addition:
                if c == "}":
                    self.list_components.append(AddedInstruments(current_component))
                    current_component = ""
                    is_an_addition = False
            
            if i == len(str_ensemble)-1 and current_component != "":
                self.list_components.append(string_to_instrument(current_component))  

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)
    
    def get_list_components(self):
        return self.list_components

    def set_list_components(self, list_components):
        self.list_components = list_components






class Function:
    """
    Function of the layer inside the segment
    In practice, stuff like ".-." or "-."

    Attributes
    ----------
    role : Role (optional or set to None)
        Role of the layer 
    relation : Relation
        Relation between the instruments inside the ensemble
    """
    def __init__(self, str_function):

        split_str_function = re.split("-", str_function)
        str_role = split_str_function[0]
        str_relation = split_str_function[1] if len(split_str_function) > 1 else ''

        self.role = string_to_role(str_role)
        self.relation = string_to_relation(str_relation)

        assert self.relation is not None, "Relation is None"

    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def get_role(self):
        return self.role

    def get_relation(self):
        return self.relation






def backtrack_tilde_with_identifier(lay, seg, score, i_segment, checking_syntax, count_warning):
    """Clean backtrack of ~

    Parameters
    ----------
    lay : Layer
        Current layer observed
    seg : Segment
        Segment in which the layer is located
    score : ScoreAnnot
        ScoreAnnot in which the segment is located...
    i_segment : int
        Index of the segment within the score
    checking_syntax : bool
        Display verbose if true
    count_warning : dict
        Dictionary gathering errors
    """

    if lay.same_function: # if there is a tilde
        i_previous_segment = i_segment-1
        is_same_function = True
        real_end = lay.internal_bar_end # we store the end of THIS segment, to update all ones we visit
        n_loops = 0

        while (is_same_function) and (i_segment > 0): # while we are in a ~-sequence 
            previous_segment = score.list_segments[i_previous_segment] 
            previous_layer_found = False

            for previous_layer in previous_segment.list_layers:
                if previous_layer.identifier == lay.identifier:
                    # we expand the VISITED layer to the right (its end becomes the end of our INITIAL layer) 
                    previous_layer.internal_bar_end = real_end 
                    # we expand the INITIAL layer to the left (its beginning becomes the end of the VISITED layer)
                    lay.internal_bar_beg = previous_layer.internal_bar_beg 
                    # are we still backtracking ?
                    is_same_function = previous_layer.same_function
                    i_previous_segment -= 1
                    previous_layer_found = True
                    # we have found the previous layer corresponding to the same identifier, we can continue backtracking
                    break 
            
            if checking_syntax and not(previous_layer_found):
                print(f"WARNING : Previous ~ not found for a layer (id: {lay.identifier}) beginning at : {seg.measure_beg}")
                count_warning['tilde_errors'] += 1
                break
                
            
            n_loops += 1
            if n_loops > 100:
                if checking_syntax:
                    print(f"WARNING : Infinite loop in backpropagation of ~ at segment beginning at : {seg.measure_beg}")
                    count_warning['tilde_errors'] += 1
                break






def set_internal_bar_limits(score, checking_syntax=False, count_warning=None):
    """Sets the internal bar limits due to tilde operators

    Parameters
    ----------
    score : ScoreAnnot
        ScoreAnnot object -without error- initialized with the ScoreAnnot() constructor
    checking_syntax : bool
        Display verbose if True
    count_warning : dict
        Dictionnary gathering errors
    
    Returns
    -------
        ScoreAnnot object with internal bar limits modified if needed
    """
    for i_segment, seg in enumerate(score.list_segments): # we look at the segments
        for j_layer, lay in enumerate(seg.list_layers): # We look at all the layers in the current segment
            backtrack_tilde_with_identifier(lay, seg, score, i_segment, checking_syntax, count_warning)

    return score






# ===============================================================================================
# =============== DEPRECATED CODE ===============================================================
# ===============================================================================================


class AddedInstruments:
    """
    Instruments added to an ensemble at a ponctual moment inside the segment

    Attributes
    ----------
    measure_beg : int
        Entrance bar of the added instruments
    measure_end : int
        Left bar of the added instruments
    added_ensemble : [Ensemble]
        Instruments involved in these added instruments
        In practice, they are just simple instruments, but it can also be another AddedInstruments...
    """
    def __init__(self, str_added_instruments):
        # Je prefere quand meme forcer cette syntaxe, sinon, ca melange un peu de partout
        # split_str_added_instruments = re.split("\[|\]|-|{|\}", str_added_instruments)
        # self.measure_beg = int(split_str_added_instruments[1])
        # self.measure_end = int(split_str_added_instruments[2])
        # self.added_ensemble = Ensemble(split_str_added_instruments[4])

        split_str_added_instruments = re.split("\[|\]|{|\}", str_added_instruments)
        self.measure_beg, self.measure_end = beg_end(split_str_added_instruments[1])
        self.added_ensemble = Ensemble(split_str_added_instruments[3])
    
    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)
    
    def __repr__(self):
        return str(self)
    
    def get_measure_beg(self):
        return self.measure_beg
    
    def get_measure_end(self):
        return self.measure_end
    
    def get_length(self):
        return self.measure_end - self.measure_beg
        



class QRLayer(Layer):
    """
    A Question/Answer layer
    In practice stuff like "(QR n [.] [.])"

    Attributes
    ----------
    length_phrase : int
        Length of a question (or an answer)
    list_phrases : [Phrase]
        List of phrases
    """
    def __init__(self, str_qr, measure_beg, measure_end):
        super().__init__(measure_beg, measure_end, identifier=None)
        
        # Retrieve parameters
        str_qr = re.split("\(|\)", str_qr)
        str_parameters = re.split(r'\s', str_qr[1])
        self.length_phrase = str_parameters[1]


        # Analyze temporal functions
        str_temporal_qr_functions = str_parameters[2][1:-1]
        list_qr_functions_by_ensembles = re.split("\/", str_temporal_qr_functions)
    
        # Analyze instrument ensembles
        str_ensembles = str_parameters[3][1:-1]
        list_str_ensembles = re.split("\/", str_ensembles)
    
        assert len(list_qr_functions_by_ensembles) == len(list_str_ensembles), f"QRLayer [{str_qr}] : Not the same length for roles ({len(list_qr_functions_by_ensembles)}) and ensembles ({len(list_str_ensembles)})"
        
        self.list_phrases = []
        for i in range(len(list_str_ensembles)):
            current_phrase = Phrase(list_qr_functions_by_ensembles[i], list_str_ensembles[i])
            self.list_phrases.append(current_phrase)
        
    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def __eq__(self, value):
        return super().__eq__(value)

    def get_list_phrases(self):
        return self.list_phrases
    
    def get_length_phrase(self):
        return float(self.length_phrase)

