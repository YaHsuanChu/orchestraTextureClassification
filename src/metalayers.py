from ast import Raise
import re


def extract_metalayers(l,inst_list, list_segments):
    
    meta_list = []

    temp = l.split('{')

    for t in temp[1:]:
        metalayer = t.strip('} ')
        meta_list.append(metalayer)

    return meta_list




class Phrase:
    """
    One phrase inside a Question/Answer block.
    In practice, stuff like "{.}"
    Example : Haydn 94th Mvnt II, Bar 57-60 : 
        2 Phrases defined by the 2 ensembles "Vln2.Vc" and "Vln1.Vla"

    Attributes
    ----------
    list_qr_internal_layers : [QRInternalLayer]
        List of internal layers of the QR block
    """
    def __init__(self, str_qr_function, str_ensemble_qr):
        # print(str_qr_function)
        list_str_temporal_qr_function = str_qr_function[1:-1].split('|')
        list_str_ensembles = str_ensemble_qr[1:-1].split('|')
        # print(list_str_temporal_qr_function)

        assert len(list_str_ensembles) == len(list_str_temporal_qr_function), f"Phrase [{str_qr_function}] : Not the same length for temporal roles ({len(list_str_temporal_qr_function)}) and ensembles ({len(list_str_ensembles)})"

        self.list_qr_internal_layers = []
        for i in range(len(list_str_ensembles)):
            self.list_qr_internal_layers.append(QRInternalLayer(list_str_temporal_qr_function[i], list_str_ensembles[i]))
        
    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)
    
    def get_list_qr_internal_layers(self):
        return self.list_qr_internal_layers





class QRInternalLayer:
    """
    One layer inside a Phrase
    In practice, stuff separated by a "," inside a {.}
    Example : Haydn 94th Mvnt II, Bar 57-60 : 
        In the first phrase, there are 2 internal layers : "mel-u:Vln2" and "repeat_note-u:Vc"
    
    Attributes
    ----------
    internal_temporal_functions : [Function]
        Function of each ensemble through time (if it plays or not and what it plays)
    internal_ensemble : Ensemble
        Instruments involved in the layer    
    """
    def __init__(self, str_temporal_functions, str_ensemble):
        # print(str_temporal_functions, str_ensemble)
        list_functions = str_temporal_functions.split(',')
        # print(list_functions)
        self.internal_temporal_functions = [Function(f) for f in list_functions]
        self.internal_ensemble = Ensemble(str_ensemble)


    def __str__(self):
        return json.dumps(self, indent=INDENT_LENGTH, cls=ObjectEncoder)

    def __repr__(self):
        return str(self)

    def get_internal_temporal_functions(self):
        return self.internal_temporal_functions

    def get_internal_ensemble(self):
        return self.internal_ensemble
