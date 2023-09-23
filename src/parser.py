"""
Parsing and unparsing functions
"""
from .classes import *
import os, io
import argparse
import pickle

# ==========================================
# ======= DISPLAY FUNCTIONS ================
# ==========================================

def print_separator():
    try:
        print("-"*os.get_terminal_size().columns)
    except OSError:
        print()

def print_list(l):
    for k in l:
        print(k)

def print_repr_list(l):
    for k in l:
        print(repr(k))




## ==============================================================
# ================= PARSE =======================================
# ==============================================================

def get_raw_segments(filename):
    """Retrieves the useful lines inside an .orchnot file

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    
    Returns
    -------
        List of raw useful lines
    """
    annot_file = open(filename, 'r')
    raw_lines = annot_file.readlines()
    annot_file.close()

    # Retrieve relevant lines (remove comments and line breaks)
    raw_segments = []
    for l in raw_lines:
        if not(l.isspace()) and (l[0] != "#"):
            if ("#" in l):
                relevant_line = l.split('#')[0].strip()
                raw_segments.append(relevant_line)
            else:
                raw_segments.append(repr(l)[1:-3])
    return raw_segments

def get_struct_segments(filename):
    """Retrieves the lines inside an .orch file with sonata form structure

    Parameters
    ----------
    filename : str
        Name of the .orch file
    
    Returns
    -------
        List of useful lines
    """
    annot_file = open(filename, 'r')
    raw_lines = annot_file.readlines()
    annot_file.close()

    # Retrieve relevant lines (remove comments and line breaks)
    raw_segments = []
    next = False
    for l in raw_lines:
        if not(l.isspace()) and ((l[0:2] == '#!') or next):
            if (l[0:2] == "#!"):
                raw_segments.append(l)
                next = True
            else:
                raw_segments.append(l)
                next = False
    return raw_segments


def to_score(filename):
    """Converts the content of an .orchnot file to a score object

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    
    Returns
    -------
        ScoreAnnot object
    """
    raw_segments = get_raw_segments(filename)
    score = ScoreAnnot(raw_segments)
    score = set_internal_bar_limits(score)
    return score



def export_object(filename):
    """Exports the content of an .orchnot file to a .pkl file as a ScoreAnnot object

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    """
    score = to_score(filename)

    new_filename = filename.split('.orchnot')[0]
    with open(new_filename + ".pkl", 'wb') as output_file:
        pickle.dump(score, output_file, pickle.HIGHEST_PROTOCOL)
    print("Export to " + new_filename + ".pkl")


def import_object(filename, DISPLAY=False):
    """Loads a ScoreAnnot object from a .pkl file

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    DISPLAY : bool
        Prints score if True
    
    Returns
    -------
        ScoreAnnot object
    """
    with open(filename, 'rb') as input_file:
        score = pickle.load(input_file)
    if DISPLAY:
        print(score)
    return score



def to_json(filename):
    """Export .orchnot file to .json

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    """
    score = to_score(filename)
    
    new_filename = filename.split('.orchnot')[0]
    with io.open(new_filename + ".json", 'w', encoding='utf-8') as f:
        f.write(str(score))
    print("Export to " + new_filename + ".json")


# ==============================================================
# ============== UNPARSE =======================================
# ==============================================================

def make_ensemble(list_ensemble):
    """Converts a list coming from an Ensemble object to a string
    
    Parameters
    ----------
    list_ensemble : [str|AddedInstruments]
        List of Instruments
    
    Returns
    -------
        String using the orchnot syntax
    """
    list_str_ens = []
    for ens in list_ensemble:
        if isinstance(ens, Instrument):
            this_instr = CYPHER_INSTRUMENT[str(ens.get_only_instr())]
            if ens.get_divisi() != -1:
                this_instr += str(ens.get_divisi())
            list_str_ens.append(this_instr)
        else:
            bar_beg = ens.measure_beg
            bar_end = ens.measure_end
            str_added = f"+[{bar_beg}-{bar_end}]" + "{" + make_ensemble(ens.added_ensemble.list_components) + "}"
            list_str_ens.append(str_added)

    return ".".join(list_str_ens)


def make_blend_layer(blend_layer):
    """Converts a BlendLayer object to a string
    
    Parameters
    ----------
    blend_layer : dict (BlendLayer)
        A BlendLayer
    
    Returns
    -------
        String using the orchnot syntax
    """
    role = CYPHER_ROLE[str(blend_layer.get_role().get_only_role())]
    if blend_layer.get_role().get_quantif() != "":
        role += blend_layer.get_role().get_quantif()
    
    relation = CYPHER_RELATION[str(blend_layer.function.relation)]
    ensemble = make_ensemble(blend_layer.ensemble.list_components)

    if blend_layer.same_function:
        return f"~({role}-{relation}:{ensemble})"
    else:
        return f"({role}-{relation}:{ensemble})" 


def make_qr_layer(qr_layer):
    """Converts a QRLayer object to a string
    
    Parameters
    ----------
    qr_layer : dict (QRLayer)
        A QRLayer
    
    Returns
    -------
        String using the orchnot syntax
    """
    str_qr = f"(QR {qr_layer.length_phrase}"

    list_phrases_function = [] # {fun}/{fun}
    list_phrases_ensembles = [] # {ens}/{ens}

    # Loop on phrases
    for phrase_obj in qr_layer.list_phrases:
        list_internal_qrlayer_function = [] # {many_fun|many_fun}
        list_internal_qr_ensembles = [] # {ens|ens}
        # Loop on internal layers
        for internal_layer_obj in phrase_obj.list_qr_internal_layers:
            list_internal_qr_ensembles.append(make_ensemble(internal_layer_obj.internal_ensemble.list_components))
            list_temporal_functions = [] # {fun,fun|
            # Loop on temporal functions
            for function in internal_layer_obj.internal_temporal_functions:
                role = CYPHER_ROLE[str(function.role.get_only_role())]
                relation = CYPHER_RELATION[str(function.relation)]
                list_temporal_functions.append(f"{role}-{relation}")
            list_internal_qrlayer_function.append(",".join(list_temporal_functions))

        list_phrases_function.append("|".join(list_internal_qrlayer_function))
        list_phrases_ensembles.append("|".join(list_internal_qr_ensembles))
    
    list_phrases_function = ["{"+ x +"}" for x in list_phrases_function]
    list_phrases_ensembles = ["{"+ x +"}" for x in list_phrases_ensembles]

    str_qr += " [" + "/".join(list_phrases_function) + "] [" + "/".join(list_phrases_ensembles) + "])"
    return str_qr



def make_segment(segment, with_label=False):
    """Converts a Segment object to a string
    
    Parameters
    ----------
    segment : dict (Segment)
        A Segment
    
    Returns
    -------
        String using the orchnot syntax
    """
    # Add bar numbers
    if with_label:
        str_segment = f"{segment.label} : [{segment.measure_beg}-{segment.measure_end}] "
    else:
        str_segment = f"[{segment.measure_beg}-{segment.measure_end}] "

    # Loop on layers
    list_layers_str = []
    for layer_obj in segment.list_layers:
        is_blend_layer = isinstance(layer_obj, BlendLayer)
        if is_blend_layer:
            str_blend_layer = make_blend_layer(layer_obj)
            list_layers_str.append(str_blend_layer)
        else:
            str_qr_layer = make_qr_layer(layer_obj)
            list_layers_str.append(str_qr_layer)


    str_segment += " / ".join(list_layers_str)
    return str_segment




def score_annot_to_orchnot(score, with_label=False):
    """Converts a json file produced by the parser to an orchnot string
    
    Parameters
    ----------
    filename : str
        Name of the json file

    Returns
    -------
        String using the orchnot syntax
    """
    orchnot_result = ""
    list_segments = score.list_segments

    for segment in list_segments:
        
        # Add bar numbers
        orchnot_result += make_segment(segment, with_label)
        orchnot_result += "\n"

    return orchnot_result




def test():
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=False, action='store_true')
    parser.add_argument("--show", required=False)
    parser.add_argument("--exportObject", required=False)
    parser.add_argument("--importObject", required=False)
    parser.add_argument("--toJSON", required=False)
    parser.add_argument("--unparseFromPkl", required=False)

    args = parser.parse_args()

    if args.test:
        test()
    elif args.show is not None:
        filename = args.show
        score = to_score(filename)
        print(score)
    elif args.exportObject is not None:
        filename = args.exportObject
        export_object(filename)
    elif args.importObject is not None:
        filename = args.importObject
        import_object(filename, DISPLAY=True)
    elif args.toJSON is not None:
        filename = args.toJSON
        to_json(filename)
    elif args.unparseFromPkl is not None:
        filename = args.unparseFromPkl
        score = import_object(filename)
        print(score_annot_to_orchnot(score))
    else:
        print("Argument needed")
