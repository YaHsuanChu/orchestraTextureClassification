"""
Gathers all the stats on the corpus
"""

import os

from .classes import *
from .annot_types import Role, CYPHER_INSTRUMENT, DECYPHER_INSTRUMENT, SPECIAL_INSTR, DECYPHER_MAIN_INSTRUMENT
from .parser import to_score, get_raw_segments
from .processing import split_to_blend_layers, make_simple_ensembles, alter_score_annot
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import NormalDist
import time



LIST_POSSIBLE_INSTRUMENTS = [k for k in DECYPHER_MAIN_INSTRUMENT]
POSSIBLE_ROLES = [k for k in CYPHER_ROLE] + ["QR"]
POSSIBLE_RELATIONS = [k for k in CYPHER_RELATION] + ["QR"]
POSSIBLE_AGGR_ROLES = ["MAIN_MEL", "RHYTHM", "HARM", "DECORATIVE_MEL", "SPARSE", "None", "QR"]
RHYTHMIC_ROLES = ['RHYTHM', 'REPEAT_NOTE', 'OSCILLATION', 'BATTERIE', 'ARPEGGIO', 'SCALE']
VERY_ALL_RHYTHM = ['RHYTHM', 'REPEAT_NOTE', 'OSCILLATION', 'BATTERIE', 'ARPEGGIO', 'SCALE', 'SPARSE', 'DECMEL']
VERY_ALL_HARM = ['HARM', 'SPARSE']
VERY_ALL_MEL = ['MAIN_MEL', 'DECORATIVE_MEL']
OSCILLATING_ROLES = ['OSCILLATION', 'BATTERIE']

family_to_instr = {
    "woodwinds" : ['Fl', 'Ob', 'Cl', 'Fg'],
    "brass" : ['Hrn', 'Trp'],
    "strings" : ['Vln1', 'Vln2', 'Vla', 'Vc', 'Cb']
}

df_pieces = pd.read_csv('orch_metadata.csv')
LIST_COMPOSERS = df_pieces['composer'].unique().tolist()

DICT_PIECES = {}
for composer in LIST_COMPOSERS:
    DICT_PIECES[composer] = df_pieces.loc[df_pieces['composer'] == composer]['piece'].tolist()

LIST_PIECES = sum(DICT_PIECES.values(), [])


# ===============================================================================

woowdind_cmap = plt.get_cmap('Greens')
brass_cmap = plt.get_cmap('Oranges')
strings_cmap = plt.get_cmap('Blues')

dict_instr_color = {
    "TUTTI" : (128/255, 25/255, 166/255, 1),
    "Fl" : woowdind_cmap(0.4),
    "Ob" : woowdind_cmap(0.6),
    "Cl" : woowdind_cmap(0.8),
    "Fg" : woowdind_cmap(0.99),
    "Hrn" : brass_cmap(0.5),
    "Trp" : brass_cmap(0.75),
    "Timp" : (118/255, 83/255, 61/255, 1),
    "Vln1" : strings_cmap(0.4),
    "Vln2" : strings_cmap(0.55),
    "Vla" : strings_cmap(0.7),
    "Vc" : strings_cmap(0.85),
    "Cb" : strings_cmap(0.99),
}
instr_to_color = [d for k,d in dict_instr_color.items()]



rhythm_cmap = plt.get_cmap('cool')

dict_role_color = {
    "MAIN_MEL" : "#ebcc34",
    "RHYTHM" : "#349db3",
    "REPEAT_NOTE" : rhythm_cmap(0.01),
    "OSCILLATION" : rhythm_cmap(0.25),
    "BATTERIE" : rhythm_cmap(0.5),
    "ARPEGGIO" : rhythm_cmap(0.75),
    "SCALE" : rhythm_cmap(0.99),
    "HARM" : "#ff69d9",
    "DECORATIVE_MEL" : "#088c11",
    "SPARSE" : "#4a00ab",
    "None" : "#696969",
    "QR" : "#b30000",
}

role_to_color = [d for k,d in dict_role_color.items()]
role_split_to_color = ["#ebcc34", "#349db3"] + role_to_color[7:]

plt.rc('axes', axisbelow=True)

# =====================================================================================
# ================ UTILS FUNCTIONS ====================================================
# =====================================================================================

def timer(func):
    def wrapper(*args, **kwargs):
        beg_time = time.time()
        val = func(*args, **kwargs)
        tot_time = time.time() - beg_time
        print("Elapsed time : ", tot_time)
        return val
    return wrapper


def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def print_dict(d):
    for k, d in d.items():
        print(k, ":", d)


# =====================================================================================
# ================ GLOBAL FUNCTIONS ===================================================
# =====================================================================================

def mean_confidence_interval(data, confidence=0.95):
    """ Computes the average and the confidence interval of a list
    """
    m = np.mean(data)
    std = np.std(data)
    t = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = t*std/np.sqrt(len(data))
    down = m - h
    up = m + h
    return m, down, up, h



def get_all_instruments_playing_in_segment(segment, type_instr='str'):
    """ Gets all the instruments playing in the segment, no matter their layer
    """
    set_instruments = set()
    for layer in segment.get_list_layers():
        if isinstance(layer, BlendLayer):
            for instr in layer.get_ensemble().get_list_components():
                if type_instr == 'str':
                    set_instruments.add(instr)
                else:
                    set_instruments.add(instr.get_only_instr())
        if isinstance(layer, QRLayer):
            for phrase in layer.list_phrases:
                for internal_layer in phrase.list_qr_internal_layers:
                    for instr in internal_layer.get_internal_ensemble().get_list_components():
                        if type_instr == 'str':
                            set_instruments.add(instr)
                        else:
                            set_instruments.add(instr.get_only_instr())
        
    list_instruments = list(set_instruments)
    return list_instruments



def plot_array_instruments(array, SAVEFIG=False, SAVE_FILENAME="plot.pdf", vmax=None, dev=False):
    """Plots a matrix with instruments on the two axis

    Parameters
    ----------
    array : numpy.array
        Matrix to plot
    SAVEFIG : bool
        Save figure or not
    SAVE_FILENAME : str
        Name of the saved figure
    vmax : float
        Maximum range of the colorbar
    dev : bool
        To improve the layout of the figure

    Returns
    -------
        Axes if dev
    """
    # Remove TUTTI
    array = array[1:,1:] 
    LIST_TICKS = LIST_POSSIBLE_INSTRUMENTS[1:]
    
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color='k')

    fig, ax = plt.subplots(1,1)
    if vmax is None:
        mat_plt = ax.imshow(array, cmap=cmap)
    else:
        mat_plt = ax.imshow(array, vmin=0, vmax=vmax, cmap=cmap)

    ax.set_xticks(np.arange(0, len(LIST_TICKS), 1))
    ax.set_yticks(np.arange(0, len(LIST_TICKS), 1))
    ax.set_yticklabels(LIST_TICKS)
    ax.set_xticklabels(LIST_TICKS)
    plt.xticks(rotation=90)
    plt.colorbar(mat_plt)
    # plt.title(SAVE_FILENAME)
    plt.tight_layout()
    
    if dev:
        return ax
    else:
        if SAVEFIG:
            plt.savefig(SAVE_FILENAME, format='pdf', bbox_inches='tight')
        else:
            plt.show()

    


def move_legend(ax, new_loc, **kws):
    """ Moves legend of a plot
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)



def make_stacked_plot(df, xAxis, yAxis, hueAxis, palette=None, norm=False, ax=None):
    """Makes a stacked plot

    Parameters
    ----------
    df : pandas.DataFrame
        Content to plot
    xAxis : str
        Data to put on the xAxis. Name of a column of df
    yAxis : str
        Data to put on the yAxis. Name of a column of df
    hueAxis : str
        Data to put on the colorAxis. Name of a column of df
    palette : list || str
        Color palette
    norm : bool
        Normalize the data or not
    ax : matplotlib.Axes
        Axes to plot on

    Returns
    -------
        Axes with the stacked plot
    """
    df_plot = df.copy()
    if norm:
        sum_all = df_plot.groupby(xAxis).agg(sum).to_dict()[yAxis]
        for index, row in df_plot.iterrows():
            if sum_all[row[xAxis]] != 0:
                df_plot.loc[index, yAxis] = row[yAxis] / sum_all[row[xAxis]]
            else:
                df_plot.loc[index, yAxis] = 0

    if palette is not None:
        this_palette = sns.color_palette(palette)
    else:
        this_palette = 'tab10'

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    
    g = sns.histplot(df_plot, x=xAxis, hue=hueAxis, weights=yAxis, multiple='stack', ax=ax, legend=True, palette=this_palette)

    for patch in ax.patches:
        this_color = list(patch.get_facecolor())
        this_color[-1] = 1
        patch.set_color(tuple(this_color))

    move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.patches, linewidth=0)
    plt.grid()

    return ax


def make_stack_catplot(df, xAxis, xCat, yAxis, hueAxis, palette=None, norm=False, THIS_POSSIBLE_ROLES=POSSIBLE_ROLES, THIS_CAT_LIST=LIST_COMPOSERS, ticks_type='role'):
    """Makes a categorical stackedPlot with the roles as xAxis.
    Handles custom layout for roles. Allow normalized plots

    Parameters
    ----------
    df : pandas.DataFrame
        Content to plot
    xAxis : str
        Data to put on the xAxis. Name of a column of df
    xCat : str
        Category in which xAxis is gathered. Name of a column of df
    yAxis : str
        Data to put on the yAxis. Name of a column of df
    hueAxis : str
        Data to put on the colorAxis. Name of a column of df
    palette : list || str
        Color palette
    norm : bool
        Normalize the data or not
    THIS_POSSIBLE_ROLES : list
        List of Roles
    THIS_CAT_LIST : list
        List of Categories xtickslabels
    ticks_type : str
        Type of ticks : 'role' || 'all'

    Returns
    -------
        Axes with the catplot
    """
    new_name_axis = f"{xAxis}_{xCat}"
    df[new_name_axis] = df[xAxis] + "_" + df[xCat]
    ax = make_stacked_plot(df=df, xAxis=new_name_axis, yAxis=yAxis, hueAxis=hueAxis, palette=palette, norm=norm)
    plt.xticks(rotation=90)
    handle_presentation_stacked_role(ax, THIS_POSSIBLE_ROLES, THIS_CAT_LIST, ticks_type=ticks_type)
    return ax

# https://stackoverflow.com/questions/64084755/how-do-i-plot-stacked-barplots-side-by-side-in-python-preferentially-seaborn
    
def stack_catplot(x, y, cat, stack, data, palette, THIS_POSSIBLE_ROLES, DISPLAY=True):
    """Makes a catplot with any data. 
    But custom layout is not handled. Normalized data cannot be plot

    Parameters
    ----------
    x : str
        Data to put on the xAxis. Name of a column of df
    y : str
        Data to put on the yAxis. Name of a column of df
    cat : str
        Category in which xAxis is gathered. Name of a column of df
    stack : str
        Data to put on the colorAxis. Name of a column of df
    data : pandas.DataFrame
        Content to plot
    palette : list || str
        Color palette
    THIS_POSSIBLE_ROLES : list
        List of Roles
    DISPLAY : bool
        Displays the order of categorical labels
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    df = data.pivot_table(values=y, index=[cat, x], columns=stack, dropna=False, aggfunc='sum').fillna(0)

    if x == 'role':
        df = df.reindex(THIS_POSSIBLE_ROLES, level=1)
    ncat = data[cat].nunique()
    nx = data[x].nunique()
    nstack = data[stack].nunique()
    range_x = np.arange(nx)
    width = 0.8 / ncat # width of each bar
    
    for i, c in enumerate(data[cat].unique()):
        if DISPLAY:
            print(c)    
        # iterate over categories
        loc_x = (0.5 + i - ncat / 2) * width + range_x
        bottom = 0
        for j, s in enumerate(reversed(data[stack].unique())):
            # iterate over stacks
            height = df.loc[c][s].values
            ax.bar(x=loc_x, height=height, bottom=bottom, width=width, zorder=10, color=palette[nstack-j-1])
            bottom += height
    ax.set_xticks(range_x)
    ax.set_xticklabels(data[x].unique(), rotation=45)
    ax.set_ylabel(y)
    plt.legend([Patch(facecolor=palette[i]) for i in range(nstack)], 
                [f"{s}" for s in data[stack].unique()],
                bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.grid()
    if DISPLAY:
        print()


def handle_presentation_simple_stacked_composer_split(ax, ticks_type):
    """Handles custom layout to separate composers

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes containing the plot
    ticks_type : str
        Type of ticks : 'all' || 'composer'
    """
    # Handles presentaiton of graph: "For one role, xAxis=composer, xCat=piece"
    dict_xticks = {"mozart" : [], "haydn" : [], "beethoven" : []}
    n_mozart = len(DICT_PIECES['mozart'])
    n_haydn = len(DICT_PIECES['haydn'])
    n_beethoven = len(DICT_PIECES['beethoven'])
    for i, patch in enumerate(ax.patches):
        if i % len(LIST_PIECES) in list(range(0, n_mozart)): # Mozart
            dict_xticks['mozart'].append(patch.get_x() + 0.5)
            continue
        elif i % len(LIST_PIECES) in list(range(n_mozart, n_mozart+n_haydn)): # Haydn
            patch.set_x(patch.get_x()+2)
            dict_xticks['haydn'].append(patch.get_x() + 0.5)
        elif i % len(LIST_PIECES) in list(range(n_mozart+n_haydn, n_mozart+n_haydn+n_beethoven)): # Beethoven
            patch.set_x(patch.get_x()+4)
            dict_xticks['beethoven'].append(patch.get_x() + 0.5)

    list_xticks = dict_xticks['mozart'] + dict_xticks['haydn'] + dict_xticks['beethoven']

    # All title
    if ticks_type == 'all':
        plt.xticks(rotation=90)
        plt.xticks(list(set(list_xticks)), LIST_PIECES) # hehe, c'est moche
    # Just composer
    else:
        plt.xticks([np.mean(s) for s in dict_xticks.values()], LIST_COMPOSERS)

    plt.xlim([min(list_xticks)-2, max(list_xticks)+2])



def handle_presentation_stacked_role(ax, THIS_POSSIBLE_ROLES, THIS_CAT_LIST, ticks_type):
    """Handles plots such that xAxis=role, xCat=composer

    Parameters
    ----------
    ax : matplotlib.Aees
        Axes containing the plot
    THIS_POSSIBLE_ROLES : list
        List of Roles
    THIS_CAT_LIST : list
        List of Categories xtickslabels
    ticks_type : str
        Type of ticks : 'role' || 'all'
    """
    # Handles presentaiton of graph: "xAxis=role, xCat=composer"
    prop_merged_roles = {
        'mel' :    {'pos' : [0, 1, 2],  'move' : 0},
        'rhythm' : {'pos' : [3, 4, 5],  'move' : 0.25},
        'harm' :   {'pos' : [6, 7, 8],  'move' : 0.50},
        'decmel' : {'pos' : [9,10,11],  'move' : 0.75},
        'sparse' : {'pos' : [12,13,14], 'move' : 1},
        'None' :   {'pos' : [15,16,17], 'move' : 1.25},
        'QR' :     {'pos' : [18,19,20], 'move' : 1.5},
    }

    prop_all_roles = {
        'mel' :         {'pos' : [0, 1, 2],  'move' : 0},

        'rhythm' :      {'pos' : [3, 4, 5],  'move' : 2},
        'repeat_note' : {'pos' : [6, 7, 8],  'move' : 2.25},
        'osc' :         {'pos' : [9,10,11],  'move' : 2.5},
        'batt' :        {'pos' : [12,13,14], 'move' : 2.75},
        'arp' :         {'pos' : [15,16,17], 'move' : 3},
        'scale' :       {'pos' : [18,19,20], 'move' : 3.25},
        
        'harm' :        {'pos' : [21,22,23], 'move' : 5.25},
        'decmel' :      {'pos' : [24,25,26], 'move' : 7.25},
        'sparse' :      {'pos' : [27,28,29], 'move' : 9.25},
        'None' :        {'pos' : [30,31,32], 'move' : 11.25},
        'QR' :          {'pos' : [33,34,35], 'move' : 13.25},
    }

    if len(THIS_POSSIBLE_ROLES) == 7:
        split_roles_prop = prop_merged_roles
    elif len(THIS_POSSIBLE_ROLES) == 12:
        split_roles_prop = prop_all_roles
    else:
        print("handle_presentation_stacked_role: THIS_POSSIBLE_ROLES not well-defined")

    dict_xticks = dict((k, []) for k in split_roles_prop)
    when_change = len(THIS_POSSIBLE_ROLES) * len(THIS_CAT_LIST)

    def change_position(i, patch):
        for key, prop in split_roles_prop.items():
            if i % when_change in prop['pos']:
                patch.set_x(patch.get_x() + prop['move'])
                dict_xticks[key].append(patch.get_x() + 0.5)
                return
        print("handle_presentation_stacked_role: Error in generating stacked_plot")
        
    for i, patch in enumerate(ax.patches):
        change_position(i, patch)

    list_xticks = sum([l for k, l in dict_xticks.items()], [])
    
    if ticks_type == 'all':
        plt.xticks(list(set(list_xticks)), [f"{role}_{composer}" for role in THIS_POSSIBLE_ROLES for composer in THIS_CAT_LIST])
        plt.xticks(rotation=90)
    else:
        plt.xticks([np.mean(s) for s in dict_xticks.values()], THIS_POSSIBLE_ROLES)

    plt.xlim([min(list_xticks)-2, max(list_xticks)+2])


# ===================================================================
# =================== OTHER COUNTS ==================================
# ===================================================================     


def count_bar_played_instruments():
    """Counts the number of bars instruments play

    Returns
    -------
    pandas.DataFrame
        Bar count, average for each instrument
    """
    dict_instrument_to_count = dict((k, 0) for k in LIST_POSSIBLE_INSTRUMENTS)
    n_tot_bars = 0

    def update_dict_instrument_to_count(score):
        for segment in score.get_list_segments():
            list_instruments_playing = get_all_instruments_playing_in_segment(segment)
            for instr in list_instruments_playing:
                cypher_instr = CYPHER_INSTRUMENT[instr.get_only_instr()]
                dict_instrument_to_count[cypher_instr] += segment.get_length()
                

    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")

        annot_score = to_score(filename)
        score = split_to_blend_layers(annot_score, DISPLAY=False)
        make_simple_ensembles(score)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score, "add_cb")

        update_dict_instrument_to_count(score)
        n_tot_bars += score.get_length()

    
    df_instr = pd.DataFrame({
        'instrument' : [k for k in dict_instrument_to_count.keys()],
        'count' : [k for k in dict_instrument_to_count.values()],
        'avg' : [k/n_tot_bars for k in dict_instrument_to_count.values()]
    }).set_index('instrument')

    dict_family_to_count = dict()
    dict_family_to_count['woodwinds'] = np.mean([df_instr['avg'][i] for i in family_to_instr['woodwinds']])
    dict_family_to_count['brass'] = np.mean([df_instr['avg'][i] for i in family_to_instr['brass']])
    dict_family_to_count['strings'] = np.mean([df_instr['avg'][i] for i in family_to_instr['strings']])

    df_family = pd.DataFrame({
        'family' : [k for k in dict_family_to_count.keys()],
        'avg' : [k for k in dict_family_to_count.values()]
    }).set_index('family')

    return df_instr, df_family


def count_independant_bass_strings():
    """Counts bars in which Vln2.Vla.Vc.Cb play the melody without Vln1
    """
    def check_role_instrument(layer, instrument, role):
        for instr in layer.get_ensemble().get_list_components():
            if str(instr.get_only_instr()) == instrument:
                return layer.get_role().get_only_role() == role
        return False

    list_ratios = []

    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")

        annot_score = to_score(filename)
        score = split_to_blend_layers(annot_score, DISPLAY=False)
        make_simple_ensembles(score)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score, "add_cb")
        
        n_bass_mel_not_vln1 = 0
        n_normalization = 0

        for segment in score.get_list_segments():
            vln1_playing_mel = False
            bass_playing_mel = False
            for layer in segment.get_list_layers():
                vln1_playing_mel = vln1_playing_mel or check_role_instrument(layer, 'VIOLIN1', 'MAIN_MEL')
                for instr_bass in ['VIOLIN2', 'VIOLA', 'CELLO', 'CONTREBASS']:
                    bass_playing_mel = bass_playing_mel or check_role_instrument(layer, instr_bass, 'MAIN_MEL')
            if bass_playing_mel and not(vln1_playing_mel):
                n_bass_mel_not_vln1 += segment.get_length()
#             if bass_playing_mel and vln1_playing_mel:
            n_normalization += segment.get_length()
            
#             print(vln1_playing_mel, bass_playing_mel, n_bass_mel_not_vln1, n_playing_mel_together, make_segment(segment)) # on cherche du False True
        print("# bars where melodic bass strings only / length piece", n_bass_mel_not_vln1 / n_normalization)
        list_ratios.append(n_bass_mel_not_vln1 / n_normalization)
    
    print()
    print("Avg # bars where melodic bass strings only / length piece : ", np.mean(list_ratios))


def count_tuttis():
    """Counts number of TUTTI (or full orchestral) bars
    """
    n_tuttis = 0
    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")
        
        annot_score = to_score(filename)
        score = split_to_blend_layers(annot_score, DISPLAY=False)
        make_simple_ensembles(score)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score, "add_cb")
            
        for segment in score.get_list_segments():
            list_instr_playing = get_all_instruments_playing_in_segment(segment, type_instr='instr')
            if len(list_instr_playing) >= 12 or ('TUTTI' in list_instr_playing):
                n_tuttis += segment.get_length()
#                 print(len(list_instr_playing), make_segment(segment))
    return n_tuttis

# ===================================================================
# =================== STATS HOW MANY ROLES ==========================
# ===================================================================        


def compute_mean_length_annotation(score):
    """Computes average length of Layers in a ScoreAnnot
    One N-long tilde layer is counted as one.
    """
    layer_lengths = []
    for segment in score.get_list_segments():
        for layer in segment.get_list_layers():
            if not(layer.same_function): # Get only ONCE each layer (in case the layer is tilde)
                layer_lengths.append(layer.get_length())
    return np.mean(layer_lengths)


def compute_n_layers_tilde_merged(score):
    """Computes the number of layers ScoreAnnot
    One N-long tilde layer is counted as one.
    """
    n_layers = 0
    for segment in score.get_list_segments():
        for layer in segment.get_list_layers():
            if not(layer.same_function): # Get only ONCE each layer (in case the layer is tilde)
                n_layers += 1
    return n_layers



def compute_mean_instr_per_bar(score_blend):
    """Computes average number of instruments per layer
    """
    list_n_instr = []
    for segment in score_blend.get_list_segments():
        for layer in segment.get_list_layers():
            if isinstance(layer, BlendLayer):
                list_n_instr.append(len(layer.get_ensemble().get_list_components()))
            if isinstance(layer, QRLayer):
                for phrase in layer.list_phrases:
                        for internal_layer in phrase.list_qr_internal_layers:
                            list_n_instr.append(len(internal_layer.get_internal_ensemble().get_list_components())) 

    return np.mean(list_n_instr)




def update_count_role_relation_dataset(role_to_count, relation_to_count, score_blend):
    """ Updates dict counting relation/role occurences within a score
    """
    for segment in score_blend.get_list_segments():
        tot_bars = segment.get_length()
        role_exists = dict([(role, False) for role in role_to_count])
        for layer in segment.get_list_layers():
            if isinstance(layer, BlendLayer):
                # role_to_count[str(layer.get_role().get_only_role())] += tot_bars
                role_exists[str(layer.get_role().get_only_role())] = True

                # To have the correct number of rhythm layers (all counted)
                role_exists['ALL_RHYTHM'] = role_exists['ALL_RHYTHM'] or str(layer.get_role().get_only_role()) in RHYTHMIC_ROLES
                role_exists['ALL_OSCILLATION'] = role_exists['ALL_OSCILLATION'] or str(layer.get_role().get_only_role()) in OSCILLATING_ROLES
                
                role_exists['ALL_VERY_RHYTHMIC'] = role_exists['ALL_VERY_RHYTHMIC'] or str(layer.get_role().get_only_role()) in VERY_ALL_RHYTHM
                role_exists['ALL_VERY_HARMONIC'] = role_exists['ALL_VERY_HARMONIC'] or str(layer.get_role().get_only_role()) in VERY_ALL_HARM
                role_exists['ALL_VERY_MELODIC'] = role_exists['ALL_VERY_MELODIC'] or str(layer.get_role().get_only_role()) in VERY_ALL_MEL

                if str(layer.get_relation() == 'UNISSON'):
                    if (len(layer.get_ensemble().get_list_components()) > 1):
                        relation_to_count[str(layer.get_relation())] += tot_bars
                    else:
                        relation_to_count['SOLO'] += tot_bars
                else:
                    relation_to_count[str(layer.get_relation())] += tot_bars
            if isinstance(layer, QRLayer):
                
                role_to_count["QR"] += tot_bars
                relation_to_count["QR"] += tot_bars
        
    
        for role, exist in role_exists.items():
            if exist:
                role_to_count[role] += tot_bars



def plot_count_roles_in_dataset(dict_piece_to_data, type_aggregation):
    """Plots number of roles in the dataset

    Parameters
    ----------
    dict_piece_to_data : dict
        Dictionnary gathering, for each piece, the number of roles in that piece
    type_aggregation : bool
        Plot for all dataset, aggregated by composer, aggregated by piece
    """
    dict_df = {'piece' : [], 'length': [], 'composer': [],  'role' : [], 'count' : []}
    for piece, data in dict_piece_to_data.items():
        dict_df['piece'] += [piece] * len(data['role_to_count'])
        dict_df['length'] += [data['length']] * len(data['role_to_count'])
        dict_df['composer'] += [piece.split('_')[0]] * len(data['role_to_count'])
        dict_df['role'] += data['role_to_count'].keys()
        dict_df['count'] += data['role_to_count'].values()
    
    df = pd.DataFrame(dict_df)

    df_lengths = df.groupby('piece').agg('mean')
    

    if type_aggregation == 'all':
        y = df.groupby('role', sort=False).agg(sum)['count']
        length_tot = sum(df_lengths['length'])
        y_norm = y / length_tot
        plt.bar(POSSIBLE_ROLES, y_norm)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"count_roles_in_dataset_all.pdf", format='pdf', bbox_inches='tight')

    elif type_aggregation == 'composer':
        y_df = df.groupby(['composer', 'role'], sort=False, as_index=False).agg(sum)
        y_df['count'] /= y_df['length']
        sns.barplot(x='role', y='count', hue='composer', data=y_df)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"count_roles_in_dataset_composer.pdf", format='pdf', bbox_inches='tight')

    elif type_aggregation == 'piece':
        for composer in LIST_COMPOSERS:
            plt.figure()
            y_df = df.loc[df['composer'] == composer].copy()
            y_df['count'] = y_df['count'] / y_df['length']
            ax = make_stacked_plot(df=y_df, xAxis='piece', yAxis='count', hueAxis='role', palette=role_to_color, norm=False)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"count_roles_in_dataset_piece_{composer}_.pdf", format='pdf', bbox_inches='tight')
    
    


def stats_count_roles_in_dataset(type_aggregation='all', only_stats=False, dev=False):
    """Counts relation and roles occurrences in dataset

    Parameters
    ----------
    type_aggregation : str
        Type of aggregation : 'all', 'composer' or 'piece'
    only_stats : bool
        Displays only stats if True
    dev : bool
        Returns dictionnaries of roles and relations if True
    """
    dict_piece_to_role = dict()
    dict_piece_to_relation = dict()

    # Number of bars
    tot_length = 0

    # Number of annotated layers in RAW (i.e. number of "()")
    tot_raw_layers = 0

    # Number of annotated layers when split in BlendLayers
    tot_blend_layers = 0

    # Number of annotated layers by bar
    tot_layers_bars = 0

    list_mean_lengths = []
    list_mean_n_instr = []

    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")

        role_to_count = dict((k, 0) for k in POSSIBLE_ROLES + ['ALL_RHYTHM', 'ALL_OSCILLATION', 'ALL_VERY_RHYTHMIC', 'ALL_VERY_MELODIC', 'ALL_VERY_HARMONIC'])
        relation_to_count = dict((k, 0) for k in POSSIBLE_RELATIONS + ['SOLO'])

        annot_score = to_score(filename)
        score_blend = split_to_blend_layers(annot_score, DISPLAY=False)
        make_simple_ensembles(score_blend)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score_blend, "add_cb")

        update_count_role_relation_dataset(role_to_count, relation_to_count, score_blend)
        mean_length = compute_mean_length_annotation(annot_score)
        n_layers = compute_n_layers_tilde_merged(annot_score)
        mean_n_instr = compute_mean_instr_per_bar(score_blend)

        dict_piece_to_role[piece_row['piece']] = {
            'role_to_count' : role_to_count,
            'length' : score_blend.get_length()
        }

        dict_piece_to_relation[piece_row['piece']] = {
            'relation_to_count' : relation_to_count,
            'length' : score_blend.get_length()
        }


        if (type_aggregation == 'all'):
            tot_length += score_blend.get_length()
            print("# bars : ", score_blend.get_length())

            # Count a N-long tilde layer as 1 unique layer
            tot_raw_layers += n_layers
            print("# annotated layers : ", n_layers)

            # # Count a N-long tilde layer as 3 distinct layers
            # tot_raw_layers += annot_score.get_n_annotated_layers()
            # print("# annotated layers : ", annot_score.get_n_annotated_layers())

            # tot_blend_layers += score_blend.get_n_annotated_layers()
            # print("# annotated layers when split in BlendLayers: ", score_blend.get_n_annotated_layers())
            # tot_layers_bars += score_blend.get_n_annotated_layers_by_bar()
            # print("# annotated layers by bar : ", score_blend.get_n_annotated_layers_by_bar())
            list_mean_lengths.append(mean_length)
            print("Average length of an annotation (in bars) : ", np.round(mean_length, 2))
            list_mean_n_instr.append(mean_n_instr)
            print("Average number of instruments in one layer : ", np.round(mean_n_instr, 2))

            print("---")

    if (type_aggregation == 'all'):
        print("Total # bars : ", tot_length)
        print("Total # annotated layers : ", tot_raw_layers)
        # print("Total # annotated layers when split in BlendLayers : ", tot_blend_layers)
        # print("Total # annotated layers by bar (CR counted as one layer): ", tot_layers_bars)
        print("Total average length of annotations (in bars) : ", np.round(np.mean(list_mean_lengths), 2))
        print("Total average number of instruments in one layer : ", np.round(np.mean(list_mean_n_instr),2))

    if not(only_stats):
        return plot_count_roles_in_dataset(dict_piece_to_role, type_aggregation)
    if dev:
        return dict_piece_to_role, dict_piece_to_relation



# ===============================================================
# =================== STATS SAME LAYER ==========================
# ===============================================================

def update_array_from_ensemble(same_layer_array, list_components, tot_bars):
    """Updates the association matrix
    """
    for instr1 in list_components:
        index_instr1 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[instr1.get_only_instr()])
        for instr2 in list_components:
            index_instr2 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[instr2.get_only_instr()])
            if index_instr1 != index_instr2:
                same_layer_array[index_instr1, index_instr2] += tot_bars

def update_opposition_array(opposed_layer_array, ens1, ens2, tot_bars, full_matrix=True):
    """Updates the opposition matrix
    """
    for instr1 in ens1:
        index_instr1 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[instr1.get_only_instr()])
        for instr2 in ens2:
            index_instr2 = LIST_POSSIBLE_INSTRUMENTS.index(CYPHER_INSTRUMENT[instr2.get_only_instr()])
            if index_instr1 != index_instr2:
                if full_matrix:
                    # Full matrice -> matrice sympétrique
                    opposed_layer_array[index_instr1, index_instr2] += tot_bars
                    opposed_layer_array[index_instr2, index_instr1] += tot_bars
                else:
                    # Juste la sous-matrice inferieure
                    opposed_layer_array[max(index_instr1, index_instr2), min(index_instr1, index_instr2)] += tot_bars



# @timer
def update_same_layer_array(same_layer_array, score_blend, type_layer, ignore_bars, full_matrix):
    """Makes the association/opposition matrix

    Parameters
    ----------
    same_layer_array : numpy.array
        Association/opposition matrix
    score_blend : ScoreAnnot
        ScoreAnnot to be analyzed
    type_layer : str
        Type of matrix
    ignore_bars : bool
        Add + n_bars at each cell if False
    full_matrix : bool
        Makes the full matrix or only half of the matrix
    """
    # Pour ne mesurer que la repartition dans les layers, sans prendre en compte la longueur du layer, ne pas mettre tot_bars
    if type_layer == "concurrent": 
        # To see who plays with who WITHIN a layer
        for segment in score_blend.get_list_segments():
            if ignore_bars:
                tot_bars = 1
            else:
                tot_bars = segment.get_length()
            for layer in segment.get_list_layers():
                if isinstance(layer, BlendLayer):
                    update_array_from_ensemble(same_layer_array, layer.ensemble.list_components, tot_bars)
                if isinstance(layer, QRLayer):
                    for phrase in layer.list_phrases:
                        for internal_layer in phrase.list_qr_internal_layers:
                            update_array_from_ensemble(same_layer_array, internal_layer.internal_ensemble.list_components, tot_bars)


    elif type_layer == "global":
        # To see who plays with who GLOBALLY in a segment
        for segment in score_blend.get_list_segments():
            if ignore_bars:
                tot_bars = 1
            else:
                tot_bars = segment.get_length()
            list_instruments = get_all_instruments_playing_in_segment(segment)
            update_array_from_ensemble(same_layer_array, list_instruments, tot_bars)

    elif type_layer == "CR_all":
        for segment in score_blend.get_list_segments():
            if ignore_bars:
                tot_bars = 1
            else:
                tot_bars = segment.get_length()
            for layer in segment.get_list_layers():
                if isinstance(layer, QRLayer):
                    if len(layer.get_list_phrases()) != 2:
                        continue  
                    list_instruments = []
                    for phrase in layer.list_phrases:
                        for internal_layer in phrase.list_qr_internal_layers:
                            list_instruments += internal_layer.internal_ensemble.list_components
                    update_array_from_ensemble(same_layer_array, list_instruments, tot_bars)


    elif type_layer == "CRopp":
        for segment in score_blend.get_list_segments():
            if ignore_bars:
                tot_bars = 1
            else:
                tot_bars = segment.get_length()
            for layer in segment.get_list_layers():
                if isinstance(layer, QRLayer):
                    if len(layer.get_list_phrases()) != 2:
                        continue  
                    ens1, ens2 = [], []
                    for internal_layer in layer.get_list_phrases()[0].get_list_qr_internal_layers():
                        ens1 += internal_layer.get_internal_ensemble().list_components
                    for internal_layer in layer.get_list_phrases()[1].get_list_qr_internal_layers():
                        ens2 += internal_layer.get_internal_ensemble().list_components
                    
                    update_opposition_array(same_layer_array, ens1, ens2, tot_bars, full_matrix)

    elif type_layer == "CR_same":
        for segment in score_blend.get_list_segments():
            if ignore_bars:
                tot_bars = 1
            else:
                tot_bars = segment.get_length()
            for layer in segment.get_list_layers():
                if isinstance(layer, QRLayer):
                    if len(layer.get_list_phrases()) != 2:
                        continue  
                    for phrase in layer.list_phrases:
                        list_instruments = []
                        for internal_layer in phrase.list_qr_internal_layers:
                            list_instruments += internal_layer.internal_ensemble.list_components
                        update_array_from_ensemble(same_layer_array, list_instruments, tot_bars)



def plot_same_layer_array(same_layer_array, type_layer, norm='line'):
    """Plots the association/opposition matrix
    """
    # No norm
    if norm == 'no':
        same_layer_array_norm = same_layer_array

    # Whole dataset
    elif norm == 'all':
        same_layer_array_norm = same_layer_array / np.sum(same_layer_array)
    
    # Per instrument (sum per line is 1)
    elif norm == 'line':
        same_layer_array_norm = same_layer_array / np.sum(same_layer_array, axis=1, keepdims=1)
        same_layer_array_norm[np.isnan(same_layer_array_norm)] = 0
    
    if type_layer == "global":
        plot_array_instruments(same_layer_array_norm, True, "composition_layer_global.pdf")
    elif type_layer == "concurrent":
        plot_array_instruments(same_layer_array_norm, True, "composition_layer_concurrent.pdf")
    elif type_layer == "CRopp":
        plot_array_instruments(same_layer_array_norm, True, "composition_phrase_opposition_CR.pdf")


def stats_same_layer(type_layer, norm='line', ignore_bars=True, stats=False, full_matrix=True):
    """Runs analysis of association/opposition matrix

    Parameters
    ----------
    type_layer : str
        Type of matrix : 'concurrent', 'global', 'CRopp', 'CR_all', 'CR_same'
    norm : str
        Normalize by what : 'line', 'no', 'all'
    ignore_bars : bool
        Add + n_bars at each cell if False
    stats : bool
        Displays stats if True
    full_matrix : bool
        Makes the full matrix or only half of the matrix
    """
    same_layer_array = np.zeros((len(LIST_POSSIBLE_INSTRUMENTS),len(LIST_POSSIBLE_INSTRUMENTS)))
    qr_split = True if type_layer in ["concurrent", "global"] else False
    for path, subdirs, files in os.walk("annotations"):
        if ("a_finir" in path) or ("pending" in path):
            continue
        for name in files:
            if not(".orchnot" in name):
                continue
            filename = os.path.join(path, name)
            print(filename, "processing...")
            annot_score = to_score(filename)
            score_blend = split_to_blend_layers(annot_score, DISPLAY=False, qr_split=qr_split)
            update_same_layer_array(same_layer_array, score_blend, type_layer, ignore_bars, full_matrix)

    if stats:
        same_layer_array_norm = same_layer_array[1:,1:]
        # print(same_layer_array_norm)
        same_layer_array_norm = same_layer_array[1:,1:] / np.sum(same_layer_array[1:,1:])

        woodwinds_intra_cr = same_layer_array_norm[0:4,0:4].copy()
        brass_intra_cr = same_layer_array_norm[4:7,4:7].copy()
        strings_intra_cr = same_layer_array_norm[7:13,7:13].copy()

        inter_brass_woodwinds = same_layer_array_norm[4:7,0:4].copy()
        inter_strings_woodwinds = same_layer_array_norm[7:13,0:4].copy()
        inter_strings_brass = same_layer_array_norm[7:13,4:7].copy()  

        # print(woodwinds_intra_cr * np.sum(same_layer_array[1:,1:]))
        # print(brass_intra_cr * np.sum(same_layer_array[1:,1:]))
        # print(strings_intra_cr * np.sum(same_layer_array[1:,1:]))

        print("---")
        print("Intra CR woodwinds :\t", np.sum(woodwinds_intra_cr))
        print("Intra CR brass :\t", np.sum(brass_intra_cr))
        print("Intra CR strings :\t", np.sum(strings_intra_cr))
        print()
        print("Inter CR brass/woodwinds:\t", np.sum(inter_brass_woodwinds))
        print("Inter CR strings/woodwinds:\t", np.sum(inter_strings_woodwinds))
        print("Inter CR strings/brass:\t\t", np.sum(inter_strings_brass))
        print()
        print("---")
        print("Intra winds :\t\t", np.sum(woodwinds_intra_cr) + np.sum(brass_intra_cr) + np.sum(inter_brass_woodwinds))
        print("Intra strings :\t\t", np.sum(strings_intra_cr))
        print("Inter strings/winds :\t", np.sum(inter_strings_woodwinds) + np.sum(inter_strings_brass))

    plot_same_layer_array(same_layer_array, type_layer, norm)

    



# ==============================================================================
# =================== STATS ASSOCIATION IN SAME LAYER ==========================
# ==============================================================================
# "quand timbale et Trp jouent ensemble, 60% du temps c'est même layer"



def process_association_same_layer(array_n_bars_same_segment, array_n_bars_same_layer, score_blend):
    """Makes co-occurrence matrix for all layers
    """
    update_same_layer_array(array_n_bars_same_segment, score_blend, "global", ignore_bars=False, full_matrix=True)
    update_same_layer_array(array_n_bars_same_layer, score_blend, "concurrent", ignore_bars=False, full_matrix=True)


def stats_association_in_same_layer(dev=False):
    """Runs association analysis for all layers

    Parameters
    ----------
    dev : bool
        Returns matrix if True
    """
    array_n_bars_same_segment = np.zeros((len(LIST_POSSIBLE_INSTRUMENTS), len(LIST_POSSIBLE_INSTRUMENTS)))
    array_n_bars_same_layer = np.zeros((len(LIST_POSSIBLE_INSTRUMENTS), len(LIST_POSSIBLE_INSTRUMENTS)))

    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")

        annot_score = to_score(filename)
        score_blend = split_to_blend_layers(annot_score, DISPLAY=False, qr_split=False)
        make_simple_ensembles(score_blend)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score_blend, "add_cb")

        process_association_same_layer(array_n_bars_same_segment, array_n_bars_same_layer, score_blend)

    if dev:
        return array_n_bars_same_layer / array_n_bars_same_segment
    else:
        plot_array_instruments(array_n_bars_same_layer / array_n_bars_same_segment, True, "association_layer_segment.pdf", vmax=1)



# =========================================================================
# =================== STATS ROLES PER INSTRUMENT ==========================
# =========================================================================


def set_df(piece_to_data, THIS_POSSIBLE_ROLES):
    """Makes a pandas.DataFrame containing count of roles per instrument for each piece
    """
    def process_data_piece(data_piece):
        count_instrument_to_role = dict()
        for instr, data in data_piece['instruments_to_role'].items():
            values, count = np.unique(np.array(data), return_counts=True)
            this_count = [0]*len(THIS_POSSIBLE_ROLES)

            for i, role in enumerate(values):
                this_count[THIS_POSSIBLE_ROLES.index(role)] = count[i]
            
            count_instrument_to_role[instr] = {
                'role' : THIS_POSSIBLE_ROLES,
                'count' : this_count
            }

        return count_instrument_to_role


    dict_df = {"composer" : [], "piece" : [], "length" : [], "instrument" : [], "role" : [], "count" : []}

    for piece, piece_data in piece_to_data.items():
        total_length = len(LIST_POSSIBLE_INSTRUMENTS) * len(THIS_POSSIBLE_ROLES)
        this_composer = df_pieces.loc[df_pieces['piece'] == piece]['composer'].values[0]
        dict_df['composer'] += [this_composer] * total_length
        dict_df['piece'] += [piece] * total_length
        dict_df['length'] += [piece_data['length']] * total_length
        count_instrument_to_role = process_data_piece(piece_data)
        for instr, instr_data in count_instrument_to_role.items():
            dict_df['instrument'] += [instr] * len(THIS_POSSIBLE_ROLES)
            dict_df['role'] += instr_data['role']
            dict_df['count'] += instr_data['count']
        
    df = pd.DataFrame(dict_df)
    

    return df

# ----------------------------------------------------------------------------
# InstrX ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def plot_instr_role(piece_to_data, split_role):
    """Plots a stacked_plot with instrument as Xaxis stacked by roles

    Parameters
    ----------
    piece_to_data : dict
        Dictionnary computed by update_dict_instr_role
    split_role : bool
        Aggregate roles or not
    """
    if split_role:
        THIS_POSSIBLE_ROLES = POSSIBLE_ROLES
        split_status = "split"
        this_palette = role_to_color
    else:
        THIS_POSSIBLE_ROLES = ["MAIN_MEL", "RHYTHM", "HARM", "DECORATIVE_MEL", "SPARSE", "None", "QR"]
        split_status = "aggr"
        this_palette = role_split_to_color


    df = set_df(piece_to_data, THIS_POSSIBLE_ROLES)
    y_df = df.groupby(['instrument', 'role'], sort=False, as_index=False).agg(sum)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Stacked plot
    ax = make_stacked_plot(df, xAxis='instrument', yAxis='count', hueAxis='role', palette=this_palette, ax=ax, norm=True)

    # Bar plot
    # sns.barplot(x='instrument', y='count', hue='role', data=y_df, ax=ax, palette=this_palette)
    # move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if split_role:
        plt.savefig(f"instr_role_{split_status}.pdf", format='pdf', bbox_inches='tight')
    else:
        plt.savefig(f"instr_role_{split_status}.pdf", format='pdf', bbox_inches='tight')



# ----------------------------------------------------------------------------
# RoleX ----------------------------------------------------------------------
# ----------------------------------------------------------------------------



def plot_role_instr(piece_to_data, norm, split_role, type_aggregation):
    """Plots a stacked_plot with roles as Xaxis stacked by instrument

    Parameters
    ----------
    piece_to_data : dict
        Dictionnary computed by update_dict_instr_role()
    norm : bool
        Normalize stacked_plot or not
    split_role : bool
        Aggregate roles or not
    type_aggregation : str
        Type of aggregation

    Returns
    -------
    pandas.DataFrame
        DataFrame computed by set_df()
    """
    if split_role:
        THIS_POSSIBLE_ROLES = POSSIBLE_ROLES
        split_status = "split"
    else:
        THIS_POSSIBLE_ROLES = ["MAIN_MEL", "RHYTHM", "HARM", "DECORATIVE_MEL", "SPARSE", "None", "QR"]
        split_status = "aggr"

    def handle_presentation_global(ax) :
        # To split mel/rythm/harm/mixed/other bars
        if not(split_role):
            return
        set_xticks = set()
        for i, patch in enumerate(ax.patches):
            if i % len(POSSIBLE_ROLES) in [0]: # Mel
                set_xticks.add(patch.get_x() + 0.5)
            elif i % len(POSSIBLE_ROLES) in [1, 2, 3, 4, 5, 6]: # Rythm
                patch.set_x(patch.get_x()+0.25)
                set_xticks.add(patch.get_x() + 0.5)
            elif i % len(POSSIBLE_ROLES) in [7]: # Harm
                patch.set_x(patch.get_x()+0.5)
                set_xticks.add(patch.get_x() + 0.5)
            elif i % len(POSSIBLE_ROLES) in [8, 9]: # Decmel, sparse
                patch.set_x(patch.get_x()+0.75)
                set_xticks.add(patch.get_x() + 0.5)
            else:
                patch.set_x(patch.get_x()+1)
                set_xticks.add(patch.get_x() + 0.5)
                
        plt.xticks(list(set_xticks), POSSIBLE_ROLES)
        plt.xlim([min(set_xticks)-0.75, max(set_xticks)+0.75])

        if norm:
            plt.ylim([0,1])



    def common_layout():
        plt.xticks(rotation=90)
        plt.tight_layout()


    df = set_df(piece_to_data, THIS_POSSIBLE_ROLES)
    df_lengths = df.groupby('piece').agg('mean')

    if type_aggregation == 'all':
        # Single stacked_plot for all the corpus
        length_tot = sum(df_lengths['length'])
        y_df = df.groupby(['role', 'instrument'], sort=False, as_index=False).agg(sum)
        y_df['count'] /= length_tot
        ax = make_stacked_plot(y_df, xAxis='role', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm)
        handle_presentation_global(ax)
        plt.title('all')
        common_layout()
        plt.savefig(f"role_instr_{split_status}_{type_aggregation}.pdf", format='pdf', bbox_inches='tight')


    elif type_aggregation == 'composer':
        # Single catplot, aggregated by composer, for all the corpus
        y_df = df.groupby(['composer', 'role', 'instrument'], sort=False, as_index=False).agg(sum)
        y_df['count'] /= y_df['length']
        l_df = [y_df.loc[y_df['composer'] == composer] for composer in LIST_COMPOSERS]
        ordered_y_df = pd.concat(l_df)
        l_df = [ordered_y_df.loc[ordered_y_df['role'] == role] for role in POSSIBLE_ROLES]
        ordered_y_df = pd.concat(l_df)
        ax = make_stack_catplot(df=ordered_y_df, xAxis='role', xCat='composer', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm, THIS_POSSIBLE_ROLES=THIS_POSSIBLE_ROLES, THIS_CAT_LIST=LIST_COMPOSERS)
        plt.title('composer')
        common_layout()
        plt.savefig(f"role_instr_{split_status}_{type_aggregation}.pdf", format='pdf', bbox_inches='tight')



    elif type_aggregation == 'multiplepiece':
        # Multiple catplot, aggregated by piece, for all the composers
        for composer in LIST_COMPOSERS:
            y_df = df.loc[df['composer'] == composer].copy()
            y_df['count'] = y_df['count'] / y_df['length']
            stack_catplot(x='role', y='count', cat='piece', stack='instrument', data=y_df, palette=instr_to_color, THIS_POSSIBLE_ROLES=THIS_POSSIBLE_ROLES)
            plt.title(composer)
            common_layout()
            plt.savefig(f"role_instr_{split_status}_{type_aggregation}_{composer}.pdf", format='pdf', bbox_inches='tight')


    elif type_aggregation == 'piece':
        # Multiple stacked_plot for all pieces
        for piece in LIST_PIECES:
            y_df = df.loc[df['piece'] == piece].copy()
            y_df['count'] = y_df['count'] / y_df['length']
            ax = make_stacked_plot(y_df, xAxis='role', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm)
            handle_presentation_global(ax)
            plt.title(piece)
            common_layout()
            plt.savefig(f"role_instr_{split_status}_{type_aggregation}_{piece}.pdf", format='pdf', bbox_inches='tight')

    elif type_aggregation == 'role_solo':
        # Multiple stacked_plot for all roles
        for role in THIS_POSSIBLE_ROLES:
            y_df = df.loc[df['role'] == role].copy()
            y_df['count'] = y_df['count'] / y_df['length']
            y_df = y_df.sort_values(by=['composer', 'piece'], ascending=[False, True])
            ax = make_stacked_plot(df=y_df, xAxis='piece', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm)
            handle_presentation_simple_stacked_composer_split(ax, ticks_type="composer")
            plt.title(role)
            plt.savefig(f"role_instr_{split_status}_{type_aggregation}_{role}.pdf", format='pdf', bbox_inches='tight')


    
    return df


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# @timer
def update_dict_instr_role(instruments_to_role, score_blend, split_role):
    """ Updates the dictionnary containing roles by instrument by piece
    """
    for segment in score_blend.get_list_segments():
        tot_bars = segment.get_length()
        for layer in segment.get_list_layers():
            if isinstance(layer, BlendLayer):
                # Group role by mel/rhythm/harm
                if split_role:
                    thir_role = str(layer.get_role().get_only_role())
                else:
                    this_real_role = str(layer.get_role().get_only_role())
                    if this_real_role in ["RHYTHM", "REPEAT_NOTE", "OSCILLATION", "BATTERIE", "ARPEGGIO", "SCALE"]:
                        thir_role = "RHYTHM"
                    else:
                        thir_role = this_real_role

                this_role_list = [thir_role] * tot_bars
                for instr in layer.ensemble.list_components:
                    instruments_to_role[CYPHER_INSTRUMENT[instr.get_only_instr()]] += this_role_list
            
            if isinstance(layer, QRLayer):
                this_role_list = ["QR"] * tot_bars
                for phrase in layer.list_phrases:
                    for internal_layer in phrase.list_qr_internal_layers:
                        for instr in internal_layer.internal_ensemble.list_components:
                            instruments_to_role[CYPHER_INSTRUMENT[instr.get_only_instr()]] += this_role_list



def stats_roles_per_instrument(type_plot, split_role=True, norm=False, type_aggregation=None, dev=False):
    """Runs analysis instrument/role

    Parameters
    ----------
    type_plot : str
        Type of plot : 'InstrX' or 'RoleX'
    split_role : bool
        Aggregate roles or not
    norm : bool
        Normalize plots or not
    type_aggregation : str
        Type of aggregation for RoleX plots : 'all', 'composer', 'multipiece', 'piece', 'role_solo'
    dev : bool
        Returns DataFrame if True
    """
    piece_to_data = dict()
    for i, piece_row in df_pieces.iterrows():
        filename = piece_row['annotation']
        print(filename, "processing...")

        annot_score = to_score(filename)
        score_blend = split_to_blend_layers(annot_score, DISPLAY=False)
        make_simple_ensembles(score_blend)
        if piece_row['piece'] in ['mozart34', 'mozart35']: 
            alter_score_annot(score_blend, "add_cb")

        instruments_to_role = dict((k, []) for k in DECYPHER_MAIN_INSTRUMENT)
        update_dict_instr_role(instruments_to_role, score_blend, split_role)
        piece_to_data[piece_row['piece']] = {
            'instruments_to_role' : instruments_to_role,
            'length' : score_blend.get_length()
        }

    if type_plot == "InstrX":
        plot_instr_role(piece_to_data, split_role)
    elif type_plot == "RoleX":
        if dev:
            return plot_role_instr(piece_to_data, norm, split_role, type_aggregation)
        else:
            plot_role_instr(piece_to_data, norm, split_role, type_aggregation)







# ==============================================================================
# =================== STATS LAYER PER BAR ======================================
# ==============================================================================

def plot_layers_per_bar_piece(dict_piece_to_layer_per_bar, list_n_layers, type_aggr):
    """Plots stuff to show number of layers per bar
    aggregated by corpus, composer or piece
    """
    
    
    dict_df = {"composer" : [], "piece" : [], "length" : [], "n_layers" : [], "count" : []}

    for piece, piece_data in dict_piece_to_layer_per_bar.items():
        total_length = len(list_n_layers)
        dict_df['composer'] += [piece.split('_')[0]] * total_length
        dict_df['piece'] += [piece] * total_length
        dict_df['length'] += [piece_data['length']] * total_length
        dict_df['n_layers'] += list_n_layers.tolist()
        dict_df['count'] += piece_data['count']

    df = pd.DataFrame(dict_df)

    if type_aggr == 'all':
        y_df = df.groupby('n_layers').agg(sum)
        plt.pie(y_df['count'], labels=list_n_layers, autopct='%1.1f%%')
        plt.savefig(f"layer_per_bar_{type_aggr}.pdf", format='pdf', bbox_inches='tight')


    elif type_aggr == 'composer':
        y_df = df.groupby(['composer', 'n_layers'], sort=False, as_index=False).agg(sum)
        y_df['count'] /= y_df['length']
        sns.barplot(x='n_layers', y='count', hue='composer', data=y_df)
        plt.savefig(f"layer_per_bar_{type_aggr}.pdf", format='pdf', bbox_inches='tight')
    

    elif type_aggr == 'multiplepiece':
        for composer in LIST_COMPOSERS:
            y_df = df.loc[df['composer'] == composer].copy()
            y_df['count'] = y_df['count'] / y_df['length']
            ax = make_stacked_plot(y_df, xAxis='piece', yAxis='count', hueAxis='n_layers', norm=False)
            plt.title(composer)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"layer_per_bar_{type_aggr}_{composer}.pdf", format='pdf', bbox_inches='tight')


    elif type_aggr == 'allpiece':
        df['count'] /= df['length']
        ax = make_stacked_plot(df, xAxis='piece', yAxis='count', hueAxis='n_layers', norm=False)
        handle_presentation_simple_stacked_composer_split(ax, ticks_type='all')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"layer_per_bar_{type_aggr}.pdf", format='pdf', bbox_inches='tight')


def update_count_layers_piece(dict_piece_to_layer_per_bar, score_blend, piece_name):
    """Updates dictionnary containing number of layers per bar
    """
    list_n_layers = np.arange(10)
    dict_piece_to_layer_per_bar[piece_name] = {
        'count' : [0 for _ in list_n_layers],
        'length' : score_blend.get_length()
    }
    for segment in score_blend.get_list_segments():
        tot_bars = segment.get_length()
        n_layers = len(segment.get_list_layers())
        dict_piece_to_layer_per_bar[piece_name]['count'][n_layers] += tot_bars



def stats_layer_per_bar(type_aggr):
    """Runs analysis on the number of layers per bar

    Parameters
    ----------
    type_aggr : str
        Type of aggregation : 'all', 'composer', 'multipiece', 'allpiece'
    """
    list_n_layers = np.arange(10)
    dict_piece_to_layer_per_bar = dict()
    for path, subdirs, files in os.walk("annotations"):
        if ("a_finir" in path) or ("pending" in path):
            continue
        for name in files:
            if not(".orchnot" in name):
                continue
            filename = os.path.join(path, name)
            piece_name = '_'.join(name.split('_')[0:2])
            print(filename, "processing...")
            annot_score = to_score(filename)
            score_blend = split_to_blend_layers(annot_score, DISPLAY=False)
            update_count_layers_piece(dict_piece_to_layer_per_bar, score_blend, piece_name)


    plot_layers_per_bar_piece(dict_piece_to_layer_per_bar, list_n_layers, type_aggr)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# ==============================================================================
# ==============================================================================
# ==============================================================================



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--statsRoleInstr", required=False)
    parser.add_argument("--statsSameLayer", required=False)
    parser.add_argument("--statsRoleCorpus", required=False, action='store_true')
    parser.add_argument("--statsAssocSameLayer", required=False, action='store_true')
    parser.add_argument("--statsLayerPerBar", required=False)
    args = parser.parse_args()


    if args.statsRoleInstr in ["RoleX", "InstrX"]:
        stats_roles_per_instrument(args.statsRoleInstr)
    elif args.statsSameLayer in ["global", "concurrent"]:
        stats_same_layer(args.statsSameLayer)
    elif args.statsRoleCorpus:
        stats_count_roles_in_dataset()
    elif args.statsAssocSameLayer:
        stats_association_in_same_layer()
    elif args.statsLayerPerBar in ["all", "composer", "piece"]:
        stats_layer_per_bar(args.statsLayerPerBar)
    else:
        parser.print_help()

