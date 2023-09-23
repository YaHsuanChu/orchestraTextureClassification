"""
Stats on
- the pieces when divided in fixed-length slices
- the corpus when divided in fixed-length slices

- the pieces when divided in sonata form sections
- the corpus when divided in sonata form sections
- similarities between sections

- textural content of developpement
"""
from .processing import *
from .parser import *
from .stats import *

import itertools 

# ==========================================================================================
# ================= FIXED-LENGTH SLICES ====================================================
# ==========================================================================================

def make_dataframe_fixed_length(score_bar_split):
    """Makes pandas.DataFrame for a bar-split ScoreAnnot

    Parameters
    ----------
    score_bar_split : ScoreAnnot
        ScoreAnnot to be analyzed

    Returns
    -------
        pandas.DataFrame containing relevant stuff
    """
    def update_dict_df(dict_df, position, instrument, role, layer_id, tot_num_layers):
        dict_df['position'].append(position)
        dict_df['instrument'].append(CYPHER_INSTRUMENT[str(instrument.get_only_instr())])
        dict_df['role'].append(str(role))
        dict_df['layer_id'].append(str(layer_id))
        dict_df['tot_num_layers'].append(tot_num_layers)

    dict_df = {
        'position' : [],
        'layer_id' : [],
        'tot_num_layers' : [],
        'instrument' : [],
        'role' : []
    }
    

    for index_bar, segment in enumerate(score_bar_split.get_list_segments()):
        n_bar = index_bar # + 1
        # print(make_segment(segment))
        tot_num_layers = len(segment.get_list_layers())
        for index_layer, layer in enumerate(segment.get_list_layers()):
            n_layer = index_layer # + 1
            layer_id = f"{n_bar}_{n_layer}"
            if isinstance(layer, BlendLayer): # A priori pas de +
                role = layer.get_role().get_only_role()
                for instr in layer.get_ensemble().get_list_components():
                    update_dict_df(dict_df, position=n_bar, instrument=instr, role=role, layer_id=layer_id, tot_num_layers=tot_num_layers)
            elif isinstance(layer, QRLayer):
                for phrase in layer.get_list_phrases():
                    for internal_layer in phrase.get_list_qr_internal_layers():
                        for instr in internal_layer.get_internal_ensemble().get_list_components():
                            update_dict_df(dict_df, position=n_bar, instrument=instr, role="QR", layer_id=layer_id, tot_num_layers=tot_num_layers)

    
    # print(dict_df)
    df = pd.DataFrame(dict_df)
    return df


def merge_into_chunks_plot(df, len_chunks, split_role, norm, role_times_instr, title_file):
    """Merges DataFrames into fixed-length chunks

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame produced by make_dataframe_fixed_length
    len_chunks : int
        Length of the chunk
    split_role : bool
        Aggregation of rhythmic roles
    norm : bool
        Normalization of the stacked plot
    role_times_instr : bool
        Count of roles units or roles times instruments
    title_file : str
        Title of the plot
    """

    if split_role:
        THIS_POSSIBLE_ROLES = POSSIBLE_ROLES
        split_status = "split"
        this_role_palette = role_to_color
    else:
        THIS_POSSIBLE_ROLES = ["MAIN_MEL", "RHYTHM", "HARM", "DECORATIVE_MEL", "SPARSE", "None", "QR"]
        split_status = "aggr"
        this_role_palette = role_split_to_color
    

    def update_parameter_to_count(instr_to_count, role_to_count, df_considered_rows, beg_chunk):
        # --- Update instrument ---
        instr_count_list = [0] * len(LIST_POSSIBLE_INSTRUMENTS)
        values_instr, count_instr = np.unique(np.array(df_considered_rows['instrument']), return_counts=True)
        for i, instr in enumerate(values_instr):
            instr_count_list[LIST_POSSIBLE_INSTRUMENTS.index(instr)] = count_instr[i]
        instr_to_count['section'] += [str(beg_chunk)] * len(LIST_POSSIBLE_INSTRUMENTS)
        instr_to_count['instrument'] += LIST_POSSIBLE_INSTRUMENTS
        instr_to_count['count'] += instr_count_list
        
        # --- Update role ---
        role_count_list = [0] * len(THIS_POSSIBLE_ROLES)
        if role_times_instr:
            values_role, count_role = np.unique(np.array(df_considered_rows['role']), return_counts=True)
        else:
            grouped_df_considered_rows = df_considered_rows.groupby(['layer_id', 'role'], sort=False, as_index=False).agg('mean')
            values_role, count_role = np.unique(np.array(grouped_df_considered_rows['role']), return_counts=True)
            
        for i, real_role in enumerate(values_role):
            if split_role:
                this_role = real_role
            else:
                if real_role in ["RHYTHM", "REPEAT_NOTE", "OSCILLATION", "BATTERIE", "ARPEGGIO", "SCALE"]:
                    this_role = "RHYTHM"
                else:
                    this_role = real_role
            role_count_list[THIS_POSSIBLE_ROLES.index(this_role)] += count_role[i]
        
        role_to_count['section'] += [str(beg_chunk)] * len(THIS_POSSIBLE_ROLES)
        role_to_count['role'] += THIS_POSSIBLE_ROLES
        role_to_count['count'] += role_count_list
        

    tot_bars = df['position'].iloc[-1]
    chunks_beg_list = np.arange(df['position'].iloc[0], len_chunks*round(1+tot_bars/len_chunks), len_chunks)

    instr_to_count = {'section' : [], 'instrument' : [], 'count' : []}
    role_to_count = {'section' : [], 'role' : [], 'count' : []}

    
    for beg_chunk in chunks_beg_list:
        end_chunk = beg_chunk + len_chunks
        df_considered_rows = df.loc[(beg_chunk <= df['position']) & (df['position'] < end_chunk)].copy()
        update_parameter_to_count(instr_to_count, role_to_count, df_considered_rows, beg_chunk)

    df_role_chunked = pd.DataFrame(role_to_count)
    df_instr_chunked = pd.DataFrame(instr_to_count)


    # Plot
    fig = plt.figure(figsize=(15,10))
    ax1 = plt.subplot(211)
    make_stacked_plot(df_instr_chunked, xAxis='section', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm, ax=ax1)
    plt.title(title_file)
    plt.xticks(rotation=90)
    ax2 = plt.subplot(212, sharex = ax1)
    ax_role = make_stacked_plot(df_role_chunked, xAxis='section', yAxis='count', hueAxis='role', palette=this_role_palette, norm=norm, ax=ax2)
    plt.xticks(rotation=90)
    plt.savefig(f"timeline_chunk_{split_status}_{title_file}.pdf", format='pdf', bbox_inches='tight')



def instr_role_per_chunk(filename, len_chunks=20, split_role=True, norm=True, role_times_instr=True, title_file=None):
    """Analysis of one piece split in fixed-length chunks

    Parameters
    ----------
    filename : str
        Name of the .orchnot
    len_chunks : int
        Length of the chunk
    split_role : bool
        Aggregation of rhythmic roles
    norm : bool
        Normalization of the stacked plot
    role_times_instr : bool
        Count of roles units or roles times instruments
    title_file : str
        Title of the plot
    """
    if title_file is None:
        title_file = filename
    print(filename, "processing...")
    score = to_score(filename)
    score_bar_split = split_to_bar_layers(score)
    df = make_dataframe_fixed_length(score_bar_split)
    merge_into_chunks_plot(df, len_chunks, split_role, norm, role_times_instr, title_file.replace('/', '_'))




# ======================================================================================================================
# ===================== STRUCTURE SONATA FORM ==========================================================================
# ======================================================================================================================




def make_dataframe_sonata(score_bar_split):
    """Make panda.DataFrame for a section-split ScoreAnnot

    Parameters
    ----------
    score_bar_split : ScoreAnnot
        ScoreAnnot to be analyzed

    Returns
    -------
        pandas.DataFrame containing relevant stuff
    """

    def update_dict_df(dict_df, position, instrument, role, layer_id, tot_num_layers):
        dict_df['label'].append(position)
        dict_df['instrument'].append(CYPHER_INSTRUMENT[str(instrument.get_only_instr())])
        dict_df['role'].append(str(role))
        dict_df['layer_id'].append(str(layer_id))
        dict_df['tot_num_layers'].append(tot_num_layers)

    dict_df = {
        'label' : [],
        'layer_id' : [],
        'tot_num_layers' : [],
        'instrument' : [],
        'role' : []
    }
    
    for index_bar, segment in enumerate(score_bar_split.get_list_segments()):
        n_bar = index_bar # + 1
        this_label = segment.get_label()
        # print(make_segment(segment))
        tot_num_layers = len(segment.get_list_layers())
        for index_layer, layer in enumerate(segment.get_list_layers()):
            n_layer = index_layer # + 1
            layer_id = f"{n_bar}_{n_layer}"
            if isinstance(layer, BlendLayer): # A priori pas de +
                role = layer.get_role().get_only_role()
                for instr in layer.get_ensemble().get_list_components():
                    update_dict_df(dict_df, position=this_label, instrument=instr, role=role, layer_id=layer_id, tot_num_layers=tot_num_layers)
            elif isinstance(layer, QRLayer):
                for phrase in layer.get_list_phrases():
                    for internal_layer in phrase.get_list_qr_internal_layers():
                        for instr in internal_layer.get_internal_ensemble().get_list_components():
                            update_dict_df(dict_df, position=this_label, instrument=instr, role="QR", layer_id=layer_id, tot_num_layers=tot_num_layers)

    
    # print(dict_df)
    df = pd.DataFrame(dict_df)
    return df



def plot_structure_role_instr(df, split_role, role_times_instr, norm, title_file, show):
    """Plots textural and instrumentation evolution trhough sonata sections

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame produced by make_dataframe_sonata
    split_role : bool
        Aggregation of rhythmic roles
    role_times_instr : bool
        Count of roles units or roles times instruments
    norm : bool
        Normalization of the stacked plot
    title_file : str
        Title of the plot
    show : bool
        Show the plot or not

    Returns
    -------
        pandas.DataFrames containing textural and instrumentation stuff
    """
    if split_role:
        THIS_POSSIBLE_ROLES = POSSIBLE_ROLES
        split_status = "split"
        this_role_palette = role_to_color
    else:
        THIS_POSSIBLE_ROLES = POSSIBLE_AGGR_ROLES
        split_status = "aggr"
        this_role_palette = role_split_to_color

    def update_parameter_to_count(instr_to_count, role_to_count, df_considered_rows, beg_chunk):
        # --- Update instrument ---
        instr_count_list = [0] * len(LIST_POSSIBLE_INSTRUMENTS)
        values_instr, count_instr = np.unique(np.array(df_considered_rows['instrument']), return_counts=True)
        for i, instr in enumerate(values_instr):
            instr_count_list[LIST_POSSIBLE_INSTRUMENTS.index(instr)] = count_instr[i]
        instr_to_count['section'] += [str(beg_chunk)] * len(LIST_POSSIBLE_INSTRUMENTS)
        instr_to_count['instrument'] += LIST_POSSIBLE_INSTRUMENTS
        instr_to_count['count'] += instr_count_list
        
        # --- Update role ---
        role_count_list = [0] * len(THIS_POSSIBLE_ROLES)
        if role_times_instr:
            values_role, count_role = np.unique(np.array(df_considered_rows['role']), return_counts=True)
        else:
            grouped_df_considered_rows = df_considered_rows.groupby(['layer_id', 'role'], sort=False, as_index=False).agg('mean')
            values_role, count_role = np.unique(np.array(grouped_df_considered_rows['role']), return_counts=True)
            
        for i, real_role in enumerate(values_role):
            if split_role:
                this_role = real_role
            else:
                if real_role in ["RHYTHM", "REPEAT_NOTE", "OSCILLATION", "BATTERIE", "ARPEGGIO", "SCALE"]:
                    this_role = "RHYTHM"
                else:
                    this_role = real_role
            role_count_list[THIS_POSSIBLE_ROLES.index(this_role)] += count_role[i]
        
        role_to_count['section'] += [str(beg_chunk)] * len(THIS_POSSIBLE_ROLES)
        role_to_count['role'] += THIS_POSSIBLE_ROLES
        role_to_count['count'] += role_count_list

    

    instr_to_count = {'section' : [], 'instrument' : [], 'count' : []}
    role_to_count = {'section' : [], 'role' : [], 'count' : []}

    for label in df.label.unique():
        df_considered_rows = df.loc[df['label'] == label].copy()
        update_parameter_to_count(instr_to_count, role_to_count, df_considered_rows, label)


    df_role_section = pd.DataFrame(role_to_count)
    df_instr_section= pd.DataFrame(instr_to_count)

    if show:
        fig = plt.figure(figsize=(15,10))
        ax1 = plt.subplot(211)
        make_stacked_plot(df_instr_section, xAxis='section', yAxis='count', hueAxis='instrument', palette=instr_to_color, norm=norm, ax=ax1)
        plt.title(title_file)
        ax2 = plt.subplot(212, sharex = ax1)
        ax_role = make_stacked_plot(df_role_section, xAxis='section', yAxis='count', hueAxis='role', palette=this_role_palette, norm=norm, ax=ax2)
        plt.savefig(f"timeline_sonata_form_{split_status}_{title_file}.pdf", format='pdf', bbox_inches='tight')

    return df_role_section, df_instr_section


def instr_role_sonata_form(filename, split_role=True, norm=True, role_times_instr=True, title_file=None, dev=False, show=True):
    """Analyzes textural and instrumentation evolution trhough sonata sections

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    split_role : bool
        Aggregation of rhythmic roles
    norm : bool
        Normalization of the stacked plot
    role_times_instr : bool
        Count of roles units or roles times instruments
    title_file : str
        Title of the plot
    dev : bool
        To manipulate data aftewards
    show : bool
        Show the plot or not

    Returns
    -------
        pandas.DataFrames containing textural and instrumentation stuff if dev
    """
    if title_file is None:
        title_file = filename
    print(filename, "processing...")
    score = set_label_to_score(filename=filename)
    score_bar_split = split_to_bar_layers(score)
    df = make_dataframe_sonata(score_bar_split)
    if dev:
        return plot_structure_role_instr(df, split_role=False, role_times_instr=False, norm=True, title_file=title_file.replace('/', '_'), show=show)
    else:
        plot_structure_role_instr(df, split_role=False, role_times_instr=False, norm=True, title_file=title_file.replace('/', '_'), show=show)


# ======================================================================================================================
# ===================== SIMILARITY SECTIONS ==========================================================================
# ======================================================================================================================


def make_dist_matrix(df_section, len_vectors, LIST_SECTIONS, n_sections):
    """Makes distance matrix between sections

    Parameters
    ----------
    df_section : pandas.DataFrame
        Normalized pandas.DataFrame produced by instr_role_sonata_form
    len_vectors : int
        Length of the vectors
    LIST_SECTIONS : list
        List of sections considered
    n_sections : int
        len(LIST_SECTION)

    Returns
    -------
        Distance matrix of sections textural/instrumental similarity, of size n_sections x n_sections 
    """
# Pas joli...
    dict_vectors = dict((k, np.zeros(len_vectors)) for k in LIST_SECTIONS)
    for section in dict_vectors.keys():
        considered_rows = df_section.loc[df_section['section'] == section].copy()
        if len(considered_rows['count'].tolist()) > 0:
            dict_vectors[section] = considered_rows['count'].to_numpy()
    

    dist_matrix = np.zeros((n_sections, n_sections))
    for i in range(n_sections):
        for j in range(n_sections):
            if i == j:
                dist_matrix[i,j] = float("nan")
            else:    
                vect_i = list(dict_vectors.values())[i]
                vect_j = list(dict_vectors.values())[j]
                dist_matrix[i,j] = np.linalg.norm(vect_j - vect_i)
            
    return dist_matrix

  




def measure_similarity_sections(list_df_param_section, type_simil, dev=False, ignore_sections=[], normalize=None):
    """Plots the average textural or instrumental similarity matrix of all the corpus

    Parameters
    ----------
    list_df_param_section : list
        List of pandas.DataFrames, all produced by instr_role_sonata_form
    type_simil : str
        Which type of similarity : 'instrument' || 'role'
    dev : bool
        To improve the layout of the figures
    ignore_sections : list
        Which sections to ignore

    Returns
    -------
        Figure and Axes of the similarity matrix if dev
    """

    if type_simil == 'role':
        LIST_PARAM = POSSIBLE_AGGR_ROLES # Force
    elif type_simil == 'instrument':
        LIST_PARAM = LIST_POSSIBLE_INSTRUMENTS


    LIST_SECTIONS = ["Int", "P", "S", "C", "Dev", "P'", "S'", "C'", "Cod"]

    # Remove ignore sections
    for section in ignore_sections:
        LIST_SECTIONS.remove(section)
    n_sections = len(LIST_SECTIONS)


    # Make dist matrix
    all_dist_matrix = np.zeros((len(list_df_param_section[:-1]), n_sections, n_sections))
    for i, df_param_section in enumerate(list_df_param_section[:-1]): # Ignore Beethoven 9
        # Norm
        df_param_section_norm = df_param_section.copy()
        sum_all = df_param_section_norm.groupby('section').agg(sum).to_dict()['count']
        for index, row in df_param_section_norm.iterrows():
            if sum_all[row['section']] != 0:
                df_param_section_norm.loc[index, 'count'] = row['count'] / sum_all[row['section']]
            else:
                df_param_section_norm.loc[index, 'count'] = 0
                
        this_dist_matrix = make_dist_matrix(df_param_section_norm, len(LIST_PARAM), LIST_SECTIONS, n_sections)
        all_dist_matrix[i,:,:] = this_dist_matrix

    if normalize is not None:
        dist_matrix = np.mean(all_dist_matrix, axis=0)
        max_dist_matrix = normalize # np.nanmax(dist_matrix)

        for i in range(len(all_dist_matrix)):
            all_dist_matrix[i] /= max_dist_matrix
        dist_matrix = np.mean(all_dist_matrix, axis=0)
    
    else:
        dist_matrix = np.mean(all_dist_matrix, axis=0)


    pos_p = LIST_SECTIONS.index("P")
    pos_pp = LIST_SECTIONS.index("P'")
    pos_s = LIST_SECTIONS.index("S")
    pos_sp = LIST_SECTIONS.index("S'")
    pos_c = LIST_SECTIONS.index("C")
    pos_cp = LIST_SECTIONS.index("C'")


    p_mean, p_down, p_up, h = mean_confidence_interval(all_dist_matrix[:,pos_p,pos_pp])
    print(f"Similarity(P,P') >> mean : {p_mean} (+/- {h}) \n\t\t 95% confidence : [{p_down} - {p_up}] ")
    s_mean, s_down, s_up, h = mean_confidence_interval(all_dist_matrix[:,pos_s,pos_sp])
    print(f"Similarity(S,S') >> mean : {s_mean} (+/- {h}) \n\t\t 95% confidence : [{s_down} - {s_up}] ")
    c_mean, c_down, c_up, h = mean_confidence_interval(all_dist_matrix[:,pos_c,pos_cp])
    print(f"Similarity(C,C') >> mean : {c_mean} (+/- {h}) \n\t\t 95% confidence : [{c_down} - {c_up}] ")

    # Compute mean matrix without P,S :
    list_values = []
    for j in range(dist_matrix.shape[0]):
        for i in range(dist_matrix.shape[0]):
            if j > i:
                if not((i == pos_p) and (j == pos_pp)) and not((i == pos_s) and (j == pos_sp)):
                    list_values.append(dist_matrix[i,j])
    
    other_mean, other_down, other_up, h = mean_confidence_interval(list_values)
    print(f"Average other couples >> mean : {other_mean} (+/- {h}) \n\t\t 95 % confidence : [{other_down} - {other_up}] ")

    fig, ax = plt.subplots(1,1)
    mat_plt = ax.imshow(dist_matrix, cmap='viridis_r')
    ax.set_xticks(np.arange(0, n_sections, 1))
    ax.set_yticks(np.arange(0, n_sections, 1))
    ax.set_yticklabels(LIST_SECTIONS)
    ax.set_xticklabels(LIST_SECTIONS)

    plt.xticks(rotation=45)
    plt.colorbar(mat_plt)

    if dev:
        return ax, dist_matrix
    else:
        plt.savefig(f"similarity_sections_{type_simil}.pdf", format='pdf', bbox_inches='tight')

    


def measure_similarity_one_piece(filename, type_simil='role'):
    """Plots the textural or instrumental similarity matrix of one piece

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    type_simil : str
        Which type of similarity : 'instrument' || 'role'
    """
    df_role_section, df_instr_section = instr_role_sonata_form(filename, split_role=False, norm=True, role_times_instr=True, title_file=None, dev=True)
    df_role_section['piece'] = filename
    df_instr_section['piece'] = filename


    if type_simil == 'role':
        LIST_PARAM = POSSIBLE_AGGR_ROLES # Force
        df_param_section = df_role_section
    elif type_simil == 'instrument':
        LIST_PARAM = LIST_POSSIBLE_INSTRUMENTS
        df_param_section = df_instr_section

    LIST_SECTIONS = ["Int", "P", "S", "C", "Dev", "P'", "S'", "C'", "Cod"]
    n_sections = len(LIST_SECTIONS)
    
    # Norm
    df_param_section_norm = df_param_section.copy()
    sum_all = df_param_section_norm.groupby('section').agg(sum).to_dict()['count']
    for index, row in df_param_section_norm.iterrows():
        if sum_all[row['section']] != 0:
            df_param_section_norm.loc[index, 'count'] = row['count'] / sum_all[row['section']]
        else:
            df_param_section_norm.loc[index, 'count'] = 0
            
    dist_matrix = make_dist_matrix(df_param_section_norm, len(LIST_PARAM), LIST_SECTIONS, n_sections)

    fig, ax = plt.subplots(1,1)
    mat_plt = ax.imshow(dist_matrix, cmap='viridis_r')
    ax.set_xticks(np.arange(0, n_sections, 1))
    ax.set_yticks(np.arange(0, n_sections, 1))
    ax.set_yticklabels(LIST_SECTIONS)
    ax.set_xticklabels(LIST_SECTIONS)

    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            ax.text(j,i, "{:.2f}".format(dist_matrix[i,j]), ha="center", va="center", color="k")

    plt.xticks(rotation=45)
    plt.colorbar(mat_plt)


# ======================================================================================================================
# ===================== TEXTURE IN DEVELOPPEMENT ==========================================================================
# ======================================================================================================================

def texture_developement(filename, n_chunks, title_file=None, dev=False):
    """Plots the textural content of the developpement of one piece

    Parameters
    ----------
    filename : str
        Name of the .orchnot file
    len_chunks : int
        Length of the chunks
    title_file : str
        Name of the figure
    dev: bool
        To manipulate data afterwards
    """
    print(filename, "processing...")
    score = set_label_to_score(filename=filename)
    score_bar_split = split_to_bar_layers(score)
    df = make_dataframe_sonata(score_bar_split)
    
    df_dev = df.loc[df['label'] == 'Dev'].copy()
    df_dev['bar'] = [int(k.split('_')[0]) for k in df_dev['layer_id']]

    length_dev = len(df_dev.bar.unique())
    pos_beg_chunk =  np.linspace(df_dev['bar'].iloc[0], df_dev['bar'].iloc[-1]+1, n_chunks+1, dtype=int)

    dict_position_to_normal = dict()
    for bar in df_dev['bar']:
        for i, n in enumerate(pos_beg_chunk[:-1]):
            if n <= bar < pos_beg_chunk[i+1]:
                dict_position_to_normal[bar] = i
                break

    df_dev['label'] = [dict_position_to_normal[k] for k in df_dev['bar']]


    if title_file is None:
        title_file = filename.replace('/', '_')
    
    if dev:
        return plot_structure_role_instr(df_dev, split_role=False, role_times_instr=True, norm=True, title_file=f"dev_{title_file}", show=True)
    else:
        plot_structure_role_instr(df_dev, split_role=False, role_times_instr=True, norm=True, title_file=f"dev_{title_file}", show=True)






    