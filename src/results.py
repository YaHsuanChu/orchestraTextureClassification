"""
Specific results :
- Melodic winds evolution
"""

from .stats import *

family_to_instr = {
    "woodwinds" : ['Fl', 'Ob', 'Cl', 'Fg'],
    "brass" : ['Hrn', 'Trp', 'Timp'],
    "winds" : ['Fl', 'Ob', 'Cl', 'Fg', 'Hrn', 'Trp', 'Timp'],
    "strings" : ['Vln1', 'Vln2', 'Vla', 'Vc', 'Cb']
}



def winds_evolution(select, confidence=95, dev=False, ticks_type="composer", ignore_tutti=False):
    """Plots the evolution of winds parts through composer
    
    Parameters
    ----------
    select : str
        What to select : 'all' || 'mel'
    confidence : float
        Confidence interval to compute
    dev : bool
        To improve the figure layout
    ticks_type : string
        Which type of xticks_labels in stacked_plot : 'composer' || 'all'
    ignore_tuttis : bool
        Ignore TUTTI instruments

    Returns
    -------
        Figure and Axes if dev
    """
    df = stats_roles_per_instrument("RoleX", split_role=False, norm=False, type_aggregation=None, dev=True)
    
    if select == 'mel':
        y_df = df.loc[df['role'] == 'MAIN_MEL'].copy()
    elif select == 'all':
        y_df = df.copy()

    # Plot piece grouped by composer to count
    y_df['count'] = y_df['count'] / y_df['length']
    y_df = y_df.sort_values(by=['composer', 'piece'], ascending=[False, True])

    if ignore_tutti:
        y_df = y_df.loc[y_df['instrument'] != 'TUTTI']
        this_palette = instr_to_color[1:]
    else:
        this_palette = instr_to_color

    fig_pieces = plt.figure(figsize=(10,6))
    ax_pieces = fig_pieces.add_subplot(111)
    make_stacked_plot(df=y_df, xAxis='piece', yAxis='count', hueAxis='instrument', palette=this_palette, norm=True, ax=ax_pieces)
    handle_presentation_simple_stacked_composer_split(ax_pieces, ticks_type=ticks_type)
    plt.title(select)
    plt.savefig(f"winds_evolution_{select}_pieces.pdf", format='pdf')

    y_df_norm = y_df.copy()
    sum_all = y_df_norm.groupby('piece').agg(sum).to_dict()['count']
    for index, row in y_df_norm.iterrows():
        if sum_all[row['piece']] != 0:
            y_df_norm.loc[index, 'count'] = row['count'] / sum_all[row['piece']]
        else:
            y_df_norm.loc[index, 'count'] = 0

    for composer in LIST_COMPOSERS:
        y_df_composer = y_df_norm.loc[y_df_norm['composer'] == composer].loc[y_df_norm['instrument'].isin(family_to_instr['winds'])].copy()
        y_df_composer_grouped = y_df_composer.groupby('piece').agg(sum)
        print(y_df_composer_grouped)
        mean, down, up, h = mean_confidence_interval(y_df_composer_grouped['count'], confidence=confidence/100)
        print(f"{down} < {mean} < {up} (+/- {h})")
        print()



    fig_stats = plt.figure()
    ax_stats = fig_stats.add_subplot(111)
    y_df_winds = y_df_norm.loc[y_df_norm['instrument'].isin(family_to_instr['winds'])].copy()
    y_df_winds = y_df_winds.groupby(['composer', 'piece'], sort=False, as_index=False).agg(sum)
    sns.barplot(x='composer', y='count', data=y_df_winds, ci=confidence, capsize=.1, ax=ax_stats) # si=95 pour l'intervalle a 95%
    plt.grid()
    plt.title(select)
    if dev:
        return fig_pieces, ax_pieces
    else:
        plt.savefig(f"winds_evolution_{select}_stats_{confidence}.pdf", format='pdf', bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--windsMelody", required=False, action='store_true')
    parser.add_argument("--windsAll", required=False, action='store_true')

    args = parser.parse_args()

    if args.windsMelody:
        winds_evolution(select='mel') 
    elif args.windsAll:
        winds_evolution(select='all') 