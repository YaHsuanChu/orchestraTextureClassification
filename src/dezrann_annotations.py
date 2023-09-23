from src.parser import *
from src.classes import *
import json

# Function to produce .dez files from annotations

def produce_dez_file(symphony_dict,symphony_name):
    raw_segments = get_raw_segments(symphony_dict[symphony_name]['filepath'] + '/' + symphony_dict[symphony_name]['filename'])
    #raw_segments.append('[3000-4000] <ff|f|fffff> f:fake')

    dez_labels = []

    # I construct two lists that will store temporarily the dezrann label (until a new label for the part is produced)
    previous_label_by_part = []
    n_parts = symphony_dict[symphony_name]['n_parts']
    for i in range(n_parts):
        previous_label_by_part.append({ "type": "Pattern",  "start": 0, "duration": 0, "staff": 0, "tag": '' })

    current_label_by_part = []
    n_parts = symphony_dict[symphony_name]['n_parts']
    for i in range(n_parts):
        current_label_by_part.append({ "type": "Pattern",  "start": 0, "duration": 0, "staff": 0, "tag": '' })


    # I scan all the lines in the annotation file
    for l in raw_segments:
        try:
            if l[:8] == 'InstList': # If the current line is a list of instrument (new syntax)
                inst_list = read_inst_list(l)
            else: # The line is not a list of instrument, it can be in either syntaxes
                # Retrieve bar limits
                bar_limits_beg_end = re.search(r'\[[0-9-]+\]', l)
                if bar_limits_beg_end:
                    bar_limits = bar_limits_beg_end.group()
                    measure_beg, measure_end = beg_end(bar_limits[1:-1])
                    # Compute label start and duration
                    time_signatures = symphony_dict[symphony_name]['time_signatures']
                    for ts in time_signatures:
                        if measure_beg >= ts['begin'] and measure_beg <= ts['end']:
                            beat_beg = ts['beats_previous_sections'] + (measure_beg-ts['begin'])*ts['n_quarter_notes']
                            dur = (measure_end-measure_beg+1)*ts['n_quarter_notes']
                else:
                    raise SyntaxError("No suitable bar limit")

                # Get layers
                l = l.replace(bar_limits + " ", '').strip() # remove bars indication
                layers_texture = l.split()[0] # String containing the layers letter codes, <aad|dd>
                layers_list = l.split()[1:]
                layers_texture_def = ' '.join(layers_list) # String containing the descriptions of the layers
                layers_texture_hl = ''
                if '{' in layers_texture_def:
                    layers_texture_hl = '{' + layers_texture_def.split('{')[1].strip() # high level features
                    layers_texture_def = layers_texture_def.split('{')[0].strip()
                if '(' in layers_texture_def:
                    layers_texture_hl = '(' + layers_texture_def.split('(')[1].strip() # high level features
                    layers_texture_def = layers_texture_def.split('(')[0].strip()
                    

                # Write the dezrann label (complete textural description)
                dez_labels.append({ "type": "Texture", "start": beat_beg, "duration": dur, "tag": layers_texture })
                dez_labels.append({ "type": "Texture Def", "start": beat_beg, "duration": dur, "tag": layers_texture_def })
                if layers_texture_hl != '':
                    dez_labels.append({ "type": "Texture High Level", "start": beat_beg, "duration": dur, "tag": layers_texture_hl })

                # Produce the staff level annotations
                layers_codes=layers_texture.replace('<','').replace('>','').replace('|','')
                staff_dict = symphony_dict[symphony_name]['staff_dict']

                # Store the layers texture codes in lists
                layers_codes_list = []
                ii = 0
                while ii < len(layers_codes):
                    if layers_codes[ii] != '(': # this instrument is not divisi
                        curr_code = layers_codes[ii]
                        layers_codes_list.append(curr_code)
                    else: # this instrument is divisi
                        ii+=1
                        curr_code = layers_codes[ii:].split(')')[0]
                        layers_codes_list.append(curr_code)
                        ii+=1
                    ii+=len(curr_code)
                                        
                # Construct dictionary with letter-label association at the current segment
                layers_dict = dict()
                for lab in layers_list: # scan all layers in the current segment
                    if lab[0] == '(' or lab[0] == '{': # stop the conversion when we arrive at the higher level relations
                        break
                    elif lab[0] == '~':
                        letter = lab[1:].split(':')[0]
                    else:
                        letter = lab.split(':')[0]
                    layers_dict[letter] = lab

                # Construct one label for each staff
                for indx in range(len(layers_codes_list)):
                    labs = []
                    tilde = False
                    if layers_codes_list: # if list is not empty
                        tilde = True
                    for ll in layers_codes_list[indx]: # for each letter present in this instrument
                        if ll != '0':
                            labs.append(layers_dict[ll])
                            if layers_dict[ll][0] != '~': # if one of them does not start with ~
                                tilde = False
                    lab = ' '.join(labs)

                    if lab.replace('~','') == previous_label_by_part[indx]['tag'].replace('~','') and tilde and previous_label_by_part[indx]['start'] + previous_label_by_part[indx]['duration'] == beat_beg:
                        previous_label_by_part[indx]['duration'] = previous_label_by_part[indx]['duration'] + dur
                    else:
                        staff = staff_dict.get(inst_list[indx])
                        if previous_label_by_part[indx]['tag'] != '':
                            dez_labels.append(previous_label_by_part[indx])
                        previous_label_by_part[indx] = { "type": "Pattern",  "start": beat_beg, "duration": dur, "staff": staff, "tag": lab }   
                    if symphony_dict[symphony_name]['time_signatures'][-1]["end"] == measure_end: # It is the last measure
                        dez_labels.append(previous_label_by_part[indx])

        except Exception as e:
            raise SyntaxError(f"*{repr(l)}*")

    # Add sonata form structure labels
    struct_segments = get_struct_segments(symphony_dict[symphony_name]['filepath'] + '/' + symphony_dict[symphony_name]['filename'])

    next = False
    for l in struct_segments:
        if (l[0:2] == "#!"):
            lab = l.split(' ')[0][2:]
            next = True
        else:
            if l.startswith("#"):
                continue
            bar_limits_beg_end = re.search(r'\[[0-9-]+\]', l)
            if bar_limits_beg_end:
                bar_limits = bar_limits_beg_end.group()
                measure_beg, measure_end = beg_end(bar_limits[1:-1])
                # Compute label start and duration
                time_signatures = symphony_dict[symphony_name]['time_signatures']
                for ts in time_signatures:
                    if measure_beg >= ts['begin'] and measure_beg <= ts['end']:
                        beat_beg = ts['beats_previous_sections'] + (measure_beg-ts['begin'])*ts['n_quarter_notes']
                        dur = (measure_end-measure_beg+1)*ts['n_quarter_notes']
            else:
                raise SyntaxError("No suitable bar limit")
            dez_labels.append({ "type": "Structure",  "start": beat_beg, "duration": ts['n_quarter_notes'], "tag": lab })
            next = False

    dez_annot = {
        "labels" : dez_labels,
        "meta": {
            "date": "2022XXXX",
            "producer": "XXX",
            "title": "XXX",
            "layout": [
                { "filter" : { "type": "Structure" },     "style" : {"line": "bot.1" } },
                { "filter" : { "type": "Texture" },     "style" : {"line": "top.2" } },
                { "filter" : { "type": "Texture Def" }, "style" : {"line": "top.3" } },
                { "filter" : { "type": "Texture High Level" }, "style" : {"line": "bot.2" } },
            ]
        }
    }

    with open(symphony_dict[symphony_name]['filepath'] + '/' + symphony_dict[symphony_name]['filename'][:-4] + 'dez' , "w") as write_file:
        json.dump(dez_annot, write_file,indent=4, sort_keys=True)