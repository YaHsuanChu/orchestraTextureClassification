"""
Declaration of classes :
- Instrument
- Role 
- Relation (Enum)
"""

from enum import Enum

# =============================================================
# ========== ENUMERATIONS =====================================
# =============================================================

# ============== INSTRUMENT ==================
class Instrument:
    divisi = -1
    name = ""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        # return self.name
        if self.divisi == -1:
            return self.name
        else:
            return self.name + "_" + str(self.divisi)
    
    def __repr__(self):
        return str(self)

    def set_divisi(self, divisi):
        self.divisi = divisi

    def get_only_instr(self):
        return self.name

    def get_divisi(self):
        return self.divisi

    def __eq__(self, o):
        return (self.name == o.name) and (self.divisi == o.divisi)

    def __hash__(self):
        return hash(self.__str__())



CYPHER_INSTRUMENT = {
    # "TUTTI" : "TUTTI",
    "FLUTE" : "Fl",
    "OBOE" : "Ob",
    "CLARINET" : "Cl",
    "FAGOTT" : "Fg",
    "HORN" : "Hrn",
    "TRUMPET" : "Trp",
    # "TROMBONE" : "Trb",
    "TIMPANI" : "Timp",
    "VIOLIN1" : "Vln1",
    "VIOLIN2" : "Vln2",
    "VIOLA" : "Vla",
    "CELLO" : "Vc",
    "CONTREBASS" : "Cb",
}


DECYPHER_MAIN_INSTRUMENT = dict((v, k) for k,v in CYPHER_INSTRUMENT.items())
DECYPHER_INSTRUMENT = dict((v, k) for k,v in CYPHER_INSTRUMENT.items())
# Add special names
SPECIAL_INSTR = {
    'HrnD' : "HORN",
    'HrnG' : "HORN",
    'HrnB' : "HORN",
    "Vln1a" : "VIOLIN1",
    "Vln1b" : "VIOLIN1",
    "HrnF" : "HORN",
    "HrnEbIII" : "HORN",
    "HrnEbII" : "HORN",
    "HrnEbI" : "HORN"
}
for k, v in SPECIAL_INSTR.items():
    DECYPHER_INSTRUMENT[k] = v

STR_TO_INSTRUMENT = dict((k, Instrument(v)) for k,v in DECYPHER_INSTRUMENT.items())

def string_to_instrument(input_str):
    """
    Converts a coded string to an Instrument object

    Parameters
    ----------
    input_str : str
        Coded instrument's name
    
    Returns
    -------
        Corresponding Instrument object
    """
    try:
        # Handle divisi here (only one digit, not a 10-part divis...)
        if input_str[-1].isdigit():
            if input_str in ['Vln1', 'Vln2']:
                this_instr_str = DECYPHER_INSTRUMENT[input_str] # To check if the instrument exists
                return Instrument(this_instr_str)
            else:
                this_instr_str = DECYPHER_INSTRUMENT[input_str[:-1]] # To check if the instrument exists
                this_instr = Instrument(this_instr_str)
                this_instr.set_divisi(int(input_str[-1]))
                return this_instr
        
        # If it is a normal instrument
        else:
            this_instr_str = DECYPHER_INSTRUMENT[input_str] # To check if the instrument exists
            return Instrument(this_instr_str)
    except KeyError:
        raise IndexError(f"Syntax error for instrument : *{input_str}*")


# ============== ROLE ==================

class Role:
    name = ""
    quantif = ""
    ALLOWED_QUANTIF = ["1", "2", "4", "8", "16", "32", "c", "f"]

    def __init__(self, name):
        self.name = name

    def __str__(self):
        # return self.name
        if self.quantif == "":
            return str(self.name)
        else:
            return str(self.name) + "_" + str(self.quantif)
    
    def __repr__(self):
        return str(self)

    def set_quantif(self, quantif):
        if quantif not in self.ALLOWED_QUANTIF:
            raise SyntaxError(f"Syntax error for quantification of role : *{self.name}_{quantif}*")
        self.quantif = quantif

    def get_only_role(self):
        return self.name

    def get_role_three_bool(self):
        is_mel = (self.name=="MAIN_MEL" or self.name=="DECORATIVE_MEL")
        is_rhythm = (self.name=="RHYTHM" or self.name=="REPEAT_NOTE" or self.name=="OSCILLATION"\
                        or self.name=="BATTERIE" or self.name=="ARPEGGIO" or self.name=="SCALE"\
                        or self.name=="DECORATIVE_MEL" or self.name=="SPARSE")
        is_harm = (self.name=="HARM" or self.name=="SPARSE")
        return (is_mel, is_rhythm, is_harm)

    def get_quantif(self):
        return self.quantif


CYPHER_ROLE = {
    "MAIN_MEL" :"mel",
    "RHYTHM" :"rhythm",
    "REPEAT_NOTE" :"repeat_note",
    "OSCILLATION" :"osc",
    "BATTERIE" : "batt",
    "ARPEGGIO" :"arp",
    "SCALE" :"scale",
    "HARM" :"harm",
    "DECORATIVE_MEL" :"decmel",
    "SPARSE" : "sparse",
    "None" : "",
}


DECYPHER_ROLE = dict((v,k) for k,v in CYPHER_ROLE.items())

STR_TO_ROLE = dict((k, Role(str(v))) for k, v in DECYPHER_ROLE.items())

def string_to_role(input_str):
    """
    Converts a coded string to an Role object

    Parameters
    ----------
    input_str : str
        Coded role's name
    
    Returns
    -------
        Corresponding Role object
    """
    try:
        if input_str == "":
            return Role(DECYPHER_ROLE[input_str])
        
        # Handle canon/fugue mel
        elif input_str[0:3] == "mel" and len(input_str) == 4:
            this_role_str = DECYPHER_ROLE[input_str[0:3]] # To check if the role exists
            this_role = Role(this_role_str)
            this_role.set_quantif(input_str[-1])
            return this_role
        
        # Handle quantification here
        elif input_str[-1].isdigit():
            if input_str[-2].isdigit():
                this_role_str = DECYPHER_ROLE[input_str[:-2]] # To check if the role exists
                this_quantif = input_str[-2:]
            else:
                this_role_str = DECYPHER_ROLE[input_str[:-1]] # To check if the role exists
                this_quantif = input_str[-1]

            this_role = Role(this_role_str)
            this_role.set_quantif(this_quantif)
            return this_role

        # If it is a normal role
        else:
            this_role_str = DECYPHER_ROLE[input_str] # To check if the role exists
            this_role = Role(this_role_str)
            return this_role
    
    except KeyError:
        raise IndexError(f"Syntax error for role : *{input_str}*")



# ============== RELATION ==================
class Relation(Enum):
    UNISSON = 0
    PARALLEL = 1
    HOMORHYTHM = 2
    REST = 3
    NONE = 4

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)

    
STR_TO_RELATION = {
    "u" : Relation.UNISSON,
    "p" : Relation.PARALLEL,
    "h" : Relation.HOMORHYTHM,
    "0" : Relation.REST,
    "" : Relation.NONE,
}

CYPHER_RELATION = {
    "UNISSON" : "u",
    "PARALLEL" : "p",
    "HOMORHYTHM" : "h",
    "REST" : "0",
    "NONE" : "",
}

DECYPHER_RELATION = dict((v,k) for k,v in CYPHER_RELATION.items())


def string_to_relation(input_str):
    """
    Converts a coded string to a Relation object

    Parameters
    ----------
    input_str : str
        Coded relation's name
    
    Returns
    -------
        Corresponding Relation object
    """
    try:
        return STR_TO_RELATION[input_str]
    except KeyError:
        raise IndexError(f"Syntax error for relation : *{input_str}*")
