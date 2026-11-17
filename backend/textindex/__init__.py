"""
Modulo textual del motor BD2 (Proyecto 2, Parte 3).

Componentes:
    preprocessing  : pipeline lexico (normalizacion, tokens, stoplist, stemming)
    query          : parser de la cadena de MATCH (AND, OR, AND NOT, parentesis)
    postings       : mezclas lineales sobre listas ordenadas (AND, OR, AND-NOT)
    inverted_index : indice invertido persistido en disco (vocab.dic,
                     postings.inv, docs.dat), construccion SPIMI y CosineScore
    fulltext_scan  : linea base sin indice (Full Text Scan) con el mismo scoring
"""
