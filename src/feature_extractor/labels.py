label_non_abs_anph = ["NonAbsAnph"]

label_non_abs_antd = ["NonAbsAntd"]

label_abs_anph = ["EvAnph","ProcAnph","StateAnph","CircAnph","DeverbAnph","FactAnph","SubjAnph","NegAnph","ModAnph","WhAnph","PropAnph"]

label_abs_antd = ["EvAntd","ProcAntd","StateAntd","CircAntd","DeverbAntd","FactAntd","SubjAntd","NegAntd","ModAntd","WhAntd","PropAntd"]

def get_lables():
    label_non_abs_anph_set = set(label_non_abs_anph)
    label_non_abs_antd_set = set(label_non_abs_antd)
    label_abs_anph_set = set(label_abs_anph)
    label_abs_antd_set = set(label_abs_antd)
    return label_non_abs_anph_set, label_non_abs_antd_set, label_abs_anph_set, label_abs_antd_set