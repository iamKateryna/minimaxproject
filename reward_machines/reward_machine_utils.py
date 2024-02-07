def evaluate_dnf(formula,true_props):
    """
    Evaluates 'formula' assuming 'true_props' are the only true propositions and the rest are false. 
    e.g. evaluate_dnf("a&b|!c&d","d") returns True 
    """
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not evaluate_dnf(f,true_props):
                return False
        return True
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if evaluate_dnf(f,true_props):
                return True
        return False
    # NOT
    if formula.startswith("!"):
        return not evaluate_dnf(formula[1:], true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props
