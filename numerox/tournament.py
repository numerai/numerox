import numerox as nx

TOURNAMENT_NAMES = {'bernie': 1, 'elizabeth': 2, 'jordan': 3, 'ken': 4,
                    'charles': 5, 'frank': 6, 'hillary': 7}


def tournament_int(tournament_int_or_str):
    "Convert tournament int or str to int"
    if nx.isstring(tournament_int_or_str):
        return tournament_str2int(tournament_int_or_str)
    elif nx.isint(tournament_int_or_str):
        if tournament_int_or_str not in TOURNAMENT_NAMES.values():
            raise ValueError("`tournament_int_ot_str` not recognized")
        return tournament_int_or_str
    raise ValueError('input must be a str or int')


def tournament_str(tournament_int_or_str):
    "Convert tournament int or str to str"
    if nx.isstring(tournament_int_or_str):
        if tournament_int_or_str not in TOURNAMENT_NAMES:
            raise ValueError('tournament name is unknown')
        return tournament_int_or_str
    elif nx.isint(tournament_int_or_str):
        return tournament_int2str(tournament_int_or_str)
    raise ValueError('input must be a str or int')


def tournament_all(as_str=True):
    "List of all tournaments as strings (default) or integers."
    tournaments = []
    if as_str:
        for number, name in tournament_iter():
            tournaments.append(name)
    else:
        for number, name in tournament_iter():
            tournaments.append(number)
    return tournaments


def tournament_iter():
    "Iterate, in order, through tournaments yielding tuple of (int, str)"
    numbers = TOURNAMENT_NAMES.values()
    numbers.sort()
    for t in numbers:
        yield t, tournament_int2str(t)


def tournament_int2str(tournament_int):
    "Convert tournament integer to string name"
    if tournament_int not in TOURNAMENT_NAMES.values():
        raise ValueError("`tournament_int` not recognized")
    for name in TOURNAMENT_NAMES:
        if TOURNAMENT_NAMES[name] == tournament_int:
            return name
    raise RuntimeError("Did not find tournament name")


def tournament_str2int(tournament_str):
    "Convert tournament name (as str) to tournament integer"
    if tournament_str not in TOURNAMENT_NAMES:
        raise ValueError('`tournament_str` name not recognized')
    return TOURNAMENT_NAMES[tournament_str]


def tournament_count():
    "Returns the number of tournaments as an integer"
    return len(TOURNAMENT_NAMES)
