import numerox as nx

TOURNAMENT_NAMES = ['bernie', 'elizabeth', 'jordan', 'ken', 'charles']


def tournament_int(tournament_int_or_str):
    "Convert tournament int or str to int"
    if nx.isstring(tournament_int_or_str):
        return tournament_str2int(tournament_int_or_str)
    elif nx.isint(tournament_int_or_str):
        if tournament_int_or_str not in (1, 2, 3, 4, 5):
            raise ValueError('tournament int must be between 1 and 5')
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
    for t in range(1, 6):
        yield t, tournament_int2str(t)


def tournament_int2str(tournament_int):
    "Convert tournament integer to string name"
    if tournament_int < 1:
        raise ValueError("`tournament_int` must be greater than 0")
    if tournament_int > 5:
        raise ValueError("`tournament_int` must be less than 6")
    return TOURNAMENT_NAMES[tournament_int - 1]


def tournament_str2int(tournament_str):
    "Convert tournament name (as str) to tournament integer"
    if tournament_str not in TOURNAMENT_NAMES:
        raise ValueError('`tournament_str` name not recognized')
    return TOURNAMENT_NAMES.index(tournament_str) + 1
