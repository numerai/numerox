import numerox as nx

<<<<<<< HEAD
<<<<<<< HEAD
# TOURNAMENTS = [{'name': 'bernie', 'number': 1, 'active': False},
#                {'name': 'elizabeth', 'number': 2, 'active': False},
#                {'name': 'jordan', 'number': 3, 'active': False},
#                {'name': 'ken', 'number': 4, 'active': False},
#                {'name': 'charles', 'number': 5, 'active': False},
#                {'name': 'frank', 'number': 6, 'active': False},
#                {'name': 'hillary', 'number': 7, 'active': False},
TOURNAMENTS = [{'name': 'kazutsugi', 'number': 1, 'active': True}]
=======
=======
>>>>>>> 4f84aeb10a9b557f01e98ec23e2572ce14e24e59
TOURNAMENTS = [{
    'name': 'bernie',
    'number': 1,
    'active': False
}, {
    'name': 'elizabeth',
    'number': 2,
    'active': False
}, {
    'name': 'jordan',
    'number': 3,
    'active': False
}, {
    'name': 'ken',
    'number': 4,
    'active': False
}, {
    'name': 'charles',
    'number': 5,
    'active': False
}, {
    'name': 'frank',
    'number': 6,
    'active': False
}, {
    'name': 'hillary',
    'number': 7,
    'active': False
}, {
    'name': 'kazutsugi',
    'number': 8,
    'active': True
}]
<<<<<<< HEAD
>>>>>>> 4f84aeb10a9b557f01e98ec23e2572ce14e24e59
=======
>>>>>>> 4f84aeb10a9b557f01e98ec23e2572ce14e24e59


def tournament_int(tournament_int_or_str):
    "Convert tournament int or str to int"
    if nx.isstring(tournament_int_or_str):
        return tournament_str2int(tournament_int_or_str)
    elif nx.isint(tournament_int_or_str):
        numbers = nx.tournament_numbers(active_only=True)
        if tournament_int_or_str not in numbers:
            raise ValueError("`tournament_int_or_str` not recognized")
        return tournament_int_or_str
    raise ValueError('input must be a str or int')


def tournament_str(tournament_int_or_str):
    "Convert tournament int or str to str"
    if nx.isstring(tournament_int_or_str):
        if tournament_int_or_str not in nx.tournament_names(active_only=True):
            raise ValueError('tournament name is unknown')
        return tournament_int_or_str
    elif nx.isint(tournament_int_or_str):
        return tournament_int2str(tournament_int_or_str)
    raise ValueError('input must be a str or int')


def tournament_all(as_str=True, active_only=True):
    "List of all tournaments as strings (default) or integers."
    tournaments = []
    if as_str:
        for number, name in tournament_iter(active_only):
            tournaments.append(name)
    else:
        for number, name in tournament_iter(active_only):
            tournaments.append(number)
    return tournaments


def tournament_iter(active_only=True):
    "Iterate, in order, through tournaments yielding tuple of (int, str)"
    numbers = nx.tournament_numbers(active_only)
    numbers.sort()
    for t in numbers:
        yield t, tournament_int2str(t, active_only)


def tournament_int2str(tournament_int, active_only=True):
    "Convert tournament integer to string name"
<<<<<<< HEAD
    if tournament_int not in nx.tournament_numbers(active_only):
        value_err = "tournament_int `{}` not recognized".format(tournament_int)
        raise ValueError(value_err)
=======
    if tournament_int not in nx.tournament_numbers(active_only=True):
        raise ValueError(
            "`tournament_int` {} not recognized".format(tournament_int))
<<<<<<< HEAD
>>>>>>> 4f84aeb10a9b557f01e98ec23e2572ce14e24e59
=======
>>>>>>> 4f84aeb10a9b557f01e98ec23e2572ce14e24e59
    for tourney in TOURNAMENTS:
        if tourney['number'] == tournament_int:
            return tourney['name']
    raise RuntimeError("Did not find tournament name")


def tournament_str2int(tournament_str, active_only=True):
    "Convert tournament name (as str) to tournament integer"
    if tournament_str not in nx.tournament_names(active_only):
        raise ValueError('`tournament_str` name not recognized')
    for tourney in TOURNAMENTS:
        if tourney['name'] == tournament_str:
            return tourney['number']
    raise RuntimeError("Did not find tournament number")


def tournament_count(active_only=True):
    "Returns the number of tournaments as an integer"
    if active_only:
        count = 0
        for tourney in TOURNAMENTS:
            if tourney['active']:
                count += 1
    else:
        count = len(TOURNAMENTS)
    return count


def tournament_names(active_only=True):
    "List of tournament names; by default active tournaments only"
    names = []
    for tourney in TOURNAMENTS:
        if active_only:
            if tourney['active']:
                names.append(tourney['name'])
        else:
            names.append(tourney['name'])
    return names


def tournament_numbers(active_only=True):
    "List of tournament numbers; by default active tournaments only"
    numbers = []
    for tourney in TOURNAMENTS:
        if active_only:
            if tourney["active"]:
                numbers.append(tourney['number'])
        else:
            numbers.append(tourney['number'])
    return numbers


def tournament_isactive(tournament_int_or_str):
    "True if tournament is active; False otherwise"
    name = nx.tournament_str(tournament_int_or_str)
    for tourney in TOURNAMENTS:
        if tourney['name'] == name:
            return tourney['active']
    raise RuntimeError("Tournament name not found")
