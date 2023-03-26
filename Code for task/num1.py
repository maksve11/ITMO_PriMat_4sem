def get_rightdiff_atpoint(function, argument, h):
    return (function(argument + h) - function(argument)) / h


def get_rightdiff(function, arguments, h):
    return [get_rightdiff_atpoint(function, x, h) for x in arguments]


def get_leftdiff_atpoint(function, argument, h):
    return (function(argument) - function(argument - h)) / h


def get_leftdiff(function, arguments, h):
    return [get_leftdiff_atpoint(function, x, h) for x in arguments]


def get_diff_atpoint(function, argument, h):
    return (function(argument + h) - function(argument - h)) / (2 * h)


def get_diff(function, arguments, h):
    return [get_diff_atpoint(function, x, h) for x in arguments]