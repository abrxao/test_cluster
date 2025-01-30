from numpy import log10


def linearToDB(x):
    return 10 * log10(x)
