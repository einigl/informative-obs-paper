""" Helpers for EMIR settings """

orionb_obstime = 4 * 0.1 # min (factor 2 still unexplained to match the data)

def emir_bands():
    return {
        "E090": [73, 117], # GHz
        "E150": [125, 184],
        "E230": [202, 274],
        "E330": [277, 375] # [277, 350]
    }
