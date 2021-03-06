HOMOGLYPH_MAP = {
    '0': [ u'\uff10' ],
    '1': [ u'\uff11' ],
    '2': [ u'\uff12' ],
    '3': [ u'\uff13' ],
    '4': [ u'\uff14' ],
    '5': [ u'\uff15' ],
    '6': [ u'\uff16' ],
    '7': [ u'\uff17' ],
    '8': [ u'\uff18' ],
    '9': [ u'\uff19' ],
    '-': [ u'\u2010' ],
    'a': [ u'\uff41', u'\u0430' ],
    'b': [ u'\uff42' ],
    'c': [ u'\uff43', u'\u0441' ],
    'd': [ u'\uff44', u'\u217e' ],
    'e': [ u'\uff45', u'\u0435' ],
    'f': [ u'\uff46' ],
    'g': [ u'\uff47' ],
    'h': [ u'\uff48', u'\u04bb' ],
    'i': [ u'\uff49', u'\u0456', u'\u2170' ],
    'j': [ u'\uff4a', u'\u0458' ],
    'k': [ u'\uff4b' ],
    'l': [ u'\uff4c', u'\u217c' ],
    'm': [ u'\uff4d', u'\u217f' ],
    'n': [ u'\uff4e' ],
    'o': [ u'\uff4f', u'\u03BF', u'\u043E' ],
    'p': [ u'\uff50', u'\u0440' ],
    'q': [ u'\uff51' ],
    'r': [ u'\uff52' ],
    's': [ u'\uff53', u'\u0455' ],
    't': [ u'\uff54' ],
    'u': [ u'\uff55' ],
    'v': [ u'\uff56', u'\u2174' ],
    'w': [ u'\uff57' ],
    'x': [ u'\uff58', u'\u0445', u'\u2179' ],
    'y': [ u'\uff59', u'\u0443' ],
    'z': [ u'\uff5a' ],
    'A': [ u'\uff21', u'\u0391', u'\u0410' ],
    'B': [ u'\uff22', u'\u0392', u'\u0412' ],
    'C': [ u'\uff23', u'\u03F9', u'\u0421', u'\u216D' ],
    'D': [ u'\uff24', u'\u216E' ],
    'E': [ u'\uff25', u'\u0395', u'\u0415' ],
    'F': [ u'\uff26', u'\u03DC' ],
    'G': [ u'\uff27' ],
    'H': [ u'\uff28', u'\u0397', u'\u041D' ],
    'I': [ u'\uff29', u'\u0399', u'\u0406', u'\u2160' ],
    'J': [ u'\uff2a', u'\u0408' ],
    'K': [ u'\uff2b', u'\u039A', u'\u041A', u'\u212A' ],
    'L': [ u'\uff2c', u'\u216C' ],
    'M': [ u'\uff2d', u'\u039C', u'\u041C', u'\u216F' ],
    'N': [ u'\uff2e', u'\u039D' ],
    'O': [ u'\uff2f', u'\u039F', u'\u041E' ],
    'P': [ u'\uff30', u'\u03A1', u'\u0420' ],
    'Q': [ u'\uff31' ],
    'R': [ u'\uff32' ],
    'S': [ u'\uff33', u'\u0405' ],
    'T': [ u'\uff34', u'\u03A4', u'\u0422' ],
    'U': [ u'\uff35' ],
    'V': [ u'\uff36', u'\u2164' ],
    'W': [ u'\uff37' ],
    'X': [ u'\uff38', u'\u03A7', u'\u0425', u'\u2169' ],
    'Y': [ u'\uff39', u'\u03A5', u'\u04AE' ],
    'Z': [ u'\uff3a', u'\u0396' ],
}

#TODO use max_change
def homoglyphs(word, max_change=2):
    homoglyphs = set()
    for i in range(len(word)):
        replacements = HOMOGLYPH_MAP.get(word[i]) or []
        for replacement in replacements:
            homoglyph = word[:i] + replacement + word[i+1:]
            homoglyphs.add(homoglyph)
    return homoglyphs