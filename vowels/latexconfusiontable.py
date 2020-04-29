
#Code taken from github user kristeey
#https://github.com/kristeey/TTT4275_EDC/


vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']

def print_confusion(conf):
    """
    Prints a latex formatted confusion matrix
    """
    print("""\\begin{table}[H]
\\caption{}
\\centering
\\begin{tabular}{|c|llllllllllll|}""")
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(vowels) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = vowels[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 11:
            rw += '\\hline'
        print(rw)
    print("""\\end{tabular}
\\end{table}""")