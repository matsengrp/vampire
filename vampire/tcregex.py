import exrex
import pandas as pd
import re
import vampire.xcr_vector_conversion as xcr_vector_conversion

# We don't take the gap character.
aas = xcr_vector_conversion.AA_LIST[:20]

# This is how we specify the "tcregex" format.
replacement_dict = {
    # . means any amino acid
    '\.': '[' + ''.join(aas) + ']',
    # Amino acid ambiguity codes as per
    # http://www.virology.wisc.edu/acp/Classes/DropFolders/Drop660_lectures/SingleLetterCode.html
    # https://febs.onlinelibrary.wiley.com/doi/pdf/10.1111/j.1432-1033.1984.tb07877.x
    'B': '[DN]',
    'Z': '[EQ]'
}


def build_regex(cdr3_tcregex):
    """
    We convert a tcregex CDR3 to a normal regex.
    """
    s = cdr3_tcregex
    for pattern, repl in replacement_dict.items():
        s = re.sub(pattern, repl, s)
    return s


def sample_cdr3_tcregex(cdr3_tcregex, n):
    """
    Sample from a CDR3 tcregex.
    """
    r = build_regex(cdr3_tcregex)
    return [exrex.getone(r) for i in range(n)]


def sample_split_tcregex(v_gene, j_gene, cdr3_tcregex, n):
    """
    Sample from a tcregex that has been split into its components.
    """
    df = pd.DataFrame({'amino_acid': sample_cdr3_tcregex(cdr3_tcregex, n)})
    df['v_gene'] = v_gene
    df['j_gene'] = j_gene
    return (df)


def sample_tcregex(tcregex, n):
    inputs = tcregex.split(',')
    if len(inputs) != 3:
        raise Exception("tcregexes should be specified in the format `v_gene,j_gene,cdr3_tcregex`.")
    v_gene, j_gene, cdr3_tcregex = inputs
    return sample_split_tcregex(v_gene, j_gene, cdr3_tcregex, n)
