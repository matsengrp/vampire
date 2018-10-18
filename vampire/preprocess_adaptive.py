"""
Preprocess data coming from Adaptive immunoSEQ.

https://clients.adaptivebiotech.com/assets/downloads/immunoSEQ_AnalyzerManual.pdf
"""


def filter_and_drop_frame(df):
    """
    Select in-frame sequences and then drop that column.
    """
    return df.query('frame_type == "In"').drop('frame_type', axis=1)


def filter_on_cdr3_bounding_aas(df):
    """
    Only take sequences that have a C at the beginning and a F or a YV at the
    end of the `amino_acid` column.

    Note that according to the Adaptive docs the `amino_acid` column is indeed
    the CDR3 amino acid.
    """
    return df[df['amino_acid'].str.contains('^C.*F$')
              | df['amino_acid'].str.contains('^C.*YV$')]


def filter_on_TCRB(df):
    """
    Only take sequences that have a resolved TCRB gene for V and J.
    """
    return df[df['v_gene'].str.contains('^TCRB')
              & df['v_gene'].str.contains('^TCRB')]


def apply_all_filters(df):
    """
    Apply all filters.
    """
    print(f"Original data: {len(df)} rows")
    df = filter_and_drop_frame(df)
    print(f"Restricting to in-frame: {len(df)} rows")
    df = filter_on_cdr3_bounding_aas(df)
    print(f"Requiring cys-phe: {len(df)} rows")
    df = filter_on_TCRB(df)
    print(f"Requiring resolved TCRB genes: {len(df)} rows")
    return df.reset_index(drop=True)
