def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs

    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    df1_id = df1[patient_col].to_numpy()
    df1_id_set = set(df1_id)
    df2_id = df2[patient_col].to_numpy()
    df2_id_set = set(df2_id)

    df1_patients_unique = len(df1_id_set)
    # print("df1 unique", df1_patients_unique)
    df2_patients_unique = len(df2_id_set)
    # print("df2 unique", df2_patients_unique)

    patients_in_both_groups = list(df1_id_set.intersection(df2_id_set))
    overlap = len(patients_in_both_groups)
    # print("print overlap", overlap)

    # leakage contains true if there is patient overlap, otherwise false.
    if overlap > 0:
        leakage = True
    else:
        leakage = False

    return leakage