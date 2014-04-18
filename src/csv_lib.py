import csv


def columns(fname):
    """
    counts the number columns in the csv
    """
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        return len(reader.next())
