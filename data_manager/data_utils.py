import re
import datetime

def string_encoding_to_datetime(value):
    match = re.fullmatch(r"\s*(\d{4})M(\d{2})\s*", value)
    if match is not None:
        return datetime.datetime(
            year = int(match.group(1)),
            month = int(match.group(2)),
            day = 1
        )
    match = re.fullmatch(r"\s*(\d{4})Q(\d)\s*", value)
    if match is not None:
        # according to https://investinganswers.com/dictionary/q/quarter-q1-q2-q3-q4
        # Not using these dates, for compatibility with other functions
        quarter_dates = {
            1: (2, 1),
            2: (5, 1),
            3: (8, 1),
            4: (1, 1),
        }
        return datetime.datetime(
            year = int(match.group(1)),
            month = quarter_dates[int(match.group(2))][0],
            day = quarter_dates[int(match.group(2))][1]
        )
    match = re.fullmatch(r"\s*(\d{4})\s*", value)
    if match is not None:
        return datetime.datetime(
            year = int(match.group(1)),
            month = 1,
            day = 1
        )
    raise ValueError("Unrecongnized datetime format '{0}'".format(value))

def select_by_dict(df, criteria):
    res = df.query(' and '.join(["{} == '{}'".format(k,v) for k,v in criteria.items()]))
    return res[[x for x in df.columns if x not in criteria.keys()]]

def get_df_with_split_time(df):
    res = df.copy()
    dts = res["time"].apply(string_encoding_to_datetime)
    res["year"] = [d.year for d in dts]
    res["month"] = [d.month for d in dts]
    res = res.drop("time", axis=1)
    return res

def print_unique_values(df, col=None):
    if col is not None:
        res = {col: df["col"].unique()}
    else:
        res = {}
        for c in df.columns:
            res[c] = df[c].unique().tolist()
    preformatted_str = "\t{0}"
    for key in res.keys():
        print(key)
        for value in res[key]:
            print(preformatted_str.format(value))
            if key == "value":
                print(preformatted_str.format("...etc"))
                break


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])