import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_management import DataManager, string_encoding_to_datetime
from datetime import datetime
plt.rcParams['figure.figsize'] = 20, 16


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


YEAR_BOUNDS = (1998, 2018)
def extrapolate_time(inp_df, kind, year_bounds = YEAR_BOUNDS):
    def __sub(sub_df):
        df = sub_df.copy()
        df.drop_duplicates(inplace=True, keep = "last", subset=[col for col in df.columns if col != "value"])
        df["time"] = df["time"].apply(string_encoding_to_datetime)

        if kind == "year":
            indices = [datetime(y, 1, 1) for y in range(YEAR_BOUNDS[0], YEAR_BOUNDS[1]+1)]
        elif kind == "month":
            indices = [
                datetime(y, m, 1) for y in range(YEAR_BOUNDS[0], YEAR_BOUNDS[1]+1) for m in range (1,12+1)
            ]

        res = pd.DataFrame(columns = df.columns)
        res = res.set_index("time")
        for geo in df["geo"].unique():
            interpolated = pd.DataFrame(df[df["geo"] == geo].set_index("time"), index=indices)
            interpolated = interpolated.interpolate(method='time', limit_direction="both")
            interpolated["geo"] = geo
            res = res.append(interpolated)
        return res.rename_axis('time').reset_index()
    extra_columns = [col for col in inp_df.columns if col not in ["time", "geo", "value"]]
    if len(extra_columns) == 0:
        return __sub(inp_df)
    else:
        res = pd.DataFrame(columns = inp_df.columns)
        groups = inp_df.groupby(extra_columns)
        for group in groups.groups:
            temp = __sub(groups.get_group(group))
            for col in extra_columns:
                temp[col] = group[col] if len(extra_columns) > 1 else group
            res.append(temp)
        return temp
        
# extrapolate_time(select_by_dict(lan_settl, {
#     "unit": "Percentage"
# }), "month")
# extrapolate_time(lan_settl, "month")

"""
Inputs format:
    dict_of_parameters = {
            "name": {
                "source": dataframe,
                "fields": {
                    "name": [list of accepted values] OR expected value OR "ALL"
                }
            }
        },
    The other inputs are the same as extrapolate_time
"""
def join_and_extrapolate_values_from_multiple_sources(dict_of_parameters, kind, year_bounds = YEAR_BOUNDS):
    final_df = None
    for name, df_parameters in dict_of_parameters.items():
        source = df_parameters["source"]
        fields_parameters = df_parameters["fields"]
        
        # Filter only to wanted values
        for field_name, accepted_values in fields_parameters.items():
            if isinstance(accepted_values, list):
                source = source[source[field_name].isin(accepted_values)]
            elif accepted_values == "ALL":
                continue
            else:
                source = source[source[field_name] == accepted_values]
        
        unqiue_field_combinations = source[fields_parameters.keys()].drop_duplicates()
        for index, row in unqiue_field_combinations.iterrows():
            identifier = "; ".join([str(r) for r in row.values])
            filtered_df = select_by_dict(source, row.to_dict())
            filtered_df = extrapolate_time(filtered_df, kind, year_bounds)
            filtered_df = filtered_df.set_index(["geo", "time"])
            if final_df is None:
                final_df = pd.DataFrame(index=filtered_df.index)
            final_df[identifier] = filtered_df["value"]
    return final_df
        
# Land cover overview by NUTS 2 regions (lan_lcv_ovw)
#     unit:
#         Percentage
#     landcover:
#         ALL

def test():
    lan_lcv_ovw = DataManager("lan_lcv_ovw.tsv").get_full_dataframe(decode=True)
    res = join_and_extrapolate_values_from_multiple_sources({
        "land_cover" : {
            "source": lan_lcv_ovw,
            "fields": {
                "unit": "Percentage",
                "landcover": "ALL",
            }
        }
    }, kind="year")
    print(res)
    print(list(res.index))

if __name__ == "__main__":
    test()