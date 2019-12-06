# Code lists are in https://ec.europa.eu/eurostat/data/metadata/code-lists
# Data sources must be directly downloaded from https://ec.europa.eu/eurostat/data/database
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from data_utils import string_encoding_to_datetime


def load_tsv(filename, decode=False):
    # Just calls legacy code because I'm lazy
    return DataManager(filename).get_full_dataframe(decode=decode)

# ONLY TAKES IN TSV, DIRECTLY DOWNLOADED FROM THE WEBSITE
# Deprecated. Lots of legacy code
class DataManager(object):

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

        # Duplicate declarations so that my IDE works correctly
        self.dataframe = None
        self.legend = None
        self.legend_reversed = None
        self.orig_index_fields = None
        self.orig_column_fields = None
        self.column_fields = None
        

        def parsing_function(value):
            if isinstance(value, str):
                if ":" in value:
                    return np.nan
                value = re.sub(r' \w*$', '', value)
            else:
                np.isnan(value)
                return value
            
            try:
                return float(value)
            except:
                print("Reformated value '{0}' to nan".format(value))
                return np.nan

        def extract_legend(legend_str):
            legend_str_parts = legend_str.split("\\")
            orig_index_fields = legend_str_parts[0].split(",")
            orig_column_fields = legend_str_parts[1].split(",")

            legend = {}
            for field in orig_index_fields:
                legend[field] = self.get_dict_from_dic(field + ".dic")
            self.legend = legend
            self.orig_index_fields = orig_index_fields
            self.orig_column_fields = orig_column_fields
            
            self.legend_reversed = {}
            for key, mapping in legend.items():
                self.legend_reversed[key] = {}
                for value, mapped_value in mapping.items():
                    self.legend_reversed[mapped_value] = value
            
        def extract_full_dataframe():
            
            df = pd.read_csv(self.get_abs_data_store_path(self.file_name), sep='\t')
            df.iloc[:,1:] = df.iloc[:,1:].applymap(parsing_function)
            df.columns = df.columns.str.strip()
            
            extract_legend(df.columns[0])
            df.rename(columns={df.columns[0]: "orig_index"}, inplace=True)
            df["orig_index"] = df["orig_index"].str.split(",")
            for i, col in enumerate(self.orig_index_fields):
                df[col] = df["orig_index"].apply(lambda x: x[i])
            df = df.drop(["orig_index"], axis=1)

            df = df.melt(id_vars=self.orig_index_fields, var_name=self.orig_column_fields[0], value_name="value")

            self.column_fields = self.orig_column_fields + self.orig_index_fields
            df["value"] = df["value"].astype(float)
            self.dataframe = df

        extract_full_dataframe()
    
    def get_full_dataframe(self, decode=False):
        if decode:
            return self.decode_values(self.dataframe)
        return self.dataframe
    
    @classmethod
    def get_dict_from_dic(cls, filename):
        result = {}
        with open(cls.get_abs_data_store_path(filename), "r", encoding="utf8") as file:
            for line in file:
                parts = line.strip().split("\t")
                result[parts[0]] = parts[1]
        return result
    
    def get_filters(self):
        return self.column_fields
    
    def get_series(self, filters, decode=False):
        if len(filters) != len(self.get_filters()) - 1:
            raise ValueError("Shoudl have {0} filters. Choose from {1}".format(
                len(self.get_filters()) - 1, self.get_filters()
                ))

        for key in filters.keys():
            if key not in self.get_filters():
                raise ValueError("Filter {0} doesn't exist in possible filters {1}".format(key, self.get_filters()))
        
        temp_df = self.dataframe

        for key, value in filters.items():
            if key in self.get_filters():
                temp_df = temp_df[temp_df[key] == value]
            
        
        if decode:
            filters = self.decode_values(filters)

        temp_df = temp_df.drop([x for x in self.get_filters() if x in temp_df], axis=1)
        res = temp_df.iloc[:,0]
        res = res.rename(str(filters))
        return res
    
    def get_unique_values(self, col=None, with_decode=False):
        def get_unique_values(col_name):
            uniques = pd.unique(self.dataframe[col_name]).tolist()
            if not with_decode:
                return uniques
            return list(zip(uniques, self.decode_values(uniques, col_name)))
        
        if col is not None:
            return get_unique_values(col)
        else:
            ret = {}
            for key in self.get_filters():
                ret[key] = get_unique_values(key)
            return ret
    
    def print_unique_values(self, col=None):
        if col is not None:
            res = self.get_unique_values(col, with_decode=True)
            res = {col: res}
        else:
            res = self.get_unique_values(with_decode=True)
        pass
        max_key_length = max([
            max([len(x[0]) for x in d]) for d in res.values()
        ])
        preformatted_str = "\t{0: <|||}  {1}".replace("|||", str(max_key_length))
        for key in res.keys():
            print(key)
            for value, decoded_value in res[key]:
                print(preformatted_str.format(value, decoded_value))
    
    def __process_encoding_df(self, df, decode):
        ndf = df.copy()
        source = self.legend if decode else self.legend_reversed
        for key, mapping in source.items():
            if key in ndf:
                def mapping_func(x):
                    if x in mapping:
                        return mapping[x]
                    raise KeyError("Requested value {0} in {1} column of dataframe does not exist in legend during {2}. Invalid".format(
                        x, key, "decode" if decode else "encode"))
                ndf[key] = ndf[key].map(mapping_func)
        return ndf
    
    def __process_encoding_dict(self, d, decode):
        source = self.legend if decode else self.legend_reversed
        res = {}
        for key in d.keys():
            if key in source.keys() and d[key] in source[key]:
                res[key] = source[key][d[key]]
            else:
                res[key] = d[key]
        return res
    
    def __process_encoding_list(self, d, source_key, decode):
        source = self.legend if decode else self.legend_reversed
        res = []
        for key in d:
            if source_key in source.keys():
                res.append(source[source_key][key])
            else:
                res.append(key)
        return res
    
    def decode_values(self, data, source_key=None):
        if isinstance(data, pd.DataFrame):
            return self.__process_encoding_df(data, True)
        elif isinstance(data, dict):
            return self.__process_encoding_dict(data, True)
        elif isinstance(data, list) or isinstance(data, pd.Series):
            return self.__process_encoding_list(data, source_key, True)
        raise TypeError("argument datatype not supported")

    def encode_values(self, data, source_key=None):
        if isinstance(data, pd.DataFrame):
            return self.__process_encoding_df(data, False)
        elif isinstance(data, dict):
            return self.__process_encoding_dict(data, False)
        elif isinstance(data, list) or isinstance(data, pd.Series):
            return self.__process_encoding_list(data, source_key, False)
        raise TypeError("argument datatype not supported")
    
    @classmethod
    def get_abs_data_store_path(cls, data_store_file_name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_store", data_store_file_name)

    
class DataComparators(object):
    @staticmethod
    def convert_to_datatime_arrary(encoded_dates):
        res = []
        for encoded_date in encoded_dates:
            res.append(string_encoding_to_datetime(encoded_date))
        return res
    
    @staticmethod
    def plot_line_over_time(ydms, yfilters, **kwargs):
        # Assumes numeric, time based data
        for ydm, fil in zip(ydms, yfilters):
            s = ydm.get_series(fil, decode=True)
            plt.plot(DataComparators.convert_to_datatime_arrary(s.index), s.values, label=s.name, **kwargs)
        
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_scatter(ydm1: DataManager, ydm2: DataManager, yfilter1: dict, yfilter2: dict, **kwargs):
        s1 = ydm1.get_series(yfilter1, decode=True)
        s2 = ydm2.get_series(yfilter2, decode=True)
        common_index = s1.index.intersection(s2.index)
        s1 = s1[common_index]
        s2 = s2[common_index]
        plt.scatter(s1, s2)

        plt.xlabel(s1.name)
        plt.ylabel(s2.name)
        plt.show()

def test_DataManager():
    dm = DataManager("tour_occ_arm.tsv")
    print(dm.get_filters())
    print(dm.get_series(
        dict([(key, vals[0]) for key, vals in dm.get_unique_values().items() if key != "time"])
    ))
    print(dm.get_series(
        dict([(key, vals[0]) for key, vals in dm.get_unique_values().items() if key != "time"]),
        decode=True
    ))
    print(dm.get_series({'c_resid': 'FOR', 'unit': 'NR', 'nace_r2': 'I551', 'time': '2018M11'}, decode=True))
    # print()
    # dm.print_unique_values()
    print(dm.get_full_dataframe())
    print(dm.get_full_dataframe(decode=True))

def test_DataComparator():
    dm1 = DataManager("tour_occ_arm.tsv")
    dm2 = DataManager("tour_lfsq1r2.tsv")
    # DataComparators.plot_line_over_time(
    #     [
    #         dm1,
    #         dm2
    #     ],
    #     [
    #         dict([(key, vals[0]) for key, vals in dm1.get_unique_values().items()]),
    #         dict([(key, vals[0]) for key, vals in dm2.get_unique_values().items()])
    #     ]
    #     )
    DataComparators.plot_scatter(
            dm1,
            dm1,
            dict([(key, vals[0]) for key, vals in dm1.get_unique_values().items()]),
            dict([(key, vals[0]) for key, vals in dm1.get_unique_values().items()])
        )

if __name__ == "__main__":
    test_DataManager()
















