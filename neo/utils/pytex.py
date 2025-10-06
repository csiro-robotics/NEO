import numpy as np
import re

from scipy.stats import rankdata
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LatexTemplates:
    """Class with templates for generating dynamic objects"""

    TABLE_TEMPLATE: str = \
    """
    \\begin{table*}[]
        \\centering
        \\scriptsize
        \\small
        \\begin{tabular}{TABLE_FORMAT} % the number of columns needed
            \\toprule
            COLUMNS \\\\
            \\midrule
            EXPERIMENT_LINES
            \\bottomrule
        \\end{tabular}
        \\caption{\\textbf{CAPTION_TITLE.} CAPTION_TEXT}
        \\label{tab:TABLE_LABEL}
    \\end{table*}
    """


class PyTex:
    @staticmethod
    def convert_nested_results_to_tablecv(scene_results, columns, exp_types=None):
        # scene_results: {method: {exp_type: {metric: value, ...}, ...}, ...}
        # columns: list of metric names
        # exp_types: list of exp types to include (e.g. ['full', 'cropped'])
        methods = list(scene_results.keys())
        if exp_types is None:
            # Find all exp_types present
            exp_types = set()
            for method in methods:
                exp_types.update(scene_results[method].keys())
            exp_types = list(exp_types)
        table_data = {"method_name": methods}
        for exp_type in exp_types:
            table_data[exp_type] = []
            for method in methods:
                metrics = scene_results[method].get(exp_type, {})
                row = {col: metrics.get(col, -1) for col in columns}
                table_data[exp_type].append(row)
        return table_data
    
    def __init__(self) -> None:
        
        self.templates = LatexTemplates()

    def remove_substrings(self, input_string: str, substrings: List[str]) -> str:
        pattern = '|'.join(map(re.escape, substrings))

        return re.sub(pattern, '', input_string)

    def get_table_columns(self, columns: List[str], skip_columns: List[int]) -> str:
        table_columns = "\n             "
        for i, column in enumerate(columns):
            if i in skip_columns:
                continue

            if "+" in column:
                table_columns += f"{' & ' if i != 0 else ''}{column.strip('+')} $\\uparrow$"

            elif "-" in column:
                table_columns += f"{' & ' if i != 0 else ''}{column.strip('-')} $\\downarrow$"

            else:
                table_columns += f"{' & ' if i != 0 else ''}{column}"

        return table_columns
    
    def get_best_values_in_columns(self, columns: List[str], data: Dict[str, float], digits: List[int] = None) -> np.ndarray:
        mat = []
        for key, value in data.items():
            if "method_name" in key:
                continue
            else:
                mat.append(value)

        mat = np.array(mat, dtype=float).T

        if digits is None:
            digits = np.uint(np.ones(len(columns)) * 2)

        for i, dec in enumerate(digits):
            mat[:, i] = np.round(mat[:, i], decimals=dec)

        min_vals = rankdata(mat, method='min', axis=0) - 1
        max_vals = mat.shape[0] - rankdata(mat, method='max', axis=0)

        return mat, min_vals, max_vals
    
    def prepare_rows(self, mat: np.ndarray, names: List[str], columns: List[str], skip_columns: List[int], digits: List[int], min_vals: np.ndarray, max_vals: np.ndarray, name_map: Dict[str, str]) -> List[str]:
        rows = []
        for i in range(mat.shape[0]):

            name = names[i]

            row = f"{name if name_map is None else name_map[name]} & "
            for j in range(mat.shape[1]):

                if j in skip_columns:
                    continue

                idxs = None
                if "+" in columns[j + 2]:
                    idxs = max_vals
                elif "-" in columns[j + 2]:
                    idxs = min_vals
                else:
                    idxs = np.ones_like(mat) * 2
                
                value = mat[i, j]
                if value != -1 and value != 1000000:
                    if idxs[i, j] == 0:
                        row += f"\\textbf{{{value:.{digits[j]}f}}} & "
                    elif idxs[i, j] == 1:
                        row += f"\\underline{{{value:.{digits[j]}f}}} & "
                    else:
                        row += f"{value:.{digits[j]}f} & "
                
                else:
                    row += "- & "

            row = row[0:-2] + "\\\\ \n            "
            rows.append(row)

        experiment_lines = "".join(rows)

        return experiment_lines

    def table(self,
              columns: List[str], 
              data: Dict[str, float], 
              table_format: str = None,
              digits: List[int] = None, 
              caption_title: str = "caption title", 
              table_label: str = "label", 
              caption_text: str = "caption text",
              skip_columns: List[int] = [],
              name_map: Dict[str, str] = None
              ) -> str:
        
        # Assign default table format
        if table_format is None:
            table_format = "l" * (len(columns) + 1)

        # Add arrows to the columns
        table_columns = self.get_table_columns(columns, skip_columns)

        # Find best values in the columns
        mat, min_vals, max_vals = self.get_best_values_in_columns(columns, data, digits)

        # Prepare rows
        names = data["method_name"]
        expeimrent_lanes = self.prepare_rows(mat, names, columns, skip_columns, digits, min_vals, max_vals, name_map)

        # Fill the template
        s = self.templates.TABLE_TEMPLATE.replace("COLUMNS", table_columns)
        s = s.replace("TABLE_FORMAT", table_format)
        s = s.replace("EXPERIMENT_LINES", expeimrent_lanes)
        s = s.replace("CAPTION_TITLE", caption_title)
        s = s.replace("CAPTION_TEXT", caption_text)
        s = s.replace("TABLE_LABEL", table_label)

        print(s)

        return s
    
    def table_cv(self,
                 columns: List[str], 
                 data: Dict[str, Dict[str, float]], 
                 table_format: str = None,
                 digits: List[int] = None, 
                 caption_title: str = "caption title", 
                 table_label: str = "label", 
                 caption_text: str = "caption text",
                 skip_columns: List[int] = [],
                 name_map: Dict[str, str] = None
                 ) -> str:
        
        # Assign default table format
        if table_format is None:
            table_format = "l" * (len(columns) + 1)

        # Add arrows to the columns
        table_columns = self.get_table_columns(columns, skip_columns)

        # Find best values in the columns
        experiments_lines = [[], []]
        for scene, data in data.items():
            names = data["method_name"]
            for idx, (data_type, digit) in enumerate(zip([key for key in data.keys() if isinstance(data[key], dict)], digits)):
                mat, min_vals, max_vals = self.get_best_values_in_columns(columns, data[data_type], digit)
                experiment_line = self.prepare_rows(mat, names, columns, skip_columns, digit, min_vals, max_vals, name_map)

                experiment_line = f'\multirow{{{3}}}{{*}}{{{scene.replace("_", " "). capitalize()}}} & ' + experiment_line
                experiment_line = re.sub(r' {12}', '             & ', experiment_line)
                experiment_line = experiment_line[:-3]
                experiment_line += "\\midrule\n            "
                experiments_lines[idx].append(experiment_line)

        table = []
        for experiment_line in experiments_lines:
            experiment_lines = "".join(experiment_line)

            # Fill the template
            s = self.templates.TABLE_TEMPLATE.replace("COLUMNS", table_columns)
            s = s.replace("TABLE_FORMAT", table_format)
            s = s.replace("EXPERIMENT_LINES", experiment_lines)
            s = s.replace("CAPTION_TITLE", caption_title)
            s = s.replace("CAPTION_TEXT", caption_text)
            s = s.replace("TABLE_LABEL", table_label)

            table.append(s)
        table = "\n\n".join(table)

        print(table)

        return s