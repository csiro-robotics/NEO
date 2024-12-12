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
    \\begin{table}[]
        \\centering
        \\captionsetup{font=scriptsize}
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
    \\end{table}
    """


class PyTex:
    
    def __init__(self) -> None:
        
        self.templates = LatexTemplates()


    def remove_substrings(self, input_string: str, substrings: List[str]) -> str:
        pattern = '|'.join(map(re.escape, substrings))

        return re.sub(pattern, '', input_string)


    def table(self,
              columns: List[str], 
              data: Dict[str, float], 
              table_format: str = None,
              digits: List[int] = None, 
              caption_title: str = "caption title", 
              table_label: str = "label", 
              caption_text: str = "caption text",
              skip_columns: List[int] = [],
              ) -> str:
        
        # Assign default table format
        if table_format is None:
            table_format = "l" * (len(columns) + 1)

        # Add arrows to the columns
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

        # Find best values in the columns
        mat = []
        names = data["Method name"]
        for key, value in data.items():
            if "Method name" in key:
                continue
            else:
                mat.append(value)

        mat = np.array(mat, dtype=float).T

        for i, dec in enumerate(digits):
            mat[:, i] = np.round(mat[:, i], decimals=dec)

        min_vals = rankdata(mat, method='min', axis=0) - 1
        max_vals = mat.shape[0] - rankdata(mat, method='max', axis=0)

        if digits is None:
            digits = np.uint(np.ones(len(columns)) * 2)

        # Prepare rows
        rows = []
        for i in range(mat.shape[0]):

            name = names[i]

            if "SDXL" in name and "x4" in name and "Random" in name:
                name = "Our Preview + Update"

            row = f"{name} & "
            for j in range(mat.shape[1]):

                if j in skip_columns:
                    continue

                idxs = None
                if "+" in columns[j + 1]:
                    idxs = max_vals
                elif "-" in columns[j + 1]:
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

        expeimrent_lanes = "".join(rows)

        # Fill the template
        s = self.templates.TABLE_TEMPLATE.replace("COLUMNS", table_columns)
        s = s.replace("TABLE_FORMAT", table_format)
        s = s.replace("EXPERIMENT_LINES", expeimrent_lanes)
        s = s.replace("CAPTION_TITLE", caption_title)
        s = s.replace("CAPTION_TEXT", caption_text)
        s = s.replace("TABLE_LABEL", table_label)

        print(s)

        return s