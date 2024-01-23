

from doespy.etl.steps.loaders import PlotLoader
from doespy.etl.steps.extractors import Extractor

from typing import Dict, List
import pandas as pd
import os
import warnings


class ErrorInfoExtractor(Extractor):

    def default_file_regex():
        return ["^stderr.log$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:

        with open(path, "r") as f:
            content = f.read().replace("\n", " ")

        if content.strip() and not content.strip().isspace():
            # report error
            return [{"path": path, "error": content}]
        else:
            # ignore empty error files
            return []


from doespy.etl.etl_util import expand_factors

class WarningLoader(PlotLoader):

    cols: List[str] = ["$FACTORS$"] # if no cols are specified, factor columns are used

    warning_col: str

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        output_dir = self.get_output_dir(etl_info)
        output_file = os.path.join(output_dir, f"{etl_info['pipeline']}.csv")

        job_id_cols = ["suite_name", "suite_id", "exp_name", "run", "rep"]

        #[memory + runtime + error msg are always there]
        df = df.sort_values(by=job_id_cols)

        if self.warning_col not in df.columns:
            df[self.warning_col] = None
        else:
            warnings.warn(f"At least one job has a warning. Please check the output file: {output_file}")

        # focus on relevant columns
        cols = self.cols
        cols += job_id_cols
        cols = expand_factors(df, self.cols)
        cols += [self.warning_col]
        df = df[cols]

        # aggregate by job
        def aggregate_data(x):
            unique_values = x.dropna().unique()
            if len(unique_values) == 1:
                return unique_values[0]
            else:
                return list(unique_values)

        df = df.groupby(job_id_cols).agg(aggregate_data)

        # write output to file

        df.to_csv(output_file)

        # print output to console
        # print(df.to_string())
