import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


class ExperimentResultsManager:
    def __init__(self, results_file: str = None):
        self.results = pd.DataFrame(
            columns=[
                "Experiment",
                "Implementation",
                "k",
                "Time [s]",
                "InputSize",
                "BlockSize",
            ]
        )

        self.loaded_results_file = False
        if results_file is not None:
            if os.path.exists(results_file):
                self.results = pd.read_csv(results_file)
                print(
                    f"[INFO] Loaded {len(self.results)} results from '{results_file}'"
                )
                self.loaded_results_file = True if len(self.results) > 0 else False
            else:
                print(
                    f"[WARN] File '{results_file}' not found. Starting with empty results."
                )

        sns.set_style("whitegrid")
        rcParams["figure.figsize"] = (10, 6)
        rcParams["font.size"] = 12

    def add_result(
        self,
        experiment: str,
        implementation: str,
        k: int = None,
        time_seconds: float = None,
        input_size=None,
        block_size=None,
        additional_info: dict = None,
    ):
        if self.loaded_results_file:
            print(
                "[WARN] Loaded results from file. Adding new results will not be saved unless explicitly done."
            )
            return

        row_data = {
            "Experiment": experiment,
            "Implementation": implementation,
            "k": k,
            "Time [s]": time_seconds,
            "InputSize": str(input_size) if input_size is not None else None,
            "BlockSize": str(block_size) if block_size is not None else None,
        }

        if additional_info:
            for key, val in additional_info.items():
                if key not in row_data:  # Avoid overwriting core columns
                    row_data[key] = val

        self.results.loc[len(self.results)] = row_data
        self.save_results()

    def save_results(self, filename="experiment_results.csv"):
        self.results.to_csv(filename, index=False)

    def get_cpu_time_for_input(self, input_size: str) -> float:
        df_cpu = self.results[
            (self.results["Experiment"] == "CPU")
            & (self.results["Implementation"] == "CPU")
            & (self.results["InputSize"] == str(input_size))
        ]
        if df_cpu.empty:
            return float("nan")
        assert len(df_cpu) == 1, "Multiple CPU times found for same input size."
        return df_cpu["Time [s]"].values[0]

    def _add_cpu_reference_line_if_applicable(self, ax, df_plot, reference_cpu: bool):
        if not reference_cpu or df_plot.empty:
            return

        unique_input_sizes = df_plot["InputSize"].dropna().unique()
        if len(unique_input_sizes) != 1:
            print("[INFO] Multiple (or zero) input sizes found. Skipping CPU line.")
            return

        cpu_time = self.get_cpu_time_for_input(unique_input_sizes[0])
        if not np.isnan(cpu_time):
            ax.axhline(
                cpu_time,
                ls="--",
                color="black",
                label=f"CPU @ {unique_input_sizes[0]}: {cpu_time:.4g}s",
            )
            ax.legend()

    def _filter_experiment(
        self,
        experiment_name: str = None,
        implementation_filter: str = None,
        inputsize_filter: str = None,
    ) -> pd.DataFrame:
        df = self.results.copy()
        if experiment_name is not None:
            df = df[df["Experiment"] == experiment_name]
        if implementation_filter is not None:
            df = df[df["Implementation"] == implementation_filter]
        if inputsize_filter is not None:
            df = df[df["InputSize"] == str(inputsize_filter)]
        return df

    def plot_line(
        self,
        experiment_name: str,
        x_col: str,
        y_col: str,
        hue_col: str = None,
        style_col: str = None,
        title: str = None,
        logy: bool = True,
        reference_cpu: bool = False,
    ):
        df_plot = self._filter_experiment(experiment_name)
        if df_plot.empty:
            print(f"[WARN] No data found for experiment='{experiment_name}'.")
            return

        df_plot = df_plot.dropna(subset=[x_col, y_col]).sort_values(by=x_col)

        plt.figure()
        ax = sns.lineplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue=hue_col,
            style=style_col,
            markers=True,
            dashes=False,
        )
        if logy:
            ax.set_yscale("log")

        self._add_cpu_reference_line_if_applicable(ax, df_plot, reference_cpu)

        ax.set_title(title if title else f"{experiment_name} - {y_col} vs {x_col}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_bar(
        self,
        experiment_name: str,
        x_col: str,
        y_col: str,
        hue_col: str = None,
        title: str = None,
        logy: bool = True,
        reference_cpu: bool = False,
    ):
        df_plot = self._filter_experiment(experiment_name)
        if df_plot.empty:
            print(f"[WARN] No data found for experiment='{experiment_name}'.")
            return

        df_plot = df_plot.dropna(subset=[x_col, y_col]).sort_values(by=y_col)

        plt.figure()
        ax = sns.barplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue=hue_col,
            edgecolor="black",
        )
        if logy:
            ax.set_yscale("log")

        self._add_cpu_reference_line_if_applicable(ax, df_plot, reference_cpu)

        plt.xticks(rotation=45, fontsize=10)
        ax.set_title(title if title else f"{experiment_name} - {y_col} by {x_col}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_block_size_vs_time(
        self,
        experiment_name: str,
        implementation_filter: str = None,
        inputsize_filter: str = None,
        sort_by_time: bool = False,
        logy: bool = True,
        title: str = None,
        reference_cpu: bool = False,
    ):
        df_plot = self._filter_experiment(
            experiment_name, implementation_filter, inputsize_filter
        )
        if df_plot.empty:
            print(
                "[WARN] No data for the specified filters "
                f"(experiment={experiment_name}, impl={implementation_filter}, input={inputsize_filter})"
            )
            return

        df_plot = df_plot.dropna(subset=["BlockSize", "Time [s]"])

        if sort_by_time:
            df_plot = df_plot.sort_values(by="Time [s]")
        else:
            df_plot = df_plot.sort_values(by="BlockSize")

        plt.figure()
        ax = sns.barplot(
            data=df_plot,
            x="BlockSize",
            y="Time [s]",
            edgecolor="black",
        )
        if logy:
            ax.set_yscale("log")

        self._add_cpu_reference_line_if_applicable(ax, df_plot, reference_cpu)

        plt.xticks(rotation=45, fontsize=9)
        ax.set_title(title if title else f"{experiment_name} - BlockSize vs Time [s]")
        ax.set_xlabel("BlockSize")
        ax.set_ylabel("Time [s]")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_compare_experiments(
        self,
        experiment_names: list,
        input_size_filter: str = None,
        x_col: str = "BlockSize",
        y_col: str = "Time [s]",
        hue_col: str = "Experiment",
        kind: str = "line",
        logy: bool = True,
        title: str = None,
        reference_cpu: bool = False,
    ):
        df_plot = self.results[self.results["Experiment"].isin(experiment_names)].copy()
        if input_size_filter is not None:
            df_plot = df_plot[df_plot["InputSize"] == str(input_size_filter)]

        if df_plot.empty:
            print(
                f"[WARN] No data found for experiments={experiment_names}, input_size={input_size_filter}"
            )
            return

        if kind == "bar":
            df_plot = df_plot.dropna(subset=[y_col]).sort_values(by=y_col)
        else:
            df_plot = df_plot.sort_values(by=x_col)

        plt.figure()
        if kind == "bar":
            ax = sns.barplot(
                data=df_plot,
                x=x_col,
                y=y_col,
                hue=hue_col,
                edgecolor="black",
            )
        else:
            ax = sns.lineplot(
                data=df_plot,
                x=x_col,
                y=y_col,
                hue=hue_col,
                markers=True,
                dashes=False,
            )

        if logy:
            ax.set_yscale("log")

        self._add_cpu_reference_line_if_applicable(ax, df_plot, reference_cpu)

        plt.xticks(rotation=45, fontsize=9)
        ax.set_title(title if title else f"Compare Experiments on {y_col} vs {x_col}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fastest_runs(
        self,
        experiment_names: list,
        x_col: str = "InputSize",
        y_col: str = "Time [s]",
        title: str = None,
        logy: bool = True,
    ):
        df_plot = self.results[self.results["Experiment"].isin(experiment_names)].copy()
        if df_plot.empty:
            print(f"[WARN] No data found for {experiment_names}")
            return

        agg_cols = ["Experiment", x_col]
        df_plot = (
            df_plot.dropna(subset=[y_col])
            .sort_values(by=y_col)
            .groupby(agg_cols, as_index=False)
            .first()
        )

        df_plot = df_plot.sort_values(by=x_col).reset_index(drop=True)

        plt.figure()
        ax = sns.barplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue="Experiment",
            edgecolor="black",
        )
        if logy:
            ax.set_yscale("log")

        plt.xticks(rotation=45, fontsize=9)
        ax.set_title(title if title else "Fastest Runs")
        ax.set_ylabel(y_col)
        ax.set_xlabel(x_col)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
