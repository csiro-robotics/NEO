import time
import threading
from rich.console import Console
from rich.live import Live
from rich.table import Table
from enum import Enum
from typing import List, Union, Dict

CONSOLE_WIDTH = 120

class Status(Enum):
    INACTIVE = 0
    ACTIVE = 1
    FINISHED = 2
    PROCESSING = 3
    FAILED = 4

class PipelineProgress:
    def __init__(self, pipeline_elements: List[List[Union[Dict, str]]]):
        self.current_idx = 1
        self.console = Console(width=CONSOLE_WIDTH)
        self.live = Live(console=self.console, refresh_per_second=10)

        self.pipeline_elements = pipeline_elements
        self.pipeline = sorted(
            [element for sublist in pipeline_elements for element in sublist if isinstance(element, dict)],
            key=lambda x: x["idx"]
        )

        idx = 1
        for element in self.pipeline:
            if element["active"]:
                element["idx"] = idx
                element["status"] = Status.ACTIVE
                element["elapsed_time"] = 0  # Initialize elapsed_time to 0
                idx += 1
            else:
                element["idx"] = "-"
                element["status"] = Status.INACTIVE
                element["elapsed_time"] = 0  # Initialize elapsed_time to 0 for inactive elements
            del element["active"]

        # Initialize the time update thread
        self._update_thread = None
        self._running = False

    def _status(self, element: Dict, width: int = None) -> str:
        if width is None:
            width = len(element["name"])

        name = f"{element['idx']}.{element['name']}"

        if element["status"] == Status.ACTIVE:
            return f"{name:^{width}}"
        elif element["status"] == Status.INACTIVE:
            return f"[dim]{name:^{width}}[/dim]"
        elif element["status"] == Status.PROCESSING:
            elapsed_time = element['elapsed_time']
            elapsed_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time))
            return f"[orange1]{name:^{width}}[/orange1] - {elapsed_time_str}"
        elif element["status"] == Status.FINISHED:
            elapsed_time = element['elapsed_time']
            elapsed_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time))
            return f"[bold green]{name:^{width}}[/bold green] - {elapsed_time_str}"
        else:
            return f"{name:^{width}}"

    def start_step(self):
        for element in self.pipeline:
            if element["idx"] == self.current_idx and element["status"] == Status.ACTIVE:
                element["status"] = Status.PROCESSING
                element["start_time"] = time.time()
                break

        self.live.update(self._create_table())

    def _update_time(self):
        """Updates the time internally while processing each step."""
        while self._running:
            for element in self.pipeline:
                if element["status"] == Status.PROCESSING:
                    elapsed_time = time.time() - element["start_time"]
                    element["elapsed_time"] = elapsed_time
            self.live.update(self._create_table())
            time.sleep(0.1)  # Update every 0.1 seconds

    def end_step(self):
        for element in self.pipeline:
            if element["idx"] == self.current_idx and element["status"] == Status.PROCESSING:
                element["status"] = Status.FINISHED
                element["elapsed_time"] = time.time() - element["start_time"]
                break
        self.live.update(self._create_table())

    def __enter__(self):
        self.live.start()
        self._running = True
        # Start the time update thread
        self._update_thread = threading.Thread(target=self._update_time, daemon=True)
        self._update_thread.start()
        self.live.update(self._create_table())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._running = False
        if self._update_thread:
            self._update_thread.join()
        self.live.stop()

    def __iter__(self):
        """Wraps the iterable and updates progress."""
        for item in self.pipeline:
            yield item
            self.current_idx += 1  # Track the progress

    def _create_table(self):
        table = Table(show_header=False, box=None)
        columns_width = [0] * len(self.pipeline_elements[0])
        for row in range(len(self.pipeline_elements)):
            for column in range(len(self.pipeline_elements[row])):
                if isinstance(self.pipeline_elements[row][column], dict):
                    width = len(self.pipeline_elements[row][column]["name"])
                    if width > columns_width[column]:
                        columns_width[column] = width + len(str(self.pipeline_elements[row][column]["idx"])) + 1
                elif isinstance(self.pipeline_elements[row][column], str):
                    width = len(self.pipeline_elements[row][column])
                    if width > columns_width[column]:
                        columns_width[column] = width
                else:
                    pass

        for row in range(len(self.pipeline_elements)):
            print_row = []
            for column in range(len(self.pipeline_elements[row])):
                if isinstance(self.pipeline_elements[row][column], dict):
                    element = self.pipeline_elements[row][column]
                    row_to_append = self._status(element, width=columns_width[column])
                elif isinstance(self.pipeline_elements[row][column], str):
                    row_to_append = f"{self.pipeline_elements[row][column]:^{columns_width[column] + 2}}"
                else:
                    row_to_append = f"{'':^{columns_width[column]}}"

                print_row.append(row_to_append)

            table.add_row(*print_row)
        
        return table

if __name__ == "__main__":

    process_bags = {"name": "Process Bags", "active": True, "idx": 1}
    colmap = {"name": "COLMAP", "active": False, "idx": 2}
    extract_arm_movement = {"name": "Extract Arm Movement", "active": True, "idx": 3}
    ready_datasets = {"name": "Ready Datasets", "active": True, "idx": 4}
    
    pipeline_elements = [[process_bags, "→", colmap, "→", ready_datasets],
                         [None, "", extract_arm_movement, "↗", None]]

    with PipelineProgress(pipeline_elements) as progress:
        for item in progress:
            progress.start_step()
            
            time.sleep(5)  # Simulating processing for 5 seconds
            progress.end_step()
