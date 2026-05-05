import os
import time

from ij import IJ
from java.io import File
from java.lang import Exception as java_exception
from java.lang import Runtime, System
from java.net import URL
from org.apache.commons.io import FileUtils

from src.ui.dialog import get_OK
from src.utils.combinatorics import pairwise
from src.utils.io import download_unzip
from src.utils.log import dlog
from src.utils.matrix import save_mat_to_tsplib
from src.utils.numeric import intr

_SOLVER_TIMEOUT_SECONDS = 60

_TSP_ROOT_URL = r"https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/cygwin"

_CONCORDE_EXE = "concorde.exe"
_CONCORDE_URL = _TSP_ROOT_URL + "/{}.gz".format(_CONCORDE_EXE)

_LINKERN_EXE = "linkern.exe"
_LINKERN_URL = _TSP_ROOT_URL + "/{}.gz".format(_LINKERN_EXE)

_CYGWINDLL_URL = (
    r"https://raw.githubusercontent.com/templiert/MagFinder/v1.0.0/cygwin1.dll"
)
_DOWNLOAD_MSG = (
    "Downloading {} to the Fiji plugins directory from \n \n{}\n \n"
    "The file is needed to compute the 'stage order' that minimizes stage travel"
    "\nand the serial order of the sections."
    "\n \nDo you agree? "
)


def _get_tsp_paths():
    plugin_folder = IJ.getDirectory("plugins")
    return (
        os.path.join(plugin_folder, "cygwin1.dll"),
        os.path.join(plugin_folder, _CONCORDE_EXE),
        os.path.join(plugin_folder, _LINKERN_EXE),
    )


def _tsp_solver_paths():
    """Returns (concorde_path, linkern_path) if all required files exist, else (None, None)."""
    cygwindll_path, concorde_path, linkern_path = _get_tsp_paths()
    if not all(
        os.path.isfile(p) for p in (cygwindll_path, concorde_path, linkern_path)
    ):
        return None, None
    return concorde_path, linkern_path


def ensure_tsp_solvers():
    """Downloads Concorde, Linkern, and cygwin1.dll to the Fiji plugins folder if missing.

    Windows-only. No-ops on other platforms.
    """
    if "windows" not in System.getProperty("os.name").lower():
        dlog("TSP solver is only supported on Windows.")
        return

    cygwindll_path, concorde_path, linkern_path = _get_tsp_paths()

    if not os.path.isfile(cygwindll_path):
        if get_OK(
            _DOWNLOAD_MSG.format("Windows-10 precompiled cygwin1.dll", _CYGWINDLL_URL),
            window_name="Download?",
        ):
            try:
                FileUtils.copyURLToFile(URL(_CYGWINDLL_URL), File(cygwindll_path))
            except (Exception, java_exception) as e:
                dlog("Failed to download cygwin1.dll due to {}".format(e))

    for path, url in zip([concorde_path, linkern_path], [_CONCORDE_URL, _LINKERN_URL]):
        if not os.path.isfile(path):
            if get_OK(
                _DOWNLOAD_MSG.format("solver for traveling salesman", url),
                window_name="Download?",
            ):
                download_unzip(url, path)

    if _tsp_solver_paths() == (None, None):
        dlog("TSP solver files missing - stage order will not be computed.")


def _parse_solution(solution_path, solver_path):
    """Parses a linkern or concorde solution file.

    Returns the order with the dummy city 0 removed and a -1 offset applied.
    """
    if _LINKERN_EXE in solver_path:
        with open(solution_path, "r") as f:
            order = [int(line.split(" ")[0]) for line in f.readlines()[1:]]
    elif _CONCORDE_EXE in solver_path:
        with open(solution_path, "r") as f:
            lines = f.readlines()
        order = [int(x) for x in lines[1].replace(" \n", "").split(" ")]
    else:
        raise ValueError("Unknown solver: {}".format(solver_path))
    # remove the dummy city 0 and apply a -1 offset
    order.remove(0)
    return [o - 1 for o in order]


def _log_order_costs(mat, order):
    """Logs the non-optimized (sequential) and optimized (TSP) total costs."""
    sequential_cost = sum(mat[idx][idx + 1] for idx in range(len(order) - 1))
    optimized_cost = sum(mat[o1][o2] for o1, o2 in pairwise(order))
    dlog("Total cost before ordering: {} (a.u.)".format(intr(sequential_cost)))
    dlog("Total cost after  ordering: {} (a.u.)".format(intr(optimized_cost)))


def compute_order_from_distance_matrix(mat, solver_path, root, solution_name=""):
    """Computes order from distance matrix.

    1. save the distance matrix in appropriate format for solver.
    2. remove any existing solution
    3. solve
    4. parse solution from file
    """
    solution_path = os.path.join(root, "solution_{}.txt".format(solution_name))
    if os.path.isfile(solution_path):
        os.remove(solution_path)

    tsplib_path = os.path.join(root, "TSPMat.tsp")
    save_mat_to_tsplib(mat, tsplib_path)

    cmd = '"{}" -o "{}" "{}"'.format(solver_path, solution_path, tsplib_path)
    Runtime.getRuntime().exec(cmd)

    solver_name = os.path.splitext(os.path.basename(solver_path))[0]
    msg = "Computing TSP solution with the {} solver...".format(solver_name)
    start_time = time.time()
    while not os.path.isfile(solution_path):
        if time.time() - start_time >= _SOLVER_TIMEOUT_SECONDS:
            raise RuntimeError(
                "TSP solver did not return within {}s; aborting.".format(
                    _SOLVER_TIMEOUT_SECONDS
                )
            )
        time.sleep(1)
        dlog(msg)
    time.sleep(1)  # wait for the file to be fully written

    try:
        order = _parse_solution(solution_path, solver_path)
        _log_order_costs(mat, order)
        return order
    finally:
        for path in (solution_path, tsplib_path):
            if os.path.isfile(path):
                os.remove(path)


def compute_tsp_order(pairwise_costs, root):
    if len(pairwise_costs) < 2:
        return []
    costs = [
        pairwise_costs[i][j]
        for i in range(len(pairwise_costs))
        for j in range(i + 1, len(pairwise_costs))
    ]
    # Solver hangs on uniform cost matrices: no signal to optimize on.
    if min(costs) == max(costs):
        dlog(
            "TSP cost matrix is degenerate, all pair costs equal; "
            "skipping order optimization."
        )
        return []
    concorde_path, linkern_path = _tsp_solver_paths()
    if concorde_path is None:
        return []
    solver_path = concorde_path if len(pairwise_costs) < 8 else linkern_path
    try:
        order = compute_order_from_distance_matrix(pairwise_costs, solver_path, root)
        dlog("The optimal order is: {}".format(order))
    except (Exception, java_exception) as e:
        dlog("We failed to compute the order:\n{}".format(e))
        return []
    return order
