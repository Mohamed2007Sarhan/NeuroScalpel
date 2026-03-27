from pathlib import Path

import yaml

# Use the file's own location to find globals.yml regardless of the CWD.
# The globals.yml lives one directory above util/ (i.e. in LyapLock/).
_GLOBALS_YML = Path(__file__).resolve().parent.parent / "globals.yml"
with open(str(_GLOBALS_YML), "r") as stream:
    data = yaml.safe_load(stream)


(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
