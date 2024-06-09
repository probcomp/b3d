from typing import NamedTuple
from pathlib import Path


class ConfigClass(NamedTuple):
    args:dict
    seed:int 
    save_dir:Path
    print_matching:bool 
    verbose:bool 
    dump_dir:str 
    dump:str

Config = ConfigClass(
args = dict(),
seed = 0,
save_dir = Path(__file__).parent.parent / "results_json",
print_matching = True,
verbose = True ,
dump_dir = "/temp",
dump = 'txt' # 'csv', 'html', 'json', 'md', 'txt'
)

