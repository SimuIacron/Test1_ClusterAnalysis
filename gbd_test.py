import os

from gbd_tool.gbd_api import GBD
import gbd_tool.eval as gbd_eval

db_path = os.environ["DBPATH"] + "meta.db" + os.pathsep +\
    os.environ["DBPATH"] + "satzilla.db" + os.pathsep +\
    os.environ["DBPATH"] + "sc2020.db"

with GBD(db_path) as gbd:
    # gbd_eval.par2(gbd, "competition_track=main_2020", ["kissat_sat", "relaxed_newtech"], 5000, None)
    print(gbd.query_search("competition_track = main_2020", [], ["family"]))

