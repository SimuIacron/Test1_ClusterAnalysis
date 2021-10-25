

from gbd_tool.gbd_api import GBD
import os

solvers = ['cadical_sc2020', 'duriansat', 'exmaple_padc_dl', 'exmaple_padc_dl_ovau_exp', 'exmaple_padc_dl_ovau_lin',
           'exmaple_psids_dl', 'kissat', 'kissat_sat', 'kissat_unsat', 'maple_scavel', 'maple_alluip_trail',
           'maple_lrb_vsids_2_init', 'maplecomsps_lrb_vsids_2', 'maple_scavel01', 'maple_scavel02', 'maple_dl_f2trc',
           'maplelcmdistchronobt_dl_v3', 'maple_f2trc', 'maple_f2trc_s', 'maple_cm_dist', 'maple_cm_dist_sattime2s',
           'maple_cm_dist_simp2', 'maple_cmused_dist', 'maple_mix', 'maple_simp', 'parafrost', 'parafrost_cbt',
           'pausat', 'relaxed', 'relaxed_newtech', 'relaxed_notimepara', 'slime', 'undominated_top16',
           'undominated_top24', 'undominated_top36', 'undominated', 'cadical_alluip', 'cadical_alluip_trail',
           'cadical_trail', 'cryptominisat_ccnr', 'cryptominisat_ccnr_lsids', 'cryptominisat_walksat',
           'exp_l_mld_cbt_dl', 'exp_v_lgb_mld_cbt_dl', 'exp_v_l_mld_cbt_dl', 'exp_v_mld_cbt_dl', 'glucose3',
           'upglucose_3_padc']


db_path = os.environ["DBPATH"] + "meta.db" + os.pathsep + \
          os.environ["DBPATH"] + "base.db" + os.pathsep + \
          os.environ["DBPATH"] + "gate.db" + os.pathsep + \
          os.environ["DBPATH"] + "sc2020.db"

with GBD(db_path) as gbd:
    base_return = gbd.query_search("competition_track = main_2020", [], solvers)
    print(base_return)

