import os
from gbd_tool.gbd_api import GBD
import plotly.express as px
from sklearn.preprocessing import StandardScaler

import biclustering
import clustering
import evaluation
import feature_reduction
import scaling
import util

# path to the databases
db_path = os.environ["DBPATH"] + "meta.db" + os.pathsep + \
          os.environ["DBPATH"] + "base.db" + os.pathsep + \
          os.environ["DBPATH"] + "gate.db" + os.pathsep + \
          os.environ["DBPATH"] + "sc2020.db"

# value used to replace the timeout and failure value in solve times
timeout_value = 5000

gate_features = ['n_vars', 'n_gates', 'n_roots', 'n_none', 'n_generic', 'n_mono', 'n_and', 'n_or', 'n_triv', 'n_equiv',
                 'n_full', 'levels_mean', 'levels_variance', 'levels_min', 'levels_max', 'levels_entropy',
                 'levels_none_mean', 'levels_none_variance', 'levels_none_min', 'levels_none_max',
                 'levels_none_entropy',
                 'levels_generic_mean', 'levels_generic_variance', 'levels_generic_min', 'levels_generic_max',
                 'levels_generic_entropy', 'levels_mono_mean', 'levels_mono_variance', 'levels_mono_min',
                 'levels_mono_max',
                 'levels_mono_entropy', 'levels_and_mean', 'levels_and_variance', 'levels_and_min', 'levels_and_max',
                 'levels_and_entropy', 'levels_or_mean', 'levels_or_variance', 'levels_or_min', 'levels_or_max',
                 'levels_or_entropy', 'levels_triv_mean', 'levels_triv_variance', 'levels_triv_min', 'levels_triv_max',
                 'levels_triv_entropy', 'levels_equiv_mean', 'levels_equiv_variance', 'levels_equiv_min',
                 'levels_equiv_max',
                 'levels_equiv_entropy', 'levels_full_mean', 'levels_full_variance', 'levels_full_min',
                 'levels_full_max',
                 'levels_full_entropy', 'gate_features_runtime']

base_features = ['clauses', 'variables', 'clause_size_1', 'clause_size_2', 'clause_size_3', 'clause_size_4',
                 'clause_size_5', 'clause_size_6', 'clause_size_7', 'clause_size_8', 'clause_size_9', 'horn_clauses',
                 'inv_horn_clauses', 'positive_clauses', 'negative_clauses', 'horn_vars_mean', 'horn_vars_variance',
                 'horn_vars_min', 'horn_vars_max', 'horn_vars_entropy', 'inv_horn_vars_mean', 'inv_horn_vars_variance',
                 'inv_horn_vars_min', 'inv_horn_vars_max', 'inv_horn_vars_entropy', 'balance_clause_mean',
                 'balance_clause_variance', 'balance_clause_min', 'balance_clause_max', 'balance_clause_entropy',
                 'balance_vars_mean', 'balance_vars_variance', 'balance_vars_min', 'balance_vars_max',
                 'balance_vars_entropy', 'vcg_vdegrees_mean', 'vcg_vdegrees_variance', 'vcg_vdegrees_min',
                 'vcg_vdegrees_max', 'vcg_vdegrees_entropy', 'vcg_cdegrees_mean', 'vcg_cdegrees_variance',
                 'vcg_cdegrees_min', 'vcg_cdegrees_max', 'vcg_cdegrees_entropy', 'vg_degrees_mean',
                 'vg_degrees_variance', 'vg_degrees_min', 'vg_degrees_max', 'vg_degrees_entropy', 'cg_degrees_mean',
                 'cg_degrees_variance', 'cg_degrees_min', 'cg_degrees_max', 'cg_degrees_entropy',
                 'base_features_runtime']

solver_features = ['cadical_sc2020', 'duriansat', 'exmaple_padc_dl', 'exmaple_padc_dl_ovau_exp',
                   'exmaple_padc_dl_ovau_lin',
                   'exmaple_psids_dl', 'kissat', 'kissat_sat', 'kissat_unsat', 'maple_scavel', 'maple_alluip_trail',
                   'maple_lrb_vsids_2_init', 'maplecomsps_lrb_vsids_2', 'maple_scavel01', 'maple_scavel02',
                   'maple_dl_f2trc',
                   'maplelcmdistchronobt_dl_v3', 'maple_f2trc', 'maple_f2trc_s', 'maple_cm_dist',
                   'maple_cm_dist_sattime2s',
                   'maple_cm_dist_simp2', 'maple_cmused_dist', 'maple_mix', 'maple_simp', 'parafrost', 'parafrost_cbt',
                   'pausat', 'relaxed', 'relaxed_newtech', 'relaxed_notimepara', 'slime', 'undominated_top16',
                   'undominated_top24', 'undominated_top36', 'undominated', 'cadical_alluip', 'cadical_alluip_trail',
                   'cadical_trail', 'cryptominisat_ccnr', 'cryptominisat_ccnr_lsids', 'cryptominisat_walksat',
                   'exp_l_mld_cbt_dl', 'exp_v_lgb_mld_cbt_dl', 'exp_v_l_mld_cbt_dl', 'exp_v_mld_cbt_dl', 'glucose3',
                   'upglucose_3_padc']

with GBD(db_path) as gbd:
    print("Starting queries...")

    # make separate queries, because a max of 64 table joins are allowed in sql
    family_return = gbd.query_search("competition_track = main_2020", [], ["family"])
    base_return = gbd.query_search("competition_track = main_2020", [], base_features)
    gate_return = gbd.query_search("competition_track = main_2020", [], gate_features)
    solver_return = gbd.query_search("competition_track = main_2020", [], solver_features)
    print("Queries finished")
    print("Merge query results...")

    # merge two lists manually together

    # contains the instance values after calculations
    instances = []
    # contains the hashes of the instances in the same order of the instance list
    instance_hash = []

    if len(base_return) != len(gate_return) or len(base_return) != len(solver_return):
        raise AssertionError()

    gate_return_without_hash = [el[1:] for el in gate_return]
    base_return_without_hash = [el[1:] for el in base_return]
    solver_return_without_hash = [el[1:] for el in solver_return]
    family_return_without_hash = [el[1:] for el in family_return]
    for i in range(len(base_return)):
        # make sure all lists are ordered the same way
        if base_return[i][0] != gate_return[i][0] and base_return[i][0] != solver_return[i][0] and \
                base_return[i][0] != family_return[i][0]:
            raise AssertionError()
        instance_hash.append(base_return[i])
        instances.append(base_return_without_hash[i] + gate_return_without_hash[i])

    # list of all base and gate features of all instances
    instances_list = [list(i) for i in instances]

    for i in range(len(solver_return_without_hash)):
        solver_return_without_hash[i] = [timeout_value if (x == 'timeout' or x == 'failed') else float(x) for x in
                                         solver_return_without_hash[i]]

    # remove empty and memout keywords and replaces them with 0
    # TODO find out what the best replacement is, 0 could be wrong and give wrong cluster results
    features = base_features + gate_features
    for inst in instances_list:
        for i in range(len(inst)):
            if inst[i] == "empty" or inst[i] == "memout":
                inst[i] = 0
            else:
                inst[i] = float(inst[i])

    print("Merge finished")

    # scaling
    print("Start scaling...")

    instances_list_s = scaling.scaling(instances_list, algorithm='SCALEMINUSPLUS1')

    print("Scaling finished")

    # reduce dimensions
    reduced_instance_list = feature_reduction.feature_reduction(instances_list_s, algorithm="VARIANCE", features=None,
                                                                variance=0.8)
    print('Remaining features: ' + str(len(reduced_instance_list[0])))

    print("Starting clustering...")

    # biclustering
    # fit_data = biclustering.bicluster(reduced_instance_list, "SPECTRALBI")
    # fig = px.imshow(util.rotateNestedLists(fit_data))
    # fig.show()

    # clustering
    (clusters, yhat) = clustering.cluster(reduced_instance_list, "DBSCAN")
    print("Clustering finished")

    evaluation.clusters_scatter_plot(yhat, reduced_instance_list, solver_return_without_hash, solver_features)

    # calculate means and median for each cluster
    # for cluster in clusters:
    #     evaluation.cluster_family_amount(cluster, yhat, family_return_without_hash)
    #     evaluation.cluster_statistics(cluster, yhat, solver_return_without_hash, solver_features)

    evaluation.clusters_family_amount(clusters, yhat, family_return_without_hash)
    evaluation.clusters_timeout_amount(clusters, yhat, timeout_value, solver_return_without_hash)
