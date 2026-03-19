import pandas as pd

from Utils.utils import *
import Utils.graph_functions as grf
from Utils.label_split import split_common_actions




def main(log_path):
    # load the log
    unseg_log = pd.read_csv(log_path)

    COMMON_ACTIONS = grf.get_common_actions(unseg_log)
    split_log = split_common_actions(unseg_log, COMMON_ACTIONS, window_size=3, distance_threshold=3, method='distance')

    dfg = grf.discover_dfg(split_log)
    G_Directed = grf.get_Network_Graph(dfg, output_filename=f"Graph_Matrix_Directed.csv")
    G_Directed_Scored2 = grf.get_scored2_grpah_directed(G_Directed, output_filename=f"Graph_Matrix_UnDirected.csv")

    infomap_clusters = grf.infomap_clustering(G_Directed_Scored2, MRT=4.0)
  



if __name__ == "__main__":
    # start_time = time.time()

    log_path = "Routine_Logs/Synthetic/3R_Interleaved/unsegment_log1.csv"
    main(log_path)