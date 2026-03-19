import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import tempfile
import shutil

from sklearn.metrics import normalized_mutual_info_score

import networkx as nx
from infomap import Infomap
from itertools import combinations
from collections import defaultdict
import community as community_louvain

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


def plot_graph_with_edge_labels(G):
    """
    Properly plots a directed graph G with separate edge labels for each direction.
    """
    plt.figure(figsize=(32, 28))
    pos = nx.spring_layout(G, seed=39)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1600)
    nx.draw_networkx_labels(G, pos)

    # Draw edges with curves
    nx.draw_networkx_edges(
        G,
        pos,
        # connectionstyle='arc3,rad=0.2',
        arrows=True,
        arrowsize=20
    )

    # Prepare edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Draw all edge labels at once
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        label_pos=0.2,  # center along the curve
        font_size=10
    )

    plt.axis('off')
    plt.show()


def plot_and_save_adj_matrix(G, plot_matrix, csv_filename="adjacency_matrix.csv"):
    """
    Display and save the adjacency matrix of a directed graph G.
    """

    # Get adjacency matrix (weighted)
    adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()

    # Create a pandas DataFrame with node labels
    adj_matrix_df = pd.DataFrame(adj_matrix, index=G.nodes(), columns=G.nodes())

    # 1. Show the matrix
    # print("Adjacency Matrix:\n")
    # print(adj_matrix_df)

    # 2. Plot the matrix as a heatmap
    if plot_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix_df, annot=True, cmap='YlGnBu', cbar=True, linewidths=0.5, linecolor='black')
        plt.title("Adjacency Matrix Heatmap")
        plt.show()

    # 3. Save the matrix to CSV
    output_dir = "out/csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    adj_matrix_df.to_csv(f"{output_dir}/{csv_filename}")
    print(f"Adjacency matrix saved successfully to '{csv_filename}'.")



def plot_graph_with_clusters(G, partition, method="Louvain"):
    """
    Plots the directed graph G with nodes colored by their cluster (from either Louvain, Infomap, or OSLOM).
    
    Parameters:
        G (networkx.Graph): The graph to plot.
        partition (dict): A dictionary where keys are nodes and values are cluster IDs.
        method (str): The method used for clustering ('Louvain', 'Infomap', or 'OSLOM').
    """
    if method == "Infomap":
        # Infomap uses integer node IDs, so we need to map nodes in G to Infomap's numeric IDs
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        # Map cluster labels from Infomap partition back to original node labels
        cluster_colors = [partition[node_mapping[node]] for node in G.nodes()]
    elif method == "Louvain":
        # Louvain clustering already has node labels as keys
        cluster_colors = [partition[node] for node in G.nodes()]
    elif method == "OSLOM":
        # OSLOM partition already has node labels as keys (primary cluster assignment)
        cluster_colors = [partition.get(node, 0) for node in G.nodes()]

    # Get layout for node positions
    pos = nx.spring_layout(G, seed=39)

    # Set up the plot size
    plt.figure(figsize=(32, 28))

    # Draw nodes with colors representing clusters
    nx.draw_networkx_nodes(G, pos, node_size=1600, cmap=plt.cm.jet, node_color=cluster_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges with curves
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=20
    )

    # Prepare edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Draw all edge labels at once
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        label_pos=0.2,  # center along the curve
        font_size=10
    )

    # Title
    plt.title(f"Graph Clusters - {method} Community Detection")

    # Display the plot
    plt.axis('off')
    plt.show()


def discover_dfg(event_log):
    
    # Drop ground-truth label
    if "routine_type" in event_log.columns:
        event_log = event_log.drop(columns=["routine_type"])
    
    event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
    event_log = log_converter.apply(event_log)
    
    # Discover DFG
    dfg = dfg_discovery.apply(event_log)
    return dfg


def get_Network_Graph(dfg, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed.csv"):
    
    # Convert DFG to a NetworkX graph
    G = nx.DiGraph()
    
    for (src, tgt), weight in dfg.items():
        G.add_edge(src, tgt, weight=weight)

    plot_graph_with_edge_labels(G) if plot_graph else None
    plot_and_save_adj_matrix(G, plot_matrix, csv_filename=output_filename)
    return G


def to_Undirected(G, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_UnDirected.csv"):
    # Convert to undirected graph and take the maximum weight between edges
    G_undirected = nx.Graph()
    
    # Iterate over directed edges and add them to the undirected graph
    for u, v, data in G.edges(data=True):
        if G_undirected.has_edge(u, v):
            # If edge exists, keep the maximum weight
            G_undirected[u][v]['weight'] = max(G_undirected[u][v]['weight'], data['weight'])
        else:
            # If edge doesn't exist, just add it
            G_undirected.add_edge(u, v, weight=data['weight'])
    
    plot_graph_with_edge_labels(G_undirected) if plot_graph else None
    plot_and_save_adj_matrix(G_undirected, plot_matrix, csv_filename=output_filename)

    return G_undirected


# === Calculate Dependency Score for each edge ===
def calculate_dependency_score1(G_directed):
    dependency_scores = {}

    for (A, B) in G_directed.edges():
        count_A_to_B = G_directed[A][B]["weight"]
        count_B_to_A = G_directed[B][A]["weight"] if G_directed.has_edge(B, A) else 0

        numerator = abs(count_A_to_B - count_B_to_A)
        denominator = count_A_to_B + count_B_to_A + 1  # Add 1 only here

        score = numerator / denominator
        dependency_scores[(A, B)] = score

    return dependency_scores


# === Calculate Dependency Score for each edge ===
def calculate_dependency_score2(G_directed):
    # Initialize a dictionary to store dependency scores
    dependency_scores = {}

    # Initialize a dictionary to store the total outgoing counts for each activity
    total_outgoing_count = {node: 0 for node in G_directed.nodes()}
    
    # Calculate total outgoing counts for each activity (node)
    for u, v, data in G_directed.edges(data=True):
        total_outgoing_count[u] += data['weight']  # Add the weight of the outgoing edge from u

    # Calculate the dependency score for each directed edge (A -> B)
    for (A, B) in G_directed.edges():
        # Count of A -> B (the weight of the edge from A to B)
        count_A_to_B = G_directed[A][B]["weight"]
        
        # Total outgoing count from A (sum of all outgoing edges from A)
        total_A_outgoing = total_outgoing_count[A]

        # Calculate the dependency score as the ratio of count_A_to_B to total_A_outgoing
        if total_A_outgoing == 0:
            # If total outgoing count from A is 0, avoid division by zero
            score = 0
        else:
            score = count_A_to_B / total_A_outgoing

        # Store the dependency score for the edge (A -> B)
        dependency_scores[(A, B)] = score

    return dependency_scores
    

# === Remove parallel activities (dependency score < 0.6) ===
def remove_parallel_activities(G_directed, dependency_scores, threshold):
    to_remove = [edge for edge, score in dependency_scores.items() if score < threshold]
    # print(f"\nRemoved Edges: {to_remove}")
    G_directed.remove_edges_from(to_remove)
    return G_directed


# === Replace the edge weights with the dependency score ===
def update_graph_with_dependency_score(G_directed, dependency_scores):
    # Update graph with dependency score as the new edge weight
    for (A, B), score in dependency_scores.items():
        if G_directed.has_edge(A, B):
            G_directed[A][B]["weight"] = score  # Update edge weight to dependency score
    return G_directed


def get_scored1_grpah_directed(G, threshold=0.6, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed_Score1.csv"):
    # creating the copy of the original object
    G_copy = G.copy()

    # Calculate dependency scores

    dependency_scores = calculate_dependency_score1(G_copy)
    # print(dependency_scores)
    G_directed_truncated = remove_parallel_activities(G_copy, dependency_scores, threshold)
    
    # Replace the edge weights with the dependency score in the truncated graph
    G_directed_updated = update_graph_with_dependency_score(G_directed_truncated, dependency_scores)
    
    plot_graph_with_edge_labels(G_directed_updated) if plot_graph else None
    plot_and_save_adj_matrix(G_directed_updated, plot_matrix, csv_filename=output_filename)

    return G_directed_updated


def get_scored2_grpah_directed(G, threshold=0.1, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed_Score2.csv"):
    # creating the copy of the original object
    G_copy = G.copy()

    # Calculate dependency scores
    dependency_scores = calculate_dependency_score2(G_copy)
    # print(dependency_scores)
    G_directed_truncated = remove_parallel_activities(G_copy, dependency_scores, threshold)
    
    # Replace the edge weights with the dependency score in the truncated graph
    G_directed_updated = update_graph_with_dependency_score(G_directed_truncated, dependency_scores)
    
    plot_graph_with_edge_labels(G_directed_updated) if plot_graph else None
    plot_and_save_adj_matrix(G_directed_updated, plot_matrix, csv_filename=output_filename)

    return G_directed_updated


def check_cluster_quality(final_clusters, min_actions=4):
    """
    Check cluster quality by minimum number of actions.

    Parameters:
        final_clusters (dict): Mapping {cluster_id: [actions...]} (e.g., output of infomap_clustering).
        min_actions (int): Clusters must have at least this many actions (default: 4).

    Returns:
        dict: Clusters that pass the quality check (same cluster_ids).
    """
    if final_clusters is None:
        return {}

    return {cluster_id: actions for cluster_id, actions in final_clusters.items() if len(actions) >= min_actions}



def get_oprimal_modularity(G_undirected, gamma_min=0.1, gamma_max=3.0, values=20, plot_graph=False):
    gamma_values = np.linspace(gamma_min, gamma_max, values)
    num_runs = 5
    num_random_graphs = 10
    
    stability_scores = []
    modularity_real = []
    modularity_random = []
    community_counts = []
    
    for gamma in gamma_values:
        partitions = []
        modularities = []
    
        # print(f"=== Louvain Clustering for γ = {gamma:.2f} ===")
    
        for run in range(num_runs):
            partition = community_louvain.best_partition(G_undirected, resolution=gamma, weight='weight')
            partitions.append(partition)
            mod = community_louvain.modularity(partition, G_undirected, weight='weight')
            modularities.append(mod)
    
            # Group by cluster
            clustered_actions = defaultdict(list)
            for activity_name, cluster in partition.items():
                clustered_actions[cluster].append(activity_name)
    
            # # Print nicely
            # print(f"\nRun {run+1}:")
            # for cluster_num in sorted(clustered_actions):
            #     for activity in sorted(clustered_actions[cluster_num]):
            #         print(f"Gamma: {gamma:.2f}, Cluster: {cluster_num}, Activity: {activity}")
    
        # Stability (approximated using NMI)
        all_nmi = []
        for p1, p2 in combinations(partitions, 2):
            labels1 = list(p1.values())
            labels2 = list(p2.values())
            nmi = normalized_mutual_info_score(labels1, labels2)
            all_nmi.append(nmi)
        stability_scores.append(np.mean(all_nmi))
    
        modularity_real.append(np.mean(modularities))
        community_counts.append(np.mean([len(set(p.values())) for p in partitions]))
    
        # Random modularity
        random_modularities = []
        for _ in range(num_random_graphs):
            rand_G = nx.configuration_model([d for _, d in G_undirected.degree()])
            rand_G = nx.Graph(rand_G)
            rand_G.remove_edges_from(nx.selfloop_edges(rand_G))
            rand_partition = community_louvain.best_partition(rand_G, resolution=gamma)
            rand_mod = community_louvain.modularity(rand_partition, rand_G)
            random_modularities.append(rand_mod)
        modularity_random.append(np.mean(random_modularities))

    if plot_graph:
        # Plotting
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(gamma_values, stability_scores, marker='o')
        plt.title("A: Stability (1 - VI via NMI)")
        plt.xlabel("Gamma")
        plt.ylabel("Stability (NMI)")
        
        plt.subplot(1, 3, 2)
        plt.plot(gamma_values, modularity_real, marker='o', label="Real")
        plt.plot(gamma_values, modularity_random, marker='x', label="Random")
        plt.title("B: Modularity Comparison")
        plt.xlabel("Gamma")
        plt.ylabel("Modularity")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(gamma_values, community_counts, marker='o')
        plt.title("C: Number of Communities")
        plt.xlabel("Gamma")
        plt.ylabel("Avg Communities")
        
        plt.tight_layout()
        plt.show()

    # Get the optimal gamma based on modularity (real modularity)
    optimal_gamma_modularity = gamma_values[np.argmax(modularity_real)]
    print(f"Optimal Gamma based on Modularity: {optimal_gamma_modularity:.2f}")
    
    # Get the optimal gamma based on stability (NMI)
    optimal_gamma_stability = gamma_values[np.argmax(stability_scores)]
    print(f"Optimal Gamma based on Stability: {optimal_gamma_stability:.2f}")
    
    return optimal_gamma_modularity, optimal_gamma_stability


def louvin_Clustering(G_undirected, optimal_gamma, document, plot_graph=False):
    # 🎯 Apply Louvain using the selected optimal gamma (e.g., 1.5)
    final_partition = community_louvain.best_partition(G_undirected, resolution=optimal_gamma, weight='weight')
    
    plot_graph_with_clusters(G_undirected, final_partition) if plot_graph else None   

    final_clusters = defaultdict(list)

    for node, cluster in final_partition.items():
        final_clusters[cluster].append(node)
    
    # 🖨️ Print final community assignments
    print(f"\n=== Final Louvain Clustering for γ = {optimal_gamma} ===")
    for cluster_id in sorted(final_clusters):
        print(f"\nCluster {cluster_id}:")
        document.add_heading(f"\nCluster {cluster_id}:", level=6)
        for activity in sorted(final_clusters[cluster_id]):
            print(f" - {activity}")
            document.add_paragraph(f" - {activity}")
            
    return final_clusters, document


def display_cluster(infomap_partition, node_mapping, document):
    # Print clusters
    print("\n🔍 Infomap Clustering (Directed):")
    
    cluster_activity_pairs = []
    for node, cluster in infomap_partition.items():
        activity_name = list(node_mapping.keys())[list(node_mapping.values()).index(node)]
        cluster_activity_pairs.append((cluster, activity_name))
    
    # Sort the list by cluster number
    cluster_activity_pairs.sort()
    
    # Print sorted results
    for cluster, activity in cluster_activity_pairs:
        print(f"Cluster {cluster}: {activity}")
        document.add_paragraph(f"Cluster {cluster}: {activity}")
    return document

    
def infomap_clustering(G, plot_graph=False, MRT=None):
    """
    Run Infomap clustering exactly once using the provided MRT value.

    If MRT is None, defaults to 1.0.
    """
    used_MRT = MRT if MRT is not None else 1.0

    params = f"--directed --two-level --num-trials 20 --markov-time {used_MRT} --silent"
    im = Infomap(params)

    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        im.add_link(node_to_id[u], node_to_id[v], weight)

    im.run()

    cluster_assignments = {id_to_node[node.node_id]: node.module_id for node in im.nodes}
    final_clusters = defaultdict(list)
    for node, cluster_id in cluster_assignments.items():
        final_clusters[cluster_id].append(node)
    final_clusters = check_cluster_quality(final_clusters)

    # Optional: plot the clustered graph
    if plot_graph:
        plot_graph_with_clusters(G, cluster_assignments, method="Infomap")

    # Display results in console and Word document
    print("\n=== Final Infomap Clustering ===")
    print(f"Used MRT: {used_MRT}")
    for cluster_id in sorted(final_clusters):
        print(f"\nCluster {cluster_id}:")
        # document.add_heading(f"\nCluster {cluster_id}:", level=6)
        for activity in sorted(final_clusters[cluster_id]):
            print(f" - {activity}")
            # document.add_paragraph(f" - {activity}")

    
    return final_clusters


def get_common_actions(log):
    # log = pd.read_csv(input_csv_path)
    # Count frequency of each action
    action_freq = log["concept:name"].value_counts()
    
    freq_df = action_freq.reset_index()
    freq_df.columns = ["action", "frequency"]
    
    # print(freq_df.head())
    q1 = freq_df["frequency"].quantile(0.25)
    q2 = freq_df["frequency"].quantile(0.50)   # Median
    q3 = freq_df["frequency"].quantile(0.75)

    return freq_df.loc[freq_df["frequency"] > q1, "action"].tolist()


def export_graph_to_oslom_format(G, output_file="network.dat", weighted=True):
    """
    Export a NetworkX directed graph to OSLOM format (.dat file) and create a mapping file.
    
    This function creates:
    1. A .dat file for OSLOM clustering
    2. A mapping file (node_mapping.txt) that shows action names and their integer IDs
    
    After exporting, you can run OSLOM manually via terminal.
    
    Parameters:
        G (networkx.DiGraph): The directed graph to export.
        output_file (str): Path to the output .dat file (default: "network.dat").
        weighted (bool): Whether the graph is weighted (default: True).
                         If False, all edge weights will be set to 1.0.
    
    Returns:
        tuple: (path_to_dat_file, path_to_mapping_file)
    
    Example OSLOM commands to run after export:
        # For 2-3 communities (tuned parameters):
        ./oslom_dir -f network.dat -w -time 0.1 -infomap 1 -copra 1 -louvain 1
        
        # Alternative (fewer communities):
        ./oslom_dir -f network.dat -w -time 0.2 -infomap 1 -copra 1 -louvain 1
    """
    # Create node mapping (OSLOM requires integer node IDs, 0-indexed)
    node_list = list(G.nodes())
    node_to_id = {node: idx for idx, node in enumerate(node_list)}
    id_to_node = {idx: node for node, idx in node_to_id.items()}
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate mapping file name
    base_name = os.path.splitext(output_file)[0]
    mapping_file = f"{base_name}_node_mapping.txt"
    
    # Write edge list file for OSLOM (format: source target weight)
    # OSLOM expects: each line is "source target weight" (space-separated)
    with open(output_file, 'w') as f:
        # Write edges: source target weight (using integer IDs, 0-indexed)
        for u, v, data in G.edges(data=True):
            if weighted:
                weight = data.get("weight", 1.0)
            else:
                weight = 1.0
            # OSLOM expects integer node IDs (0-indexed)
            f.write(f"{node_to_id[u]} {node_to_id[v]} {weight}\n")
    
    # Write mapping file: ID -> Action Name
    with open(mapping_file, 'w') as f:
        f.write("OSLOM Node ID -> Action Name Mapping\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'ID':<10} {'Action Name'}\n")
        f.write("-" * 50 + "\n")
        for node_id in sorted(id_to_node.keys()):
            action_name = id_to_node[node_id]
            f.write(f"{node_id:<10} {action_name}\n")
    
    # Check network statistics
    num_nodes = len(node_list)
    num_edges = G.number_of_edges()
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    
    print(f"Graph exported to OSLOM format: {output_file}")
    print(f"Node mapping saved to: {mapping_file}")
    print(f"Number of nodes (actions): {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    
    # Warn if network is too sparse
    if avg_degree < 2.0:
        print(f"\n⚠️  WARNING: Network is very sparse (avg degree {avg_degree:.2f} < 2.0).")
        print(f"   OSLOM may struggle to find meaningful clusters.")
        print(f"   Consider: 1) Check edge weights are significant, 2) Verify network structure")
    print(f"\nTo run OSLOM clustering for 2-3 communities, use:")
    print(f"\n⚠️  IMPORTANT: If you get 'not found' errors for infomap_dir or louvain_script,")
    print(f"   run OSLOM WITHOUT those parameters (set them to 0).")
    print(f"\n  # Option 1 (RECOMMENDED - without infomap/louvain dependencies):")
    if weighted:
        print(f"  ./oslom_dir -f {output_file} -w -time 1.0 -infomap 0 -copra 0 -louvain 0")
    else:
        print(f"  ./oslom_dir -f {output_file} -uw -time 1.0 -infomap 0 -copra 0 -louvain 0")
    print(f"\n  # Option 2 (if Option 1 gives too many communities):")
    if weighted:
        print(f"  ./oslom_dir -f {output_file} -w -time 2.0 -infomap 0 -copra 0 -louvain 0")
    else:
        print(f"  ./oslom_dir -f {output_file} -uw -time 2.0 -infomap 0 -copra 0 -louvain 0")
    print(f"\n  # Option 3 (if still too many, try even higher):")
    if weighted:
        print(f"  ./oslom_dir -f {output_file} -w -time 5.0 -infomap 0 -copra 0 -louvain 0")
    else:
        print(f"  ./oslom_dir -f {output_file} -uw -time 5.0 -infomap 0 -copra 0 -louvain 0")
    print(f"\n  # Option 4 (if still getting too many clusters):")
    if weighted:
        print(f"  ./oslom_dir -f {output_file} -w -time 10.0 -infomap 0 -copra 0 -louvain 0")
    else:
        print(f"  ./oslom_dir -f {output_file} -uw -time 10.0 -infomap 0 -copra 0 -louvain 0")
    print(f"\n  # Alternative: Try with COPRA only (if COPRA works):")
    if weighted:
        print(f"  ./oslom_dir -f {output_file} -w -time 1.0 -infomap 0 -copra 1 -louvain 0")
    else:
        print(f"  ./oslom_dir -f {output_file} -uw -time 1.0 -infomap 0 -copra 1 -louvain 0")
    print(f"\nNote: Higher -time values create FEWER, LARGER communities.")
    print(f"      If you got 24 clusters (one per node), you need MUCH higher -time values.")
    print(f"      If you get 'bad group' errors, your network might be too sparse.")
    print(f"      Check your edge weights - they should be significant (not too small).")
    
    return output_file, mapping_file


def parse_oslom_output(clusters_file, mapping_file, document=None):
    """
    Parse OSLOM output clusters file and map back to action names.
    
    OSLOM outputs clusters in various formats. This function parses the cluster file
    and maps integer node IDs back to action names using the mapping file.
    
    Parameters:
        clusters_file (str): Path to OSLOM clusters output file (e.g., "clusters-network1.dat")
        mapping_file (str): Path to the node mapping file (e.g., "network_node_mapping.txt")
        document: Optional Word document object to add results to.
    
    Returns:
        dict: Dictionary mapping cluster_id to list of action names.
    """
    # Read mapping file to get ID -> Action Name mapping
    id_to_node = {}
    try:
        with open(mapping_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines (first 3-4 lines typically)
            for line in lines[3:]:  # Skip header
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('='):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            node_id = int(parts[0])
                            action_name = ' '.join(parts[1:])  # Handle action names with spaces
                            id_to_node[node_id] = action_name
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found. Using node IDs as names.")
        # Create a fallback mapping
        id_to_node = {i: f"Node_{i}" for i in range(24)}
    
    # Parse clusters file
    # OSLOM format can vary, but typically:
    # Format 1: Each line is "node_id cluster_id"
    # Format 2: Each line is just "node_id" (in cluster order)
    # Format 3: Multiple lines per cluster with node IDs
    
    clusters = defaultdict(list)
    node_to_cluster = {}
    
    try:
        with open(clusters_file, 'r') as f:
            cluster_id = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        node_id = int(parts[0])
                        # If there's a second number, it might be cluster ID
                        if len(parts) >= 2:
                            try:
                                cluster_id = int(parts[1])
                            except ValueError:
                                pass
                        
                        # Map to action name
                        if node_id in id_to_node:
                            action_name = id_to_node[node_id]
                            clusters[cluster_id].append(action_name)
                            node_to_cluster[action_name] = cluster_id
                        else:
                            # Use node ID if mapping not found
                            clusters[cluster_id].append(f"Node_{node_id}")
                            node_to_cluster[f"Node_{node_id}"] = cluster_id
                        
                        # If no explicit cluster ID in line, increment for next line
                        if len(parts) == 1:
                            cluster_id += 1
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: Clusters file {clusters_file} not found.")
        return {}
    
    # Print results
    print("\n" + "=" * 60)
    print("OSLOM Clustering Results")
    print("=" * 60)
    print(f"Total clusters found: {len(clusters)}")
    print(f"Total actions: {sum(len(nodes) for nodes in clusters.values())}")
    
    if len(clusters) == len(id_to_node):
        print("\n⚠️  WARNING: Each action is in its own cluster!")
        print("   This means no real clustering occurred.")
        print("   You need to adjust OSLOM parameters to get 2-3 clusters.")
        print("\n   Try these commands with higher -time values:")
        print("   ./oslom_dir -f network.dat -w -time 1.0 -infomap 1 -copra 1 -louvain 1")
        print("   ./oslom_dir -f network.dat -w -time 2.0 -infomap 1 -copra 1 -louvain 1")
        print("   ./oslom_dir -f network.dat -w -time 5.0 -infomap 1 -copra 1 -louvain 1")
    
    print("\nCluster Assignments:")
    print("-" * 60)
    for cluster_id in sorted(clusters.keys()):
        actions = clusters[cluster_id]
        print(f"\nCluster {cluster_id} ({len(actions)} actions):")
        for action in sorted(actions):
            print(f"  - {action}")
    
    # Add to document if provided
    if document:
        document.add_heading("\nOSLOM Clustering Results", level=5)
        document.add_paragraph(f"Total clusters: {len(clusters)}")
        if len(clusters) == len(id_to_node):
            document.add_paragraph("⚠️  WARNING: Each action is in its own cluster! No real clustering occurred.")
        for cluster_id in sorted(clusters.keys()):
            document.add_heading(f"\nCluster {cluster_id}:", level=6)
            for action in sorted(clusters[cluster_id]):
                document.add_paragraph(f"  - {action}")
    
    return clusters


def oslom_clustering(G, document, plot_graph=False, oslom_executable="oslom_dir", p_value=0.1, cp_value=0.5, 
                     time_param=0.005, infomap_param=None, copra_param=None, louvain_param=None):
    """
    Perform OSLOM clustering on a directed graph G.
    OSLOM supports overlapping communities, allowing nodes to belong to multiple clusters.
    
    IMPORTANT: Installation Requirements:
    - You MUST install the OSLOM software separately (even if you install python-oslom-runner via pip).
    - The pip package is just a wrapper; it does NOT include the OSLOM executable.
    
    SIMPLE INSTALLATION METHODS (choose one):
    
    Method 1: Download Pre-compiled Binary (Easiest)
    - Visit: http://www.oslom.org/
    - Download the pre-compiled binary for Linux
    - Extract and use the oslom_dir executable directly
    
    Method 2: Original OSLOM (Simpler than OSLOM2)
    - Download from: http://www.oslom.org/ (beta version 2.4 or earlier)
    - Extract the archive
    - Compile: g++ -o oslom_dir ./main_directed.cpp -O3 -Wall
    - Note: Original OSLOM has fewer dependencies than OSLOM2
    
    Method 3: Use Python Runner (if available)
    - pip install oslom-runner (still requires OSLOM executable separately)
    
    - Make sure oslom_dir is in your PATH or provide the full path in oslom_executable parameter.
    
    Parameters:
        G (networkx.DiGraph): The directed graph to cluster.
        document: Word document object for output (similar to infomap_clustering).
        plot_graph (bool): Whether to plot the clustered graph.
        oslom_executable (str): Path to OSLOM executable for directed graphs (default: "oslom_dir").
                                "oslom_dir" is the OSLOM executable name for directed networks.
                                Can be:
                                - Command name if OSLOM is in your PATH (e.g., "oslom_dir" or "oslom_dir.exe")
                                - Full path to executable (e.g., "C:/oslom/oslom_dir" or "/usr/local/bin/oslom_dir")
        p_value (float): P-value threshold for community detection (default: 0.1).
        cp_value (float): Coverage parameter affecting community sizes (default: 0.5).
        time_param (float): Time parameter for OSLOM (default: 0.005). Set to None to skip.
        infomap_param (int): Infomap parameter for OSLOM (default: None). Set to None to skip.
        copra_param (int): COPRA parameter for OSLOM (default: None). Set to None to skip.
        louvain_param (int): Louvain parameter for OSLOM (default: None). Set to None to skip.
    
    Returns:
        final_clusters (dict): Dictionary mapping cluster_id to list of nodes.
                               Note: OSLOM supports overlapping communities, so nodes may appear
                               in multiple clusters. This dict shows all cluster memberships.
        document: Updated document with clustering results.
    """
    # Create temporary directory for OSLOM input/output
    temp_dir = tempfile.mkdtemp()
    edge_file = os.path.join(temp_dir, "network.dat")
    
    try:
        # Create node mapping (OSLOM requires integer node IDs)
        node_list = list(G.nodes())
        node_to_id = {node: idx for idx, node in enumerate(node_list)}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        
        # Write edge list file for OSLOM (format: source target weight)
        # OSLOM expects: each line is "source target weight" (space-separated)
        with open(edge_file, 'w') as f:
            # Write edges: source target weight (using integer IDs, 0-indexed)
            for u, v, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                # OSLOM expects integer node IDs (0-indexed)
                f.write(f"{node_to_id[u]} {node_to_id[v]} {weight}\n")
        
        # Run OSLOM for directed networks
        # OSLOM parameters:
        # -p: p-value threshold
        # -cp: coverage parameter
        # -w: weighted network
        oslom_output_dir = os.path.join(temp_dir, "oslom_output")
        os.makedirs(oslom_output_dir, exist_ok=True)
        
        # Change to temp directory for OSLOM execution
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Run OSLOM command
            # OSLOM syntax: oslom_dir -f <filename> -w/-uw [-time <value>] [-infomap <value>] [-copra <value>] [-louvain <value>]
            # OSLOM requires either -w (weighted) or -uw (unweighted) flag
            
            # Ensure executable path is properly quoted if it contains spaces
            # subprocess.run handles this automatically when using list format
            cmd = [
                oslom_executable,
                "-f", "network.dat",
                "-w"  # weighted network (required flag)
            ]
            
            # Add optional parameters if provided
            if time_param is not None:
                cmd.extend(["-time", str(time_param)])
            if infomap_param is not None:
                cmd.extend(["-infomap", str(infomap_param)])
            if copra_param is not None:
                cmd.extend(["-copra", str(copra_param)])
            if louvain_param is not None:
                cmd.extend(["-louvain", str(louvain_param)])
            
            print(f"Running OSLOM command: {' '.join(cmd)}")
            print(f"Working directory: {temp_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            
            # Print output for debugging
            if result.stdout:
                print(f"OSLOM stdout: {result.stdout[:500]}")  # First 500 chars
            if result.stderr:
                print(f"OSLOM stderr: {result.stderr[:500]}")  # First 500 chars
            
            if result.returncode != 0:
                print(f"OSLOM warning/error (return code {result.returncode}):")
                print(f"Full stderr: {result.stderr}")
                # Continue anyway as OSLOM may still produce output despite non-zero return code
            
        finally:
            os.chdir(original_dir)
        
        # Parse OSLOM output
        # OSLOM creates a directory with the network name and outputs communities
        # Look for community files (typically in a subdirectory)
        community_files = []
        
        # Check for OSLOM output directory structure
        # OSLOM typically creates: network.dat_oslom_dir/ or network_oslom_dir/
        possible_output_dirs = [
            os.path.join(temp_dir, "network.dat_oslom_dir"),
            os.path.join(temp_dir, "network_oslom_dir"),
            os.path.join(temp_dir, "network.dat_oslom"),
            temp_dir
        ]
        
        # Also check subdirectories recursively
        def find_community_files(directory):
            files = []
            if not os.path.exists(directory):
                return files
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    # OSLOM output files: tp (two-level partition), or files with "community" in name
                    if filename.startswith("tp") or "community" in filename.lower() or filename.endswith(".dat"):
                        filepath = os.path.join(root, filename)
                        # Skip the input network file
                        if "network.dat" not in filename or "oslom" in root:
                            files.append(filepath)
            return files
        
        for output_dir in possible_output_dirs:
            found_files = find_community_files(output_dir)
            community_files.extend(found_files)
        
        # Remove duplicates
        community_files = list(set(community_files))
        
        # Parse community files
        # OSLOM format: each line contains nodes in a community (space-separated)
        final_clusters = defaultdict(list)
        cluster_assignments = {}  # For plotting: node -> primary cluster (first cluster found)
        overlapping_memberships = defaultdict(set)  # Track all cluster memberships per node
        
        if community_files:
            cluster_id = 0
            for comm_file in sorted(community_files):
                try:
                    with open(comm_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            
                            # OSLOM format: space-separated node IDs (integers)
                            # Try to parse as integers
                            parts = line.split()
                            node_ids = []
                            for part in parts:
                                try:
                                    node_id = int(part)
                                    node_ids.append(node_id)
                                except ValueError:
                                    # Skip non-integer parts
                                    continue
                            
                            if node_ids:
                                # Map back to original node names and build clusters
                                # Use same pattern as Infomap: append nodes to cluster lists
                                for node_id in node_ids:
                                    if node_id in id_to_node:
                                        original_node = id_to_node[node_id]
                                        final_clusters[cluster_id].append(original_node)
                                        overlapping_memberships[original_node].add(cluster_id)
                                        # For plotting, assign to first cluster found
                                        if original_node not in cluster_assignments:
                                            cluster_assignments[original_node] = cluster_id
                                
                                if final_clusters[cluster_id]:  # Only increment if cluster has nodes
                                    cluster_id += 1
                except Exception as e:
                    print(f"Warning: Could not parse community file {comm_file}: {e}")
                    continue
        
        # If no communities found, create a fallback (each node in its own cluster)
        if not final_clusters:
            print("Warning: No communities found in OSLOM output. Creating fallback clusters.")
            for idx, node in enumerate(node_list):
                final_clusters[idx] = [node]
                cluster_assignments[node] = idx
        
        # Optional: plot the clustered graph
        if plot_graph:
            # For overlapping communities, we'll plot with primary cluster assignment
            plot_graph_with_clusters(G, cluster_assignments, method="OSLOM")
        
        # Display results in console and Word document
        print("\n=== Final OSLOM Clustering (Directed) ===")
        print(f"P-value threshold: {p_value}, Coverage parameter: {cp_value}")
        
        # Check for overlapping nodes
        overlapping_nodes = {node: clusters for node, clusters in overlapping_memberships.items() 
                            if len(clusters) > 1}
        
        if overlapping_nodes:
            print(f"\nNote: {len(overlapping_nodes)} nodes belong to multiple clusters (overlapping communities):")
            for node, clusters in sorted(overlapping_nodes.items()):
                print(f"  {node}: clusters {sorted(clusters)}")
        
        for cluster_id in sorted(final_clusters):
            print(f"\nCluster {cluster_id}:")
            document.add_heading(f"\nCluster {cluster_id}:", level=6)
            for activity in sorted(final_clusters[cluster_id]):
                print(f" - {activity}")
                document.add_paragraph(f" - {activity}")
        
        return final_clusters, document
        
    except FileNotFoundError:
        print(f"Error: OSLOM executable '{oslom_executable}' not found.")
        print("Please install OSLOM and ensure it's in your PATH, or provide the full path to the executable.")
        print("OSLOM can be downloaded from: http://www.oslom.org/")
        raise
    except Exception as e:
        print(f"Error during OSLOM clustering: {e}")
        raise
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


from collections import defaultdict, Counter

def build_cooccurrence_graph(actions, k=3):
    adj = defaultdict(Counter)
    n = len(actions)
    for i in range(n):
        u = actions[i]
        for j in range(i + 1, min(i + k + 1, n)):
            v = actions[j]
            if u != v:
                adj[u][v] += 1
                adj[v][u] += 1

    adj_list = defaultdict(list)
    for u, neigh in adj.items():
        for v, w in neigh.items():
            adj_list[u].append((v, w))

    return adj_list, list(adj.keys())


def discover_cooccurrence(log_df, k=3):

    global_adj = defaultdict(Counter)

    for case_id, trace in log_df.groupby("case:concept:name"):

        actions = (
            trace.sort_values("time:timestamp")["concept:name"]
            .tolist()
        )

        adj_list, nodes = build_cooccurrence_graph(actions, k=k)

        # Merge into global adjacency
        for u, neighs in adj_list.items():
            for v, w in neighs:
                global_adj[u][v] += w

    return global_adj


def cooccurrence_to_weighted_edges(global_adj):

    edge_dict = {}

    for u, neighs in global_adj.items():
        for v, w in neighs.items():
            edge_dict[(u, v)] = w

    return edge_dict


def cooccurrence_to_digraph(global_adj):
    G = nx.DiGraph()

    for u, neighs in global_adj.items():
        for v, w in neighs.items():
            G.add_edge(u, v, weight=w)

    return G

