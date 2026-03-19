import pandas as pd
import numpy as np
try:
    import igraph as ig
except ImportError:
    ig = None

def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences (lists of strings).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = matrix[x-1, y-1]
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return matrix[size_x-1, size_y-1]

def get_context(df, idx, window_size):
    """
    Extracts the context window around a given index.
    Returns a list of action names (prefix + suffix).
    """
    start_idx = max(0, idx - window_size)
    end_idx = min(len(df), idx + window_size + 1)
    
    # Get the sequence of actions in the window
    context = df.iloc[start_idx:end_idx]['concept:name'].tolist()
    return context

def split_common_actions(log_df, common_actions, window_size, distance_threshold=0, method='distance'):
    """
    Splits common actions based on their context.
    
    Parameters:
    - method: 'distance' (greedy) or 'clustering' (graph-based community detection)
    """
    df_out = log_df.copy()
    
    # Ensure pandas index is range index for easier access by integer location
    df_out.reset_index(drop=True, inplace=True)

    for action in common_actions:
        print(f"Processing action: {action} with method: {method}")
        # Find all indices of the common action
        action_indices = df_out.index[df_out['concept:name'] == action].tolist()
        
        if not action_indices:
            continue
        
        # Extract all contexts first
        contexts = []
        for idx in action_indices:
            contexts.append(get_context(df_out, idx, window_size))
            
        labels_map = {} # Maps index in action_indices to new label suffix ID (0-based)
        
        if method == 'distance':
            groups = [] # List of dictionaries: {'representative_context': list, 'indices': list}
            for i, idx in enumerate(action_indices):
                current_context = contexts[i]
                best_group_idx = -1
                min_dist = float('inf')
                
                # Find the best matching group (minimum distance)
                for group_idx, group in enumerate(groups):
                    dist = levenshtein_distance(current_context, group['representative_context'])
                    # print(f"  Distance between {current_context} and {group['representative_context']}: {dist}")
                    if dist <= distance_threshold and dist < min_dist:
                        min_dist = dist
                        best_group_idx = group_idx
                
                # Assign to best group if found
                if best_group_idx != -1:
                    labels_map[i] = best_group_idx
                else:
                    # Create new group if no match found
                    groups.append({
                        'representative_context': current_context,
                    })
                    labels_map[i] = len(groups) - 1
                    
        elif method == 'clustering':
            if ig is None:
                raise ImportError("igraph is required for 'clustering' method. Please install python-igraph.")
                
            # Optimization: Compute distance on unique contexts only
            # Convert list of strings to tuple for hashing
            context_tuples = [tuple(c) for c in contexts]
            unique_contexts = list(set(context_tuples))
            unique_map = {u: i for i, u in enumerate(unique_contexts)} # Map tuple -> unique_id
            
            n_unique = len(unique_contexts)
            print(f"  Found {len(action_indices)} instances, {n_unique} unique contexts.")
            
            # Build similarity graph on unique contexts
            edges = []
            weights = []
            
            for i in range(n_unique):
                for j in range(i + 1, n_unique):
                    # Convert tuples back to lists for distance function
                    dist = levenshtein_distance(list(unique_contexts[i]), list(unique_contexts[j]))
                    
                    # Connection criteria:
                    # If dist <= threshold, we add an edge.
                    # Weight can be inverse distance.
                    if dist <= distance_threshold:
                        edges.append((i, j))
                        # Avoid division by zero if dist is 0
                        weights.append(1.0 / (1.0 + dist)) 
            
            g = ig.Graph(n=n_unique, edges=edges)
            g.es['weight'] = weights
            
            # Community detection (Louvain)
            # If graph is disconnected, this works on components
            if len(edges) > 0 or n_unique > 0:
                # Use community_multilevel (Louvain) which is fast and good
                try:
                    communities = g.community_multilevel(weights='weight')
                    membership = communities.membership
                except Exception as e:
                    print(f"  Clustering failed: {e}. Fallback to components.")
                    # Fallback to connected components if something goes wrong
                    components = g.components()
                    membership = components.membership
            else:
                 # Should not happen if n_unique > 0
                membership = []

            # Map back to original instances
            if membership:
                for i, idx in enumerate(action_indices):
                    u_idx = unique_map[context_tuples[i]]
                    labels_map[i] = membership[u_idx]

        # Apply new labels
        if not labels_map:
            continue
            
        # Normalize group IDs to 1..N
        unique_group_ids = sorted(list(set(labels_map.values())))
        group_id_map = {old: new+1 for new, old in enumerate(unique_group_ids)}
        
        print(f"  Generated {len(unique_group_ids)} variant labels.")
        
        for i, idx in enumerate(action_indices):
            group_id = labels_map[i]
            new_label_id = group_id_map[group_id]
            new_label = f"{action}_{new_label_id}"
            df_out.at[idx, 'concept:name'] = new_label
            
    return df_out


if __name__ == "__main__":
    # Configuration
    log_path = r"Transformed_Logs_and_Results\Our\Interleaved\Transformed_NonActivity_Normalized_Log_With_Noise_0\unsegment_log1.csv"
    
    COMMON_ACTIONS = ['insertValue_type_Web', 'insertValue_age_Web', 'insertValue_name_Web', 'insertValue_surname_Web', 'clickButton_OK_Web', 'clickButton_confirm_Web'] 
    WINDOW_SIZE = 3
    DISTANCE_THRESHOLD = 3
    METHOD = 'distance' # 'distance' or 'clustering'
    
    try:
        print(f"Reading log from {log_path}...")
        df = pd.read_csv(log_path)
        
        if 'concept:name' not in df.columns:
            print("Error: 'concept:name' column not found in log.")
        else:
            print(f"Log loaded. rows={len(df)}")
            
            refined_df = split_common_actions(df, COMMON_ACTIONS, WINDOW_SIZE, DISTANCE_THRESHOLD, method=METHOD)
            
            # Save the result
            output_path = log_path.replace(".csv", f"_split_{METHOD}.csv")
            refined_df.to_csv(output_path, index=False)
            print(f"Refined log saved to {output_path}")
            
    except FileNotFoundError:
        print(f"File not found: {log_path}")
    except ImportError as e:
        print(f"Import Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
