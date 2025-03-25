import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

# Συνάρτηση Υπολογισμού Μήτρας Διαδρομών
def compute_paths_matrix(B, Dv, kmax):
    size = len(B)
    Pv = np.zeros((size, size))
    
    # Υπολογισμός της πρώτης δύναμης της μήτρας
    power_matrix = np.copy(B)
    
    # Υπολογίζουμε μόνο τα στοιχεία του ανώτερου τριγωνικού πίνακα και τη διαγώνιο
    for k in range(1, kmax + 1):
        for i in range(size):
            for j in range(i, size):  # Ανώτερος τριγωνικός πίνακας (συμπεριλαμβανομένης της διαγωνίου)
                if Dv[i].get(j, float('inf')) == k:
                    Pv[i, j] += power_matrix[i, j]
                    if i != j:  # Ενημέρωση και για το συμμετρικό στοιχείο
                        Pv[j, i] = Pv[i, j]

        # Ενημέρωση της power_matrix για την επόμενη δύναμη
        if k < kmax:
            power_matrix = np.dot(power_matrix, B)

    return Pv


# Συνάρτηση Υπολογισμού Γενικευμένου Βαθμού
def generalized_degree(Pv, Dv, kmax):
    size = len(Dv)
    d = np.zeros((kmax, size))
    for k in range(kmax):
        for node in Dv.keys():
            neighbors_at_k = [neighbor for neighbor, dist in Dv[node].items() if dist == k + 1]
            d[k][node] = sum(Pv[node, neighbor] for neighbor in neighbors_at_k)
    return d


# Συνάρτηση Υπολογισμού Αναμενόμενων Αποστάσεων
def compute_expected_distance(d, m, kmax):
    size = d.shape[1]
    Dv_exp = np.zeros((size, size))
    for k in range(kmax):
        if m[k] == 0:
            continue
        Pr_matrix = np.outer(d[k], d[k]) / (4 * m[k] ** 2)
        Dv_exp += (k + 1) * Pr_matrix
    return Dv_exp

# Συνάρτηση Υπολογισμού \( Q_d \) για τα clusters
def compute_qd(G,a,clusters):

    # Δημιουργία αντιστοίχισης κόμβων σε δείκτες
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    # Υπολογισμός αποστάσεων μεταξύ κόμβων (Dv)
    Dv = {
        node_to_index[node]: {node_to_index[neighbor]: dist for neighbor, dist in dict(neighbors).items()}
        for node, neighbors in nx.shortest_path_length(G)
    }

    # Υπολογισμός μήτρας γειτνίασης
    B = nx.adjacency_matrix(G).todense()
    
    # Υπολογισμός μέγιστης απόστασης (kmax)
    kmax = max(max(lengths.values()) for lengths in Dv.values())
    
    # Υπολογισμός μήτρας διαδρομών
    Pv = compute_paths_matrix(B, Dv, kmax)
    
    # Υπολογισμός γενικευμένου βαθμού
    d = generalized_degree(Pv, Dv, kmax)
    
    # Υπολογισμός αναμενόμενων αποστάσεων
    m = np.sum(d, axis=1) / 2
    Dv_exp = compute_expected_distance(d, m, kmax)


    # Αρχικοί υπολογισμοί αποστάσεων
    expected_distances = [
        np.sum(Dv_exp[np.ix_([node_to_index[i] for i in cluster], [node_to_index[j] for j in cluster])]) / 2
        for cluster in clusters
    ]
    actual_distances = [
        sum(Dv[node_to_index[i]].get(node_to_index[j], 0) for i in cluster for j in cluster) / 2
        for cluster in clusters
    ]
    b=1
    # Αρχικό Qd
    qd = sum(a * exp - b * act for exp, act in zip(expected_distances, actual_distances))
    return qd


def optimize_clusters(G, a):
    # Δημιουργία αντιστοίχισης κόμβων σε δείκτες
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    # Υπολογισμός αποστάσεων μεταξύ κόμβων (Dv)
    Dv = {
        node_to_index[node]: {node_to_index[neighbor]: dist for neighbor, dist in dict(neighbors).items()}
        for node, neighbors in nx.shortest_path_length(G)
    }

    # Υπολογισμός μήτρας γειτνίασης
    B = nx.adjacency_matrix(G).todense()
    
    # Υπολογισμός μέγιστης απόστασης (kmax)
    kmax = max(max(lengths.values()) for lengths in Dv.values())
    
    # Υπολογισμός μήτρας διαδρομών
    Pv = compute_paths_matrix(B, Dv, kmax)
    
    # Υπολογισμός γενικευμένου βαθμού
    d = generalized_degree(Pv, Dv, kmax)
    
    # Υπολογισμός αναμενόμενων αποστάσεων
    m = np.sum(d, axis=1) / 2
    Dv_exp = compute_expected_distance(d, m, kmax)

    # Κάθε κόμβος ξεκινά σε ξεχωριστό cluster
    clusters = [[node] for node in G.nodes()]

    # Αρχικοί υπολογισμοί αποστάσεων
    expected_distances = [
        np.sum(Dv_exp[np.ix_([node_to_index[i] for i in cluster], [node_to_index[j] for j in cluster])]) / 2
        for cluster in clusters
    ]
    actual_distances = [
        sum(Dv[node_to_index[i]].get(node_to_index[j], 0) for i in cluster for j in cluster) / 2
        for cluster in clusters
    ]

    b=1
    # Αρχικό Qd
    qd = sum(a * exp - b * act for exp, act in zip(expected_distances, actual_distances))

    # Βελτιστοποίηση
    while True:
        improvement_found = False

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Αν δεν υπάρχει σύνδεση μεταξύ των clusters, παράβλεψέ τα
                if not any(G.has_edge(node1, node2) for node1 in clusters[i] for node2 in clusters[j]):
                    continue

                # Νέο cluster
                new_cluster = clusters[i] + clusters[j]

                # Υπολογισμός αποστάσεων για το νέο cluster
                new_exp_dist = np.sum(Dv_exp[np.ix_(
                    [node_to_index[k] for k in new_cluster],
                    [node_to_index[l] for l in new_cluster]
                )]) / 2

                new_act_dist = sum(
                    Dv[node_to_index[k]].get(node_to_index[l], 0)
                    for k in new_cluster for l in new_cluster
                ) / 2

                # Υπολογισμός νέου Qd
                new_qd = qd
                new_qd -= a * expected_distances[i] - b * actual_distances[i]
                new_qd -= a * expected_distances[j] - b * actual_distances[j]
                new_qd += a * new_exp_dist - b * new_act_dist

                # Αν υπάρχει βελτίωση, κάνε τη συγχώνευση
                if new_qd > qd:
                    clusters[i] = new_cluster
                    clusters.pop(j)
                    expected_distances[i] = new_exp_dist
                    actual_distances[i] = new_act_dist
                    expected_distances.pop(j)
                    actual_distances.pop(j)
                    qd = new_qd
                    improvement_found = True
                    break

            if improvement_found:
                break

        # Τερματισμός αν δεν βρέθηκε βελτίωση
        if not improvement_found:
            break

    return clusters, qd


def generate_custom_graph(x, z, y):
    '''
    Δημιουργεί έναν γράφο με x κοινότητες.
    - Κάθε κοινότητα έχει z κόμβους και πλήρη σύνδεση (πλήρες υπογράφημα).
    - Οι κοινότητες συνδέονται μεταξύ τους με y ακμές.
    
    - x: αριθμός κοινών περιοχών (clusters)
    - z: αριθμός κόμβων σε κάθε κοινότητα
    - y: αριθμός ακμών μεταξύ των κοινοτήτων
    '''
    G = nx.Graph()

    # Δημιουργία των x κοινών περιοχών
    communities = []
    for i in range(x):
        # Δημιουργία κόμβων για την κοινότητα
        cluster_nodes = np.arange(i * z, (i + 1) * z)
        G.add_nodes_from(cluster_nodes)
        communities.append(cluster_nodes)
        
        # Δημιουργία πλήρους υπογράμματος (σύνδεση όλων των κόμβων μέσα στην κοινότητα)
        for i in range(z):
            for j in range(i + 1, z):
                G.add_edge(cluster_nodes[i], cluster_nodes[j])

    # Σύνδεση των κοινοτήτων σε κύκλο με y ακμές
    for i in range(x):
        current_cluster = communities[i]
        next_cluster = communities[(i + 1) % x]  # Κυκλική σύνδεση
        for j in range(y):
            G.add_edge(current_cluster[j % z], next_cluster[j % z])
    
    return G

# Δημιουργία τυχαίων χρωμάτων
def generate_random_colors(num_colors):
    return ["#" + ''.join(random.choices("0123456789ABCDEF", k=6)) for _ in range(num_colors)]

def visualization_graph(clusters,G):
    
    if isinstance(clusters, tuple):
        clusters = clusters[0] 
    # Μετατροπή των clusters σε hashable tuples
    clusters_as_tuples = [tuple(cluster) for cluster in clusters]

    # Δημιουργία λεξικού χρωμάτων
    colors = generate_random_colors(len(clusters_as_tuples))
    cluster_colors = {cluster: color for cluster, color in zip(clusters_as_tuples, colors)}

    # Χρωματισμός κόμβων με βάση το cluster
    node_to_cluster = {node: cluster for cluster in clusters_as_tuples for node in cluster}
    node_colors = [cluster_colors[node_to_cluster[node]] for node in G.nodes]

    # Σχεδίαση γράφου
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Καθορισμός διάταξης
    nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=800,
    edge_color="gray",
    font_weight="bold"
    )
    plt.title("Optimized Clusters Visualization")
    plt.show()

def save_clusters_to_csv(clusters, output_file):
    if isinstance(clusters, tuple):
        clusters = clusters[0] 
    # Δημιουργία αντιστοίχισης κόμβων σε clusters
    node_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_id

    # Γράψιμο δεδομένων σε CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Πρώτη γραμμή: Ονόματα στηλών
        writer.writerow(["id", "clustering"])
        # Εγγραφές για κάθε κόμβο
        for node, cluster_id in sorted(node_to_cluster.items()):
            writer.writerow([node, cluster_id])