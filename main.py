import networkx as nx
import functions

file_path = "email-Eu-core.csv"  # Διαδρομή προς το αρχείο
edges = []

# Ανάγνωση και επεξεργασία ακμών
with open(file_path, 'r') as file:
    for line in file:
        node1, node2 = map(int, line.split())  # Χώρισμα με βάση το διάστημα
        edges.append((node1, node2))

# Δημιουργία  μεγάλου γραφουγράφου
G = nx.Graph()
G.add_edges_from(edges)

#Δημιουργία μικρών γράφων και Δοκιμή Αλγορίθμου
G1=functions.generate_custom_graph(5, 5, 3)
G2=functions.generate_custom_graph(5,20,20)

optimized_clusters_G1=functions.optimize_clusters(G1,120)
optimized_clusters_G2=functions.optimize_clusters(G2,2000)

#Οπτικοποίηση των γράφων
functions.visualization_graph(optimized_clusters_G1,G1)
functions.visualization_graph(optimized_clusters_G2,G2)

"""Χρήση μεγιστοποίησης Qd στον μεγάλο γράφο και αποθήκευση των clusters 
σε αρχείο csv για οπτικοποίηση κοινότητας μέσω Gephi"""

optimized_clusters_G_a_65000=functions.optimize_clusters(G,65000)

#optimized_clusters_G_a_60000=functions.optimize_clusters(G,60000)

functions.save_clusters_to_csv(optimized_clusters_G_a_65000,"clusters_a_65000.csv")

#functions.save_clusters_to_csv(optimized_clusters_G_a_60000,"clusters_a_60000.csv")