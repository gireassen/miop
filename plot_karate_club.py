import itertools
import networkx as nx

def generate_itemsets(data, min_support):
    # Count item occurrences
    item_counts = {}
    for transaction in data:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    
    # Filter items based on minimum support
    frequent_items = {item for item, count in item_counts.items() if count >= min_support}
    
    return frequent_items

def generate_association_rules(data, min_support, min_confidence):
    frequent_items = generate_itemsets(data, min_support)
    
    # Generate 1-item frequent itemsets
    itemsets = [frozenset([item]) for item in frequent_items]
    
    # Generate k-item frequent itemsets
    k = 2
    while True:
        candidate_itemsets = set()
        for itemset in itemsets:
            for item in frequent_items:
                if item not in itemset:
                    candidate = itemset | frozenset([item])
                    candidate_itemsets.add(candidate)
        
        # Count candidate itemset occurrences
        itemset_counts = {}
        for transaction in data:
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction):
                    if candidate in itemset_counts:
                        itemset_counts[candidate] += 1
                    else:
                        itemset_counts[candidate] = 1
        
        # Filter itemsets based on minimum support
        frequent_itemsets = {itemset for itemset, count in itemset_counts.items() if count >= min_support}
        
        if not frequent_itemsets:
            break
        
        # Generate association rules
        for itemset in frequent_itemsets:
            for item in itemset:
                antecedent = frozenset([item])
                consequent = itemset - antecedent
                confidence = itemset_counts[itemset] / [antecedent]
                if confidence >= min_confidence:
                    print(f"{antecedent} => {consequent}, Confidence: {confidence}")
        
        itemsets = frequent_itemsets
        k += 1

# Load data
G = nx.karate_club_graph()
data = nx.to_numpy_array(G)

# Set minimum support and minimum confidence
min_support = 2
min_confidence = 0.5

# Generate association rules
generate_association_rules(data, min_support, min_confidence)