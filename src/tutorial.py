"""
A set of functions supporting the different parts of our causal learning tutorial
"""

import numpy as np
import networkx as nx 
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def logit(p):
    return np.log(p) - np.log(1 - p)

def generate_data(n, seed_val):
    np.random.seed(seed_val)
    # randomly initialise comorbidities 
    w1 = np.random.binomial(1, 0.5, n)        
    w2 = np.random.binomial(1, 0.65, n)    
    w3 = np.round(np.random.uniform(0, 4, n), 3)
    w4 = np.round(np.random.uniform(0, 5, n), 3)
    
    # probability of receiving treatment model
    p_A = sigmoid(-5 + 0.05*w2 + 0.25*w3 + 0.6*w4 +0.4*w2*w4)
    # binary outcome based on this calculated probability
    A = np.random.binomial(1, p_A, n)
    
    # probability of outcome models
    p_y1 = sigmoid(-1 + 1 -0.1*w1 + 0.35*w2 + 0.25*w3 + 0.2*w4 + 0.15*w2*w4)
    p_y0 = sigmoid(-1 + 0 -0.1*w1 + 0.35*w2 + 0.25*w3 + 0.2*w4 + 0.15*w2*w4)
    Y1 = np.random.binomial(1, p_y1, n) # what would the outcome be had they been treated?
    Y0 = np.random.binomial(1, p_y0, n) # what would the outcome be had they not been treated?
    
    # actual observed outcome (taking into account treatment variable )
    Y = Y1*A + Y0*(1-A)
    
    cols = ['w1', 'w2', 'w3', 'w4', 'A','Y', 'Y1', 'Y0']
    df = pd.DataFrame([w1, w2, w3, w4, A, Y, Y1, Y0]).T
    df.columns = cols
    return df

def generate_data_nonlinearity(n, seed_val):
    np.random.seed(seed_val)
    # randomly initialise comorbidities 
    w1 = np.random.binomial(1, 0.5, n)        
    w2 = np.random.binomial(1, 0.65, n)    
    w3 = np.round(np.random.uniform(0, 4, n), 3)
    w4 = np.round(np.random.uniform(0, 5, n), 3)
    
    # probability of receiving treatment model
    p_A = sigmoid(-5 + 0.05*w2 + 0.25*w3 + 0.6*w4 +0.4*w2*w4 + 0.5*w1*w2*w3*w4 + 0.3*w4**2)
    # binary outcome based on this calculated probability
    A = np.random.binomial(1, p_A, n)
    
    # probability of outcome models
    p_y1 = sigmoid(-1 + 1 -0.1*w1 + 0.35*w2 + 0.25*w3 + 0.2*w4 + 0.15*w2*w4 + 0.2*w1*w3)
    p_y0 = sigmoid(-1 + 0 -0.1*w1 + 0.35*w2 + 0.25*w3 + 0.2*w4 + 0.15*w2*w4 + 0.2*w1*w3)
    Y1 = np.random.binomial(1, p_y1, n) # what would the outcome be had they been treated?
    Y0 = np.random.binomial(1, p_y0, n) # what would the outcome be had they not been treated?
    
    # actual observed outcome (taking into account treatment variable )
    Y = Y1*A + Y0*(1-A)
    
    cols = ['w1', 'w2', 'w3', 'w4', 'A','Y', 'Y1', 'Y0']
    df = pd.DataFrame([w1, w2, w3, w4, A, Y, Y1, Y0]).T
    df.columns = cols
    return df

def produce_dag():
    G = nx.DiGraph()
    # Add edges
    edges = [\
            ('W2', 'W3'), ('W2', 'W4'), ('W2', 'Y'), \
            ('W3', 'A'), ('W3', 'Y'), \
            ('W1', 'W3'), ('W1', 'A'), ('W1', 'Y'), ('W1', 'W4'), \
            ('W4', 'A'), ('W4', 'Y'), \
            ('A', 'Y')]

    G.add_edges_from(edges)

    # Assign layers for multipartite layout (optional, improves clarity)
    for node in G.nodes:
        if ((node == 'W1') or (node == 'W2')):
            G.nodes[node]['layer'] = 0
        elif ((node == 'W3') or (node == 'W4')):
            G.nodes[node]['layer'] = 1
        elif node == 'A':
            G.nodes[node]['layer'] = 2
        elif node == 'Y':
            G.nodes[node]['layer'] = 3

    # Use multipartite layout
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig = nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=15, font_color="black", font_weight="bold")
    return fig