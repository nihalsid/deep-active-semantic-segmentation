import os
import sys
import json

cluster_dict = {}

for cluster in os.listdir(sys.argv[1]):
    cluster_id = cluster
    cluster_dict[cluster_id] = []
    for f in os.listdir(os.path.join(sys.argv[1], cluster)):
        cluster_dict[cluster_id].append('/leftImg8bit/train/' + f.split('_')[0] + '/' + f)

with open('clusters_0.txt', 'w') as fptr:
    fptr.write(json.dumps(cluster_dict))
