import os, sys
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

class Database():
    def __init__(self, n_dim=4096, distance="euclidean"):
        self.f = n_dim
        self.distance = distance
        self.db = None
        self.df = None

    def load_db(self, db_path="data/Db_feature.ann", df_path="data/Df_Search.csv"):
        self.db = AnnoyIndex(self.f, self.distance)
        self.db.load(db_path)
        self.df = pd.read_csv(df_path)

    def create_db(self, featureDB, model, use_svm=False, base_path=""):
        self.db = AnnoyIndex(self.f, self.distance)
        featureDB['isExtracted'] = 1
        for i in range(len(featureDB)):
          sys.stdout.write('\r'+f'idx={i+1}')
          tmp_img = featureDB.iloc[i]["name"]
          # X: [(probs, feats)]
          X = model.extract_feature(tmp_img, base_path=base_path, use_svm=use_svm, verbose=False)
          if len(X) < 1:
            featureDB.loc[i,['isExtracted']] = 0
            continue
          tmp_results = X[np.argmax([X[i][0] for i in range(len(X))])] # Get feature with the highest prob
          feature = tmp_results[1]
          feature = feature.flatten()
          self.db.add_item(i, feature)

        self.db.build(1000)
        self.df=featureDB

    def save_db(self, save_folder="data"):
        self.db.save(os.path.join(save_folder,"Db_feature.ann"))
        self.df.to_csv(os.path.join(save_folder,"Df_Search.csv"),index=False)
        print(f"Saving database to {save_folder}")

    def set_distance_function(self, distance_function):
        self.distance_function = distance_function
