import sys
import configparser
import pandas as pd
import numpy as np
from sklearn.metrics import (max_error, mean_absolute_error,mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from PIL import Image
from IPython.display import Image
import sys
import matplotlib
matplotlib.use('Agg') 

from sklearn.linear_model import LinearRegression

MMS_COLUMNS = ['chembl-id', 'pot.(log,Ki)', 'pot.(nMol,Ki)', 'aromatic_smiles', 'non_stereo_aromatic_smieles',
               'all-chembl-ids', 'no.-meas.', 'pref_name', 'accession', 'natoms',
               'core', 'sub', 'sub_carbon_replacement', 'arorings', 'a_acc',
               'a_don', 'a_heavy', 'logP(o/w)', 'RBC', 'rings',
               'TPSA', 'vdw_vol', 'Weight']
MMS_COLRENAME = {"arorings": "arings", "a_acc": "acc", "a_don": "don", "logP(o/w)": "logp", "RBC": "rbc",
                 "TPSA": "tpsa", "Weight": "mw", "pot.(log,Ki)":"pot"}
MMS_FEATLIST = {'10': ["arings", "acc", "don", "a_heavy", "logp", "rbc", "rings", "tpsa", "vdw_vol", "mw"],
                '7' : ["arings", "acc", "don", "logp", "rbc", "tpsa", "mw"],
                '4' : ["logp", "rbc", "tpsa", "mw"],}
MMS_PROPERTY = "pot"

config = configparser.ConfigParser()
config.read(sys.argv[1])
print(config.sections())

file       = config["MLR"]["INPUT_FILE"]
result_dir = config["MLR"]["RESULT_DIR"]
njobs      = int(config["MLR"]["NJOBS"]) if "NJOBS" in config["MLR"].keys() else 1
nfeat      = config["MLR"]["NFEAT"] if "NFEAT" in config["MLR"].keys() else "7"
mseparate  = config["MLR"]["MSEPARATE"] if "MSEPARATE" in config["MLR"].keys() else "SIMPLE"
rtrain     = float(config["MLR"]["RTRAIN"]) if "RTRAIN" in config["MLR"].keys() else 0.8
d_rstate   = int(config["MLR"]["D_RSTATE"]) if "D_RSTATE" in config["MLR"].keys() else 0

mms_featlist = MMS_FEATLIST[nfeat]                
    

print("FILE:", file)
print("OUTDIR:", result_dir)
print("NJOBS:", njobs)
print("NFEAT:", nfeat)
print("MSEPARETE:", mseparate)
print("RTRAIN:", rtrain)
print("D_RSTATE:", d_rstate)
print("mms_featlist:", mms_featlist)

    
df = pd.read_table(file, index_col=0)
df = df.rename(columns=MMS_COLRENAME)
print(df.columns)
print(file, df["core"].iloc[0])
ndata = len(df.index)
ntrain = int(rtrain*ndata)
print(f"ndata: {ndata}, ntrain: {ntrain}")

X = df.loc[:, mms_featlist]
y = df.loc[:, MMS_PROPERTY]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ntrain, random_state=d_rstate)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
ydomain = y.min(), y.max()

print("output_dir", result_dir)

est = LinearRegression(n_jobs=njobs)

# traininig
est.fit(X_train, y_train)
y_train_pred = est.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

# save the training results
res_val = (rmse_train, r2_train, est)
print("TRAIN RESULTS: RMSE, R2")
print((res_val[0],res_val[1]))
best_model = res_val[2]
#best_model.save_all()

y_test_pred = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

print("TEST RESULTS (TRAIN BEST): RMSE, R2")
print((rmse_test, r2_test))
