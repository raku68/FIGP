import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../external/python_library"))
#print(sys.path)
import configparser
import pandas as pd
import numpy as np
from sklearn.metrics import (max_error, mean_absolute_error,mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from PIL import Image
from IPython.display import Image
import matplotlib
matplotlib.use('Agg') 

from svr.svr_wrapper import cross_validation_nu_svr_custom_kernel

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

file       = config["NUSVR"]["INPUT_FILE"]
result_dir = config["NUSVR"]["RESULT_DIR"]
nfeat      = config["NUSVR"]["NFEAT"] if "NFEAT" in config["NUSVR"].keys() else "7"
mseparate  = config["NUSVR"]["MSEPARATE"] if "MSEPARATE" in config["NUSVR"].keys() else "SIMPLE"
rtrain     = float(config["NUSVR"]["RTRAIN"]) if "RTRAIN" in config["NUSVR"].keys() else 0.8
d_rstate   = int(config["NUSVR"]["D_RSTATE"]) if "D_RSTATE" in config["NUSVR"].keys() else 0

mms_featlist = MMS_FEATLIST[nfeat]                
    

print("FILE:", file)
print("OUTDIR:", result_dir)
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

est = cross_validation_nu_svr_custom_kernel(X_train.values, y_train.values, paramset=None, nf=5,
                                            xts=X_test.values, yts=y_test.values,
                                            varbose=True, my_kernel='tanimoto', is_scaling=False)
# scoring 'score()' function which is default r2_score()
"""
self.gmodel = grid_opt_model
self.ytrp = YtrPred
self.ytsp = YtsPred
"""

# traininig
#est.fit(X_train, y_train)
#y_train_pred = est.predict(X_train)
y_train_pred = est.ytrp
rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))
r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

# save the training results
res_val = (rmse_train, r2_train, est)
print("TRAIN RESULTS: RMSE, R2")
print((res_val[0],res_val[1]))
best_model = res_val[2].gmodel
print(f"Best Parameters: {best_model.best_params_}")
#best_model.save_all()

#y_test_pred = best_model.predict(X_test)
y_test_pred = est.ytsp
rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

print("TEST RESULTS (TRAIN BEST): RMSE, R2")
print((rmse_test, r2_test))
