import sys
import time
import configparser
import pandas as pd
import numpy as np
from sklearn.metrics import (max_error, mean_absolute_error,mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from PIL import Image
from IPython.display import Image
import sys

"""for Symbilic_Reg_ST"""
import operator
import random
import copy
#from deap import algorithms, base, creator, gp, tools
#from figp import Symbolic_Reg
from figp.deap_based_FGP_NLS2 import Symbolic_Reg
#from figp.deap_based_FGP_algorithms import FGP_NLS_algorithm
#from figp.deap_based_func_node import Node_space, NumpyBasedFunction
# add, cube, div, exp, ln,
#                                            mul, sqrt, square, sub, protected_division,
#                                        protected_sqrt, protected_ln)

# def gnoise(x):
#     _std = np.std(x, axis=0)
#     _ret = pd.DataFrame(index=x.index, columns=x.columns, dtype="float64")

#     for row in x.index:
#         _gnoise = np.random.normal(loc=0, scale=1, size=_ret.shape[1])
#         _ret.loc[row, :] = pd.Series(_std*_gnoise, index=_ret.columns).astype("float64")
    
#     # print(_ret)
#     return _ret


# class Symbolic_Reg_ST(Symbolic_Reg):
#     def __init__(self, 
#                  population_size = 1000, 
#                  generations     = 200, 
#                  tournament_size = 5,
#                  num_elite_select = 1,
#                  max_depth       = 4,
#                  function_set    = ['add', 'sub', 'mul', 'div', 'ln', 'exp', 'sqrt', 'square', 'cube'], 
#                  metric          = 'mae',
#                  p_crossover     = 0.7,
#                  p_mutation      = 0.2,
#                  random_state    = 0,
#                  const_range     = (-1, 1),
#                  init_max_trial  = 50000,
#                  init_unique     = True,
#                  var_max_trial   = 20,
#                  function_filter = True, 
#                  variable_filter = True, 
#                  xydomain_filter = True,
#                  constonly_filter= True,
#                  x_domain        = None,
#                  y_domain        = None,
#                  domain_equal    = (True, True),
#                  results_dir     = './deap_based_FGP_results',
#                  stabilize       = 1,
#                  s_gnoise        = False,
#                  s_lmd1          = 1.0,
#                  s_lmd2          = 0.5,
#                  s_clmd2         = 0.1
#                  ):
#         """[summary]

#         Args:
#             population_size (int, optional): [description]. Defaults to 1000.
#             generations (int, optional): [description]. Defaults to 200.
#             tournament_size (int, optional): [description]. Defaults to 5.
#             num_elite_select (int, optional): [description]. Defaults to 1.
#             max_depth (int, optional): [description]. Defaults to 4.
#             function_set (list, optional): [description]. Defaults to ['add', 'sub', 'mul', 'div', 'ln', 'log', 'sqrt'].
#             metric (str, optional): [description]. Defaults to 'mae'.
#             p_crossover (float, optional): [description]. Defaults to 0.7.
#             p_mutation (float, optional): [description]. Defaults to 0.2.
#             random_state (int, optional): [description]. Defaults to 0.
#             const_range (tuple, optional): [description]. Defaults to (0, 1).
#             x_domain ([type], optional): [description]. Defaults to None.
#             y_domain ([type], optional): [description]. Defaults to None.
#             results_dir (str, optional): [description]. Defaults to './results'.
#         """
        
#         super().__init__(
#             population_size,
#             generations, 
#             tournament_size,
#             num_elite_select,
#             max_depth,
#             function_set,
#             metric,
#             p_crossover,
#             p_mutation,
#             random_state,
#             const_range,
#             init_max_trial,
#             init_unique,
#             var_max_trial,
#             function_filter,
#             variable_filter,
#             xydomain_filter,
#             constonly_filter,
#             x_domain,
#             y_domain,
#             domain_equal,
#             results_dir)

#         self.stabilize = stabilize
#         self.s_gnoise = s_gnoise
#         self.s_lmd1 = s_lmd1
#         self.s_lmd2 = s_lmd2
#         self.s_clmd2 = s_clmd2

#     def fit(self, x, y):
#         _start_gen = time.time()
#         self.fit_x_ = x
#         self.fit_y_ = y
        
#         self._surveyed_individuals_ = surveyed_individuals(self.fit_x_)
        
#         self.pset = gp.PrimitiveSet("MAIN", x.shape[1])
        
#         for i, x_name in enumerate(x.columns):
#             p = {'ARG{}'.format(i):f'{x_name}'}
#             self.pset.renameArguments(**p)

#         if 'add' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.add, 2)
#         if 'sub' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.sub, 2)
#         if 'mul' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.mul, 2)
#         if 'div' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.div, 2)
#         if 'ln'  in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.ln, 1)
#         if 'sqrt' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.sqrt, 1)
#         if 'square' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.square, 1)
#         if 'cube' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.cube, 1)
#         if 'exp' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.exp, 1)
#         if 'protected_division' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_division, 2)
#         if 'protected_sqrt' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_sqrt, 1)
#         if 'protected_ln' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_ln, 1)

    
#         # add initial constant to be optimized
#         n_c_node = 1
#         add_n_c_node = 0
#         _count = 0
#         while add_n_c_node < n_c_node:
#             try:
#                 self.pset.addEphemeralConstant(f'c_node_{_count}', lambda: random.uniform(self.const_range[0],self.const_range[1]))
#                 add_n_c_node += 1
#                 _count += 1
#             except:
#                 _count += 1
#                 pass

#         creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#         creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#         def _progress_bar():
#             if int(self.population_size*0.01)*self._n_time == self._n_ind_gen_successes:
#                 if self._n_time%10 == 0:
#                     print(f'{int(self._n_time)}%', end='')
#                 else:
#                     print('|', end='')
#                 self._n_time += 1

#         def filter_initIterate(container, generator, max_trial=self.init_max_trial, unique=self.init_unique, text_log=self.text_log_):
#             for i in range(max_trial):
#                 ind = creator.Individual(generator())
#                 score = self._evalSymbReg(ind, self.fit_x_, self.fit_y_)
#                 self._n_ind_generations += 1
                
#                 if score[0] != np.inf:
#                     if unique:
#                         if self._surveyed_individuals_.is_unobserved(creator.Individual(ind)):
#                             self._n_ind_gen_successes += 1
#                             _progress_bar()
#                             return container(ind)
#                     else:
#                         self._n_ind_gen_successes += 1
#                         _pgogress_bar()
#                         return container(ind)
                
#             raise NameError(f'The maximum number of trials has been reached. \nNumber already generated : {self._n_ind_gen_successes}\nNumber of challenges : {self._n_ind_generations}')

#         self.toolbox_ = base.Toolbox()
#         self.toolbox_.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2) # gp.genHalfAndHalf https://deap.readthedocs.io/en/master/api/tools.html#deap.gp.genHalfAndHalf
        
#         self.toolbox_.register("individual", filter_initIterate, creator.Individual, self.toolbox_.expr)
#         # if (self.function_filter | self.variable_filter | self.xydomain_filter):
#         #     self.toolbox_.register("individual", filter_initIterate, creator.Individual, self.toolbox_.expr)
#         # else:
#         #     # Normal generation without filter
#         #     self.toolbox_.register("individual", tools.initIterate, creator.Individual, self.toolbox_.expr)
            
#         self.toolbox_.register("population", tools.initRepeat, list, self.toolbox_.individual)
#         self.toolbox_.register("compile", gp.compile, pset=self.pset)
#         self.toolbox_.register("evaluate", self._evalSymbReg, x=x, y_true=y)
#         self.toolbox_.register("select", tools.selTournament, tournsize=self.tournament_size)

#         # gp.cxOnePoint : 1 point crossover
#         # https://deap.readthedocs.io/en/master/api/tools.html?highlight=bloat#deap.gp.cxOnePoint
#         self.toolbox_.register("mate", gp.cxOnePoint) 
#         self.toolbox_.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

#         self.toolbox_.register("expr_mut", gp.genFull, min_=0, max_=2)

#         # gp.mutUniform 
#         # https://deap.readthedocs.io/en/master/api/tools.html?highlight=bloat#deap.gp.mutUniform
#         self.toolbox_.register("mutate", gp.mutUniform, expr=self.toolbox_.expr_mut, pset=self.pset) 
#         self.toolbox_.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        
#         self._time0 = time.time()

#         print('Generation of initial generation')
#         pop = self.toolbox_.population(n=self.population_size)
#         print('\nGeneration complete')
        
#         self.text_log_.print([f' {self._n_ind_gen_successes} [ ind ] / {self._n_ind_generations} [ trials ] (time : {(time.time() - self._time0)/60:.3f} min)\n'])
        
#         self.hof = tools.HallOfFame(self.num_elite_select)
        
#         pop, log = FGP_NLS_algorithm(population       = pop,
#                             toolbox          = self.toolbox_, 
#                             cxpb             = self.p_crossover,
#                             mutpb            = self.p_mutation,
#                             ngen             = self.generations,
#                             halloffame       = self.hof,
#                             num_elite_select = self.num_elite_select,
#                             var_max_trial    = self.var_max_trial, 
#                             check_func       = self._surveyed_individuals_,
#                             text_log         = self.text_log_,
#                             save_dir         = self.results_dir,
#                             func_name        = self.function_set)
        
#         self.expr = tools.selBest(pop, 1)[0]
#         self.tree = gp.PrimitiveTree(self.expr)
#         self.nodes, self.edges, self.labels = gp.graph(self.expr)
#         self.log = log
        
#         self.text_log_.print([f'FGP-NLS All Execution Time : {time.time() - _start_gen:.3f} s', 
#                               f'FGP-NLS All Execution Time : {(time.time() - _start_gen)/60:.1f} min',
#                               f'FGP-NLS All Execution Time : {(time.time() - _start_gen)/60/60:.1f} h',
#                               f'Number of constant optimization warnings : {self._warnings}'
#                               ])
#         return self

#     # def predict(self, x):
#     #     y_pred = self._pred(x, self.expr)
#     #     return y_pred
    
#     # def _pred(self, x, expr):
#     #     func = self.toolbox_.compile(expr=expr)
#     #     x_data = (x['{}'.format(i)] for i in list(x.columns))
#     #     # y_pred = func(*x_data)
#     #     try:
#     #         y_pred = func(*x_data)
#     #     except:
#     #         self._warnings += 1
#     #         self.text_log_.print(['ERROR!! nan is included.', f'{str(expr)}', self.root])
#     #         self.text_log_.print(self.temporary)
#     #         self.text_log_.print(['ERROR!! _results'])
#     #         self.text_log_.print(self.temporary2)
            
#     #         y_pred = np.inf 
            
#     #     if np.isscalar(y_pred):      # Avoid scalar errors.
#     #         y_pred = pd.Series(np.full(x.shape[0], float(y_pred)))
#     #     elif len(y_pred.shape) == 0: # Avoid errors due to singleton arrays.
#     #         y_pred = y_pred.item()
#     #         y_pred = pd.Series(np.full(x.shape[0], float(y_pred)))
#     #     return y_pred
        
#     def _evalSymbReg(self, individual, x, y_true):
#         individual.state = ''
        
#         # >>>>> func of const opt
#         def _func(constants, x, y, individual, compiler, constant_nodes):
#             _idx = 0
#             for i in constant_nodes:
#                 if ~np.isnan(constants[_idx]):
#                     cnode = copy.deepcopy(individual[i])
#                     cnode.value = constants[_idx]
#                     cnode.name  = str(constants[_idx])
                    
#                     individual[i] = cnode
#                 _idx += 1

#             _f = compiler(expr=individual)
#             _x_data = (x['{}'.format(i)] for i in list(x.columns))
#             _y = _f(*_x_data)
#             if np.isscalar(_y):         # Avoid scalar errors.
#                 _y = np.full(x.shape[0], float(_y))
#             elif len(_y.shape) == 0:    # Avoid errors due to singleton arrays.
#                 _y = _y.item()
#                 _y = np.full(x.shape[0], float(_y))
#             elif type(_y) is pd.Series: # Set it to np.ndarray
#                 _y = _y.values
#             y = y.values.reshape(len(_y))
#             residual = y - _y
            
#             return residual
#         # <<<<< func of const opt

#         filter_results1 = FVD_filter(individual, 
#                                     function_filter = self.function_filter, 
#                                     variable_filter = self.variable_filter, 
#                                     xydomain_filter = False, 
#                                     constonly_filter = self.constonly_filter,
#                                     function_group=[ 
#                                         ['sqrt'], 
#                                         ['square', 'cube'], 
#                                         ['ln', 'log', 'exp']],
#                                     x_domain=None, 
#                                     y_domain=None, 
#                                     y_pred=None, 
#                                     equal=self.domain_equal, 
#                                     )
#         if filter_results1[0]:
#             pass
#         else:
#             individual.state = filter_results1[1]
#             return np.inf,
        
#         # filter_results = True, 'all_pass'
#         # if self.function_filter or self.variable_filter or self.constonly_filter:
#         #     filter_results = FVD_filter(individual, 
#         #                                 function_filter = self.function_filter, 
#         #                                 variable_filter = self.variable_filter, 
#         #                                 xydomain_filter = False, 
#         #                                 function_group=[ ['sqrt'], ['square', 'cube'], ['ln', 'log', 'exp']],
#         #                                 x_domain=None, 
#         #                                 y_domain=None, 
#         #                                 y_pred=None, 
#         #                                 equal=self.domain_equal, 
#         #                                 constonly_filter = self.constonly_filter)
#         # if filter_results[0]:
#         #     pass
#         # else:
#         #     return np.inf,

#         _is_const = [isfloat(n.name) for n in individual]

#         if sum(_is_const):
#             constant_nodes = [e for e, i in enumerate(_is_const) if i]
#             constants0 = [individual[idx].value for idx in constant_nodes]
#             self.temporary = constants0
#             self.temporary2 = ['']
#             if 0 < len(constants0):
#                 try:
#                     _result=least_squares(_func, x0 = constants0, args=(x, y_true, individual, self.toolbox_.compile, constant_nodes), method='lm')
#                     _idx = 0
#                     if _result.status >= 2:
#                         for i in constant_nodes:
#                             cnode = copy.deepcopy(individual[i])
#                             cnode.value = _result.x[_idx]
#                             cnode.name  = str(_result.x[_idx])
#                             individual[i] = cnode
#                             _idx += 1
#                         self.root = 'A'
#                         self.temporary2 = [_result.x, _result.success, 'status', _result.status, _result.message]
                        
#                     else:
#                         for i in constant_nodes:
#                             cnode = copy.deepcopy(individual[i])
#                             if ~np.isnan(constants0[_idx]):
#                                 cnode.value = constants0[_idx]
#                                 cnode.name  = str(constants0[_idx])
                            
#                             individual[i] = cnode
#                             _idx += 1
#                         self.root = 'B'
                        
#                 except:
#                     _idx = 0
#                     for i in constant_nodes:
#                         cnode = copy.deepcopy(individual[i])
#                         if ~np.isnan(constants0[_idx]):
#                             cnode.value = constants0[_idx]
#                             cnode.name  = str(constants0[_idx])
                            
#                         individual[i] = cnode
#                         _idx += 1
#                     self.root = 'C'

#         filter_results2 = FVD_filter(individual, 
#                                     function_filter = False, 
#                                     variable_filter = False, 
#                                     xydomain_filter = self.xydomain_filter,
#                                     constonly_filter = False,
#                                     x_domain=self.x_domain, 
#                                     y_domain=self.y_domain, 
#                                     y_pred=self._pred(self.x_domain, individual), 
#                                     equal=self.domain_equal, 
#                                     )
#         if filter_results2[0]:
#             pass
#         else:
#             individual.state = filter_results1[1] + filter_results2[1]
#             return np.inf,
        
#         # filter_results = FVD_filter(individual, 
#         #                             function_filter = False, 
#         #                             variable_filter = False, 
#         #                             xydomain_filter = self.xydomain_filter,
#         #                             x_domain=self.x_domain, 
#         #                             y_domain=self.y_domain, 
#         #                             y_pred=self._pred(self.x_domain, individual), 
#         #                             equal=self.domain_equal, constonly_filter = False)
#         # if filter_results[0]:
#         #     pass
#         # else:
#         #     return np.inf,
        
#         y_pred = self._pred(self.fit_x_, individual)
#         individual.state = filter_results1[1] + filter_results2[1]

#         try:
#             if self.metric == 'mae':
#                 error = mean_absolute_error(y_true, y_pred)
#             elif self.metric == 'rmse':
#                 error = mean_squared_error(y_true, y_pred, squared=False)
#             elif self.metric == 'mse':
#                 error = mean_squared_error(y_true, y_pred, squared=True)
#             else:
#                 error = mean_absolute_error(y_true, y_pred)
#         except:
#             individual.state += '=opt-error'
#             error = np.inf

#         error_noise = None
#         if 1 == self.stabilize:
#             """stability for variable noise"""
            
#             if self.s_gnoise:
#                 # _x_gnoise = self.fit_x_
#                 _x_gnoise = self.fit_x_ + self.s_lmd2*gnoise(self.fit_x_)
#                 # print(self.fit_x_)
#                 # print(_x_gnoise)
#                 y_pred_noise = self._pred(_x_gnoise, individual)
#                 try:
#                     if self.metric == 'mae':
#                         error_noise = mean_absolute_error(y_pred, y_pred_noise)
#                     elif self.metric == 'rmse':
#                         error_noise = mean_squared_error(y_pred, y_pred_noise, squared=False)
#                     elif self.metric == 'mse':
#                         error_noise = mean_squared_error(y_pred, y_pred_noise, squared=True)
#                     else:
#                         error_noise = mean_absolute_error(y_pred, y_pred_noise)
#                 except:
#                     error_noise = np.inf
#             else:
#                 # _x_gnoise = self.fit_x_
#                 _std = np.std(self.fit_x_, axis=0)
#                 _x_noise_p = self.fit_x_ + self.s_lmd2*_std
#                 _x_noise_m = self.fit_x_ - self.s_lmd2*_std
#                 # print("org")
#                 # print(self.fit_x_.iloc[0,:])
#                 # print("noise p")
#                 # print(_x_noise_p.iloc[0,:])
#                 # print("noise p")
#                 # print(_x_noise_m.iloc[0,:])
#                 y_pred_noise = self._pred(pd.concat([_x_noise_p,_x_noise_m]), individual)
#                 # print(pd.concat([_x_noise_p,_x_noise_m]).shape, _x_noise_p.shape)

#                 try:
#                     if self.metric == 'mae':
#                         error_noise = mean_absolute_error(pd.concat([y_pred,y_pred]), y_pred_noise)
#                     elif self.metric == 'rmse':
#                         error_noise = mean_squared_error(pd.concat([y_pred,y_pred]), y_pred_noise, squared=False)
#                     elif self.metric == 'mse':
#                         error_noise = mean_squared_error(pd.concat([y_pred,y_pred]), y_pred_noise, squared=True)
#                     else:
#                         error_noise = mean_absolute_error(pd.concat([y_pred,y_pred]), y_pred_noise)
#                 except:
#                     error_noise = np.inf

#         elif 2 == self.stabilize:
#             """stability for regression coefficient"""
#             #print(individual)
#             _org_coeffs = {}
#             for idx,node in enumerate(individual):
#                 if node.arity == 0 and isfloat(node.name):
#                     #print(individual, individual[idx].value)
#                     _org_coeffs[idx] = node.value
#                     #print(type(individual[idx].value))
#                     individual[idx].value = node.value*(1.0 + self.s_clmd2)
#                     #print(individual, individual[idx].value)
#                     #print(_org, individual[idx].value, individual[idx].value-_org)

#             #print(individual)
#             y_pred_noise_p = self._pred(self.fit_x_, individual)
#             """restore org coeffs"""
#             for idx,value in _org_coeffs.items():
#                 individual[idx].value = value

#             _org_coeffs = {}
#             for idx,node in enumerate(individual):
#                 if node.arity == 0 and isfloat(node.name):
#                     _org = node.value
#                     _org_coeffs[idx] = node.value
#                     individual[idx].value = node.value*(1.0 - self.s_clmd2)
#                     #print(_org, individual[idx].value, individual[idx].value-float(_org))
                    
#             #print(individual)
#             y_pred_noise_m = self._pred(self.fit_x_, individual)
#             """restore org coeffs"""
#             for idx,value in _org_coeffs.items():
#                 individual[idx].value = value
                
#             #print(y_pred.iloc[0], y_pred_noise_p.iloc[0], y_pred_noise_m.iloc[0])
                

#             try:
#                 if self.metric == 'mae':
#                     error_noise = mean_absolute_error(pd.concat([y_pred,y_pred]), pd.concat([y_pred_noise_p, y_pred_noise_m]))
#                 elif self.metric == 'rmse':
#                     error_noise = mean_squared_error(pd.concat([y_pred,y_pred]), pd.concat([y_pred_noise_p, y_pred_noise_m]), squared=False)
#                 elif self.metric == 'mse':
#                     error_noise = mean_squared_error(pd.concat([y_pred,y_pred]), pd.concat([y_pred_noise_p, y_pred_noise_m]), squared=True)
#                 else:
#                     error_noise = mean_absolute_error(pd.concat([y_pred,y_pred]), pd.concat([y_pred_noise_p, y_pred_noise_m]))
#             except:
#                 error_noise = np.inf
            
#         else:
#             raise RuntimeError(f"stabilize={self.stabilize} is not implemented. use stabilize=1")
#             # self.s_clmd2 = s_clmd2

#         # print("ratio:", error_noise/error)

#         if error_noise is not None:
#             error += self.s_lmd1*error_noise
#             # return error + self.stb_lambda*error_noise, error, error_noise
            
#         return error,


MMS_COLUMNS = ['chembl-id', 'pot.(log,Ki)', 'pot.(nMol,Ki)', 'aromatic_smiles', 'non_stereo_aromatic_smieles',
               'all-chembl-ids', 'no.-meas.', 'pref_name', 'accession', 'natoms',
               'core', 'sub', 'sub_carbon_replacement', 'arorings', 'a_acc',
               'a_don', 'a_heavy', 'logP(o/w)', 'RBC', 'rings',
               'TPSA', 'vdw_vol', 'Weight']
MMS_COLRENAME = {"arorings": "arings", "a_acc": "acc", "a_don": "don", "logP(o/w)": "logp", "RBC": "rbc",
                 "TPSA": "tpsa", "Weight": "mw", "pot.(log,Ki)":"pot"}
MMS_FEATLIST = ["arings", "acc", "don", "logp", "rbc", # Rotatable Bond Counts
                "tpsa", "mw"]
MMS_PROPERTY = "pot"

config = configparser.ConfigParser()
config.read(sys.argv[1])
print(config.sections())

file       = config["FIGP"]["INPUT_FILE"]
result_dir = config["FIGP"]["RESULT_DIR"]
filter     = config["FIGP"]["FILTER"]
metric     = config["FIGP"]["METRIC"].lower() if "METRIC" in config["FIGP"].keys() else "rmse"
stabilize  = int(config["FIGP"]["STABILIZE"]) if "STABILIZE" in config["FIGP"].keys() else False
s_gnoise   = bool(config["FIGP"]["G_NOISE"]) if "G_NOISE" in config["FIGP"].keys() else False
s_lmd1     = float(config["FIGP"]["S_LMD1"]) if "S_LMD1" in config["FIGP"].keys() else 1.0
s_lmd2     = float(config["FIGP"]["S_LMD2"]) if "S_LMD2" in config["FIGP"].keys() else 0.5
s_clmd2    = float(config["FIGP"]["S_CLMD2"]) if "S_CLMD2" in config["FIGP"].keys() else 0.1

# STABILIZE=1 # 0: no stabilize, 1: variable stabilize, 2: regression coefficient stabilize, 3: variable and coefficient stabilize
# S_GNOISE=0 # 0: constant noise, 1: gaussian noise
# S_LMD1=1.0 # coefficient of stability (RMSE)
# S_LMD2=0.5 # magnitude of variable noise for stability check of variables
# S_CLMD2=0.1 # magnitude of regression coefficient noise


if filter == "FVD":
    function_filter = True
    variable_filter = True 
    xydomain_filter = True
    constonly_filter= True
    
elif filter == "FV":
    function_filter = True
    variable_filter = True 
    xydomain_filter = False
    constonly_filter= True

else:
    function_filter = False
    variable_filter = False
    xydomain_filter = False
    constonly_filter= True

print("FILE:", file)
print("OUTDIR:", result_dir)
print("FILTER:", filter)
print("METRIC:", metric)
print("STABILIZE:", stabilize)
print("S_LMD1:", s_lmd1)
print("S_LMD2:", s_lmd2)
print("S_CLMD2:", s_clmd2)
    
df = pd.read_table(file, index_col=0)
df = df.rename(columns=MMS_COLRENAME)
print(df.columns)
print(file, df["core"].iloc[0])
ndata = len(df.index)
ntrain = int(0.8*ndata)
print(f"ndata: {ndata}, ntrain: {ntrain}")

X = df.loc[:, MMS_FEATLIST]
y = df.loc[:, MMS_PROPERTY]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ntrain, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
ydomain = y.min(), y.max()

print("output_dir", result_dir)

res = dict()
for random_state in range(5):
    print("RANDOM STATE:", random_state)

    if 0 == stabilize:
        print("Symblic_Reg(org)")
        est = Symbolic_Reg( population_size=200,
                            generations=100,
                            tournament_size=5,
                            num_elite_select=1,
                            max_depth=4,
                            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'square', 'cube', 'ln', 'exp'),
                            metric=metric, 
                            p_crossover=0.7, 
                            p_mutation=0.2, 
                            random_state=random_state,
                            x_domain=X,
                            y_domain=ydomain,
                            var_max_trial=5000,
                            function_filter = function_filter, 
                            variable_filter = variable_filter, 
                            xydomain_filter = xydomain_filter,
                            constonly_filter= constonly_filter,
                            domain_equal    = (True, True),
                            results_dir=result_dir)
    elif 1 <= stabilize:
        print("Symblic_Reg_ST")
        est = Symbolic_Reg( population_size=200,
                            generations=100,
                            tournament_size=5,
                            num_elite_select=1,
                            max_depth=4,
                            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'square', 'cube', 'ln', 'exp'),
                            metric=metric, 
                            p_crossover=0.7, 
                            p_mutation=0.2, 
                            random_state=random_state,
                            x_domain=X,
                            y_domain=ydomain,
                            var_max_trial=5000,
                            function_filter=function_filter, 
                            variable_filter=variable_filter, 
                            xydomain_filter=xydomain_filter,
                            constonly_filter=constonly_filter,
                            domain_equal    =(True, True),
                            results_dir=result_dir,
                            stabilize=stabilize,
                            s_gnoise=s_gnoise,
                            s_lmd1=s_lmd1,
                            s_lmd2=s_lmd2,
                            s_clmd2=s_clmd2)


    # traininig
    est.fit(X_train, y_train)
    y_train_pred = est.predict(X_train)
    rmse_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred, squared=False)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    res[random_state] = (rmse_train, r2_train, est)

# save the training results
res_vals = sorted(res.values(), key=lambda x: (x[0],x[1]))
print("train all results: rmse, r2")
print([(val[0],val[1]) for val in res_vals])
print("TRAIN RESULTS: RMSE, R2")
print((res_vals[0][0],res_vals[0][1]))
best_model = res_vals[0][2]
best_model.save_all()

y_test_pred = best_model.predict(X_test)
rmse_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred, squared=False)
r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

print("TEST RESULTS: RMSE, R2")
print((rmse_test, r2_test))
