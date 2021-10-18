#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mlrose_hiive
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


# 
# #### 8-Queens Using Custom Fitness Function
# 

# In[2]:


def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i])                 and (state[j] != state[i] + (j - i))                 and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness


# In[4]:


state = np.array([1, 4, 1, 3, 5, 5, 2, 7])

# The fitness of this state should be 22
queens_max(state)


# In[5]:


fitness_cust = mlrose_hiive.CustomFitness(queens_max)


# In[27]:


problem_cust = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)


# In[10]:


schedule = mlrose_hiive.ExpDecay()
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem_cust, schedule = schedule, 
                                                      max_attempts = 10, max_iters = 1000, 
                                                      init_state = init_state, random_state = 1)


# In[12]:


print(best_state)


# In[13]:


print(best_fitness)


# In[26]:



best_state, best_fitness = mlrose_hiive.genetic_alg(problem_cust, mutation_prob = 0.2, max_attempts = 100,
                                              random_state = 2)


# In[ ]:





# In[16]:





# In[18]:





# In[19]:


best_state, best_fitness, _ = mlrose_hiive.genetic_alg(problem_cust, mutation_prob = 0.2, max_attempts = 100,
                                              random_state = 2)


# In[20]:


best_state


# In[21]:


best_fitness


# In[ ]:





# #### Max K Color using the GA algorithm

# In[21]:


from mlrose_hiive import SARunner, GARunner, NNGSRunner
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator
from ast import literal_eval
problem = MaxKColorGenerator().generate(seed=123456, number_of_nodes=10, max_connections_per_node=3, max_colors=3)


# In[3]:


state = problem.get_state()


# In[4]:


sa = SARunner(problem=problem,
              experiment_name='queens8_sa',
              output_directory=None, # note: specify an output directory to have results saved to disk
              seed=123456,
              iteration_list=2 ** np.arange(11),
              max_attempts=500,
              temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0],
              decay_list=[mlrose_hiive.GeomDecay])

# the two data frames will contain the results
df_run_stats, df_run_curves = sa.run()


# In[5]:


df_run_stats


# In[6]:


df_run_curves


# In[15]:


minimum_evaluations = best_runs['FEvals'].min()
best_fitness = df_run_curves['Fitness'].min()
best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]
best_init_temperature = best_curve_run['Temperature'].iloc()[0].init_temp
run_stats_best_run = df_run_stats[df_run_stats['schedule_init_temp'] == best_init_temperature]


best_state = run_stats_best_run[['schedule_current_value', 'schedule_init_temp', 'schedule_min_temp']].tail(1)


# In[16]:


best_state


# In[28]:


import matplotlib.pyplot as plt
import networkx as nx
state = literal_eval(run_stats_best_run['State'].tail(1).values[0])
color_indexes = literal_eval(run_stats_best_run['State'].tail(1).values[0])
ordered_state = [color_indexes[n] for n in problem.source_graph.nodes]
colors = ['lightcoral', 'lightgreen', 'yellow']
node_color_map = [colors[s] for s in ordered_state]

nx.draw(problem.source_graph,
        pos=nx.spring_layout(problem.source_graph, seed = 3),
        with_labels=True,
        node_color=node_color_map)
plt.show()


# In[ ]:


## Neural network

outcome = []
train_outcome = []

for i in x:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    result = confusion_matrix(y_test,predictions)
    accuracy = (result[0,0] + result[1,1])/(result[0,0] + result[1,1] + result[1,0] + result[0,1])
    outcome.append(accuracy)
    
    predictions_train = clf.predict(X_train)
    result_train = confusion_matrix(y_train,predictions_train)
    accuracy_train = (result_train[0,0] + result_train[1,1])/(result_train[0,0] + result_train[1,1] + result_train[1,0] + result_train[0,1])
    train_outcome.append(accuracy_train)

df_temp = list(zip(x, outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df = pd.DataFrame(df_temp,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df)

df_train = list(zip(x, train_outcome)) 
# Converting lists of tuples into
# pandas Dataframe.
df_train = pd.DataFrame(df_train,columns = ['proportion', 'accuracy'])
sns.regplot(x="proportion", y="accuracy", data=df_train)

