import openml
from flaml import AutoVW
import numpy as np
import string
import matplotlib.pyplot as plt

# Download dataset
# did = 42183
did = 41506
ds = openml.datasets.get_dataset(did)
target_attribute = ds.default_target_attribute
data = ds.get_data(target=target_attribute, dataset_format='array')
X, y = data[0], data[1]
print(X.shape, y.shape)

# Convert the openml dataset into vowpalwabbit examples
NS_LIST = list(string.ascii_lowercase) + list(string.ascii_uppercase)
max_ns_num = 10 # the maximum number of namespaces
orginal_dim = X.shape[1]
max_size_per_group = int(np.ceil(orginal_dim / float(max_ns_num)))
# sequential grouping
group_indexes = []
for i in range(max_ns_num):
    indexes = [ind for ind in range(i * max_size_per_group,
                min((i + 1) * max_size_per_group, orginal_dim))]
    if len(indexes) > 0:
        group_indexes.append(indexes)

vw_examples = []
for i in range(X.shape[0]):
    ns_content = []
    for zz in range(len(group_indexes)):
        ns_features = ' '.join('{}:{:.6f}'.format(ind, X[i][ind]) for ind in group_indexes[zz])
        ns_content.append(ns_features)
    ns_line = '{} |{}'.format(str(y[i]), '|'.join('{} {}'.format(NS_LIST[j], ns_content[j]) for j in range(len(group_indexes))))
    vw_examples.append(ns_line)
print('openml example:', y[0])
print('vw example:', vw_examples[0])

from sklearn.metrics import mean_squared_error

def online_learning_loop(iter_num, vw_examples, vw_alg):
    """Implements the online learning loop.
    """
    print('Online learning for', iter_num, 'steps...')
    loss_list = []
    for i in range(iter_num):
        vw_x = vw_examples[i]
        # print(vw_x)
        y_true = float(vw_examples[i].split('|')[0])
        # print('s',y_true)
        # predict step
        y_pred = vw_alg.predict(vw_x)
        # learn step
        # print(y_pred)
        vw_alg.learn(vw_x)
        # calculate one step loss
        loss = mean_squared_error([y_pred], [y_true])
        loss_list.append(loss)
    return loss_list

max_iter_num = 10000  # or len(vw_examples)

from vowpalwabbit import pyvw
''' create a vanilla vw instance '''
vanilla_vw = pyvw.vw()

# online learning with vanilla VW
loss_list_vanilla = online_learning_loop(max_iter_num, vw_examples, vanilla_vw)
print('Final progressive validation loss of vanilla vw:', sum(loss_list_vanilla)/len(loss_list_vanilla))

''' import AutoVW class from flaml package '''


'''create an AutoVW instance for tuning namespace interactions'''
# configure both hyperparamters to tune, e.g., 'interactions', and fixed arguments about the online learner,
# e.g., 'quiet' in the search_space argument.
autovw_ni = AutoVW(max_live_model_num=5, search_space={'interactions': AutoVW.AUTOMATIC})
#, 'quiet': ''
# online learning with AutoVW
loss_list_autovw_ni = online_learning_loop(max_iter_num, vw_examples, autovw_ni)
print('Final progressive validat gion loss of autovw:', sum(loss_list_autovw_ni)/len(loss_list_autovw_ni))



from flaml.tune import loguniform
''' create another AutoVW instance for tuning namespace interactions and learning rate'''
# set up the search space and init config
search_space_nilr = {'interactions': AutoVW.AUTOMATIC, 'learning_rate': loguniform(lower=2e-10, upper=1.0)}
init_config_nilr = {'interactions': set(), 'learning_rate': 0.5}
# create an AutoVW instance
autovw_nilr = AutoVW(max_live_model_num=5, search_space=search_space_nilr, init_config=init_config_nilr)

# online learning with AutoVW
loss_list_autovw_nilr = online_learning_loop(max_iter_num, vw_examples, autovw_nilr)
print('Final progressive validation loss of autovw_nilr:', sum(loss_list_autovw_nilr)/len(loss_list_autovw_nilr))

plt.figure(figsize=(8, 6))
def plot_progressive_loss(obj_list, alias, result_interval=1):
    """Show real-time progressive validation loss
    """
    avg_list = [sum(obj_list[:i]) / i for i in range(1, len(obj_list))]
    total_obs = len(avg_list)
    warm_starting_point = 10 #0
    plt.plot(range(warm_starting_point, len(avg_list)), avg_list[warm_starting_point:], label = alias)
    plt.xlabel('# of data samples',)
    plt.ylabel('Progressive validation loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
# plot_progressive_loss(loss_list_vanilla, 'VanillaVW')
# plot_progressive_loss(loss_list_autovw_ni, 'AutoVW:NI')
# plt.show()

plot_progressive_loss(loss_list_vanilla, 'VanillaVW')
plot_progressive_loss(loss_list_autovw_ni, 'AutoVW:NI')
plot_progressive_loss(loss_list_autovw_nilr, 'AutoVW:NI+LR')
plt.show()