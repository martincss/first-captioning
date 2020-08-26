from params import running_on_cluster

if running_on_cluster():

    search_name = None

    grid = [
            {
            'embedding_dim': [512],
            'units': [512],
            'lstm_units': [1800],
            'n_layers_init': [1],
            'n_layers_att': [2],
            'lambda_reg':[1.],
            'optimizer':['Adam'],
            'learning_rate':[0.01],
            'init_dropout': [0.5],
            'attn_dropout': [0],
            'lstm_dropout': [0],
            'logit_dropout': [0.5],
            'l1_reg':[0.],
            'l2_reg':[0.]
            }
            ]

else:

    search_name = None

    grid = [
            {
            'embedding_dim': [64],
            'units': [64],
            'lstm_units': [60],
            'n_layers_init': [1],
            'n_layers_att': [2],
            'lambda_reg':[1],
            'optimizer':['Adam'],
            'learning_rate':[0.01],
            'init_dropout': [0.],
            'attn_dropout': [0],
            'lstm_dropout': [0],
            'logit_dropout': [0.],
            'l1_reg':[0.],
            'l2_reg':[0.]
            }
            ]
