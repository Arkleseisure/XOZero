class Config:
    def __init__(self, wab_config):
        # root exploration noise
        self.root_dirichlet_alpha = wab_config['root_dirichlet_alpha']
        self.root_exploration_fraction = wab_config['root_exploration_fraction']

        # UCB formula
        self.pb_c_base = wab_config['pb_c_base']
        self.pb_c_init = wab_config['pb_c_init']

        # Training
        self.training_steps = 10000
        self.checkpoint_interval = 100
        self.checkpoint_game_num = 50
        self.batch_size = wab_config['batch_size']
        self.step_size = 4096
        self.batch_number = wab_config['batch_number']
        self.epochs = wab_config['epochs']
        self.val_split = 0
        self.processes = 4
        self.max_mem = 0.1

        # Neural net
        self.neural_net_blocks = wab_config['neural_net_blocks']
        self.num_simulations = wab_config['num_simulations']
        self.net_name = 'test'
