"""
Colgiht agent.
observations: [lane_num_vehicle, cur_phase]
reward: -queue_length
"""
import numpy as np
import os
from .agent import Agent
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from utils.make_mask_noise import make_guassion_noise, make_U_rand_noise
from inferences.state_filling import Diffusion_Predictor
from utils.batch_buffer import ReplayBuffer
import torch
def build_memory():
    return []


class CoLightDSIAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(CoLightDSIAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()
        self.device = dic_traffic_env_conf['device']
        # self.inference_model = Diffusion_Predictor((self.len_feature-8), self.num_actions, self.device, dic_traffic_env_conf['inference_config'], log_writer=False)
        self.inference_model = Diffusion_Predictor((self.len_feature), self.num_actions, self.device, dic_traffic_env_conf['inference_config'], log_writer=False)

        self.timestep = None
        self.long_state_con = torch.zeros([self.num_agents, 4, self.len_feature]).to(self.dic_traffic_env_conf['device'])
        if cnt_round == 0:
            # initialization
            self.q_network = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)),
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        # load_sota
        
        if self.dic_traffic_env_conf['is_test']:
            self.q_network_bar.load_weights(os.path.join(self.dic_traffic_env_conf['sota_path'], "round_{0}_inter_{1}.h5".format(cnt_round+79-4, self.intersection_id)), by_name=True)
            self.q_network.load_weights(os.path.join(self.dic_traffic_env_conf['sota_path'], "round_{0}_inter_{1}.h5".format(cnt_round+79, self.intersection_id)), by_name=True)
            #self.inference_model.load_model(os.path.join(self.dic_traffic_env_conf['sota_path'], "round_{0}_inter_0".format(cnt_round+79, self.intersection_id)),  int(self.device[-1]))
            #self.inference_model.load_model(os.path.join('model/colightDSI_old/anon_3_4_jinan_real', "round_{0}_inter_0".format(cnt_round+79, self.intersection_id)),  int(self.device[-1]))
            self.inference_model.load_model(self.dic_traffic_env_conf['diffusion_path'],  int(self.device[-1]))
    
    def _cal_len_feature(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                N += 8
            else:
                N += 12
        return N

    @staticmethod
    def MLP(ins, layers=None):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,dim]
        -outpout: [batch,#agents,dim]
        """
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)

        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        """
        input: [batch, agent, dim] feature
               [batch, agent, nei, agent] adjacency
        input:[bacth,agent,128]
        output:
              [batch, agent, dim]
        """
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)

        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)

        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr = Lambda(lambda x: tf.matmul(x[0], x[1]))([in_nei, neighbor_repr])

        # attention computation
        # [batch, agent, 1, dim]->[batch, agent, 1, h_dim*head]
        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        # [batch,agent,1,h_dim,head]->[batch,agent,head,1,h_dim]
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(agent_repr_head)

        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,h_dim_head]
        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        # [batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(neighbor_repr_head)

        # [batch,agent,head,1,h_dim]x[batch,agent,head,neighbor,h_dim]->[batch,agent,head,1,neighbor]
        att = Lambda(lambda x: K.softmax(tf.matmul(x[0], x[1], transpose_b=True)))([agent_repr_head,
                                                                                    neighbor_repr_head])
        # [batch,agent,nv,1,neighbor]->[batch,agent,head,neighbor]
        att_record = Reshape((self.num_agents, head, self.num_neighbors))(att)

        # self embedding again
        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 4, 2, 3)))(
            neighbor_hidden_repr_head)
        out = Lambda(lambda x: K.mean(tf.matmul(x[0], x[1]), axis=2))([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal', name='MLP_after_relation_%d' % suffix)(
            out)
        return out, att_record

    def adjacency_index2matrix(self, adjacency_index):
        # [batch,agents,neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        lab = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return lab

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        # TODO
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(self.dic_traffic_env_conf['PHASE'][s[i][feature][0]])
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])

            feats0.append(tmp)
        # [1, agent, dim]
        # feats = np.concatenate([np.array([feat1]), np.array([feat2])], axis=-1)
        feats = np.array([feats0])
        # [1, agent, nei, agent]
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj]

    def choose_action(self, is_test, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        """
        xs = self.convert_state_to_input(states)
        q_values = self.q_network(xs)
        # TODO: change random pattern
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0], axis=1)
            # trainning过程没有 attack所以这里timestep为0
            # if is_test is not True:
            #     timestep = 0
            #     recover_state = self.inference_model.ema_model(x.type(torch.float32), cur_action.type(torch.float32), self.long_state_con.type(torch.float32), timestep
            #                                         ).cpu().detach().numpy()
            #     xs[0][0] = recover_state
            #     q_values = self.q_network(xs)
            #     action = np.argmax(q_values[0], axis=1)
        import time
        ran = random.uniform(0, 1)
        if is_test:
            if self.dic_traffic_env_conf['NOISE_LEVEL'] == 0:   
                if self.dic_traffic_env_conf['NOISE_TYPE'] == 2:
                    timestep = int((self.dic_traffic_env_conf['NOISE_SCALE'] + 0.03) / 0.1 + 0.0001)   # 0.05
                    noise_rpt = self.num_agents
                    xs = self.convert_state_to_input(states)
                    actions = []
                    state_real_tensor = torch.tensor(xs[0]).to(self.dic_traffic_env_conf['device'])
                    rpt_state = torch.repeat_interleave(state_real_tensor, repeats=noise_rpt, dim=0)
                    rpt_noised_state = rpt_state + (2 * self.dic_traffic_env_conf['NOISE_SCALE']  * torch.rand_like(rpt_state) - self.dic_traffic_env_conf['NOISE_SCALE'] )
                    rpt_long_state_con_array = []
                    if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                        for k in range(self.num_agents):
                            cur_action = torch.tensor(np.eye(4)[np.array(action)]).to(self.dic_traffic_env_conf['device'])[k]
                            rpt_con_action = torch.repeat_interleave(cur_action.reshape(1, -1), repeats=noise_rpt, dim=0)
                            rpt_long_state_con = torch.repeat_interleave(self.long_state_con[k].reshape(1,self.long_state_con.shape[1], -1), repeats=noise_rpt, dim=0)
                            rpt_state_predict = self.inference_model.denoise_state(rpt_noised_state[:,k,:].type(torch.float32), rpt_con_action.type(torch.float32), rpt_long_state_con.type(torch.float32), timestep, method="mean").cpu()
                            rpt_noised_state[:,k,:] = rpt_state_predict
                            rpt_long_state_con_array.append(rpt_state_predict)
                    state_phase = torch.tensor(xs[1]).to(self.dic_traffic_env_conf['device'])
                    rpt_state_phase = torch.repeat_interleave(state_phase, repeats=noise_rpt, dim=0)
                    rpt_noised_action = self.q_network([rpt_noised_state.cpu().numpy(), rpt_state_phase.cpu().numpy()])
                    x = torch.tensor(xs[0][0]).to(self.dic_traffic_env_conf['device'])
                    for k in range(self.num_agents):
                        # 输出action
                        tmp_action = np.argmax(rpt_noised_action[:,k,:], axis=1)
                        # 从重复的噪声中选择q最小的，索引，从tmp_action中根据索引找到action
                        tmp_q = []
                        for j in range(noise_rpt):
                            tmp_q.append(rpt_noised_action[j,k,tmp_action[j]].cpu().numpy().tolist())
                        idx = tmp_q.index(min(tmp_q))
                        actions.append(tmp_action[idx])
                        if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                            x[k] = rpt_long_state_con_array[k][idx]
                    if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                        self.long_state_con = torch.cat([self.long_state_con[:, 1:], x.view(self.long_state_con.shape[0],1,-1)], dim=1)

                    action = actions
                if self.dic_traffic_env_conf['NOISE_TYPE'] == 3:
                    timestep = int((self.dic_traffic_env_conf['NOISE_SCALE'] + 0.03) / 0.1 + 0.0001)       # 0.05
                    noise_rpt = self.num_agents
                    xs = self.convert_state_to_input(states)
                    actions = []
                    state_real_tensor = torch.tensor(xs[0]).to(self.dic_traffic_env_conf['device'])
                    rpt_state = torch.repeat_interleave(state_real_tensor, repeats=noise_rpt, dim=0)
                    rpt_noised_state = rpt_state + (2 * self.dic_traffic_env_conf['NOISE_SCALE']  * torch.rand_like(rpt_state) - self.dic_traffic_env_conf['NOISE_SCALE'] )
                    rpt_long_state_con_array = []
                    if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                        for k in range(self.num_agents):
                            cur_action = torch.tensor(np.eye(4)[np.array(action)]).to(self.dic_traffic_env_conf['device'])[k]
                            rpt_con_action = torch.repeat_interleave(cur_action.reshape(1, -1), repeats=noise_rpt, dim=0)
                            rpt_long_state_con = torch.repeat_interleave(self.long_state_con[k].reshape(1,self.long_state_con.shape[1], -1), repeats=noise_rpt, dim=0)
                            rpt_state_predict = self.inference_model.denoise_state(rpt_noised_state[:,k,:].type(torch.float32), rpt_con_action.type(torch.float32), rpt_long_state_con.type(torch.float32), timestep, method="mean").cpu()
                            rpt_noised_state[:,k,:] = rpt_state_predict
                            rpt_long_state_con_array.append(rpt_state_predict)
                    state_phase = torch.tensor(xs[1]).to(self.dic_traffic_env_conf['device'])
                    rpt_state_phase = torch.repeat_interleave(state_phase, repeats=noise_rpt, dim=0)
                    rpt_noised_all_q = self.q_network([rpt_noised_state.cpu().numpy(), rpt_state_phase.cpu().numpy()])
                    x = torch.tensor(xs[0][0]).to(self.dic_traffic_env_conf['device'])
                    original_action_all_q = self.q_network([rpt_state.cpu().numpy(), rpt_state_phase.cpu().numpy()])
                    for k in range(self.num_agents):
                        # 输出action
                        rpt_noised_action_q = rpt_noised_all_q[:,k,:]
                        original_action_q = np.mean(original_action_all_q[:,k,:], axis=0)
                        rpt_action = np.repeat(original_action_q.reshape(1, -1), repeats=noise_rpt, axis=0)
                        difference = np.sum(np.abs(rpt_action - rpt_noised_action_q), 1)
                        idx = np.argmax(difference)
                        action = np.argmax(rpt_noised_action_q[idx])
                        actions.append(action)
                        if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                            x[k] = rpt_long_state_con_array[k][idx]
                    if self.dic_traffic_env_conf['is_inference'] and ran < self.dic_traffic_env_conf['DETECT_RATE']:
                        self.long_state_con = torch.cat([self.long_state_con[:, 1:], x.view(self.long_state_con.shape[0],1,-1)], dim=1)

                    action = actions
                    
        
        return action

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        slice_size = len(memory[0])
        _adjs = []
        # state : [feat1, feat2]
        # feati : [agent1, agent2, ..., agentn]
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])
                # TODO
                _state[j].append(self._concat_list([state[used_feature[i]] for i in range(len(used_feature))]))
                _next_state[j].append(self._concat_list([next_state[used_feature[i]] for i in range(len(used_feature))]))
            _adjs.append(_adj)
        # [batch, agent, nei, agent]
        _adjs2 = self.adjacency_index2matrix(np.array(_adjs))

        # [batch, 1, dim] -> [batch, agent, dim]
        _state2 = np.concatenate([np.array(ss) for ss in _state], axis=1)

        _next_state2 = np.concatenate([np.array(ss) for ss in _next_state], axis=1)

        self.replay_buffer = ReplayBuffer((np.swapaxes(_state2,0,1), np.eye(4)[np.array(_action)], np.swapaxes(_next_state2,0,1), np.array(_reward)[:,:, np.newaxis]),
                                    int(self.dic_traffic_env_conf['RUN_COUNTS']/self.dic_traffic_env_conf['MIN_ACTION_TIME']), self.device)

        target = self.q_network([_state2, _adjs2])
        next_state_qvalues = self.q_network_bar([_next_state2, _adjs2])
        # [batch, agent, num_actions]
        final_target = np.copy(target)
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])

        self.Xs = [_state2, _adjs2]
        self.Y = final_target

    def build_network(self, MLP_layers=[32, 32]):
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)
        In = list()
        # In: [batch,agent,dim]
        # In: [batch,agent,neighbors,agents]
        In.append(Input(shape=(self.num_agents, self.len_feature), name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))

        feature = self.MLP(In[0], MLP_layers)

        # feature:[batch,agents,feature_dim]
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
            if CNN_layer_index == 0:
                h, _ = self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
            else:
                h, _ = self.MultiHeadsAttModel(
                    h,
                    In[1],
                    d_in=MLP_layers[-1],
                    h_dim=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    head=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                )
        # action prediction layer
        # [batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions, kernel_initializer='random_normal', name='action_layer')(h)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model = Model(inputs=In, outputs=out)

        model.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                      loss=self.dic_agent_conf["LOSS_FUNCTION"])
        model.summary()
        return model

    def train_network(self):    
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        if self.dic_traffic_env_conf['is_test'] is not True:
           #self.inference_model.train(self.replay_buffer, self.dic_traffic_env_conf['inference_epoch'], self.replay_buffer.batch_sample, self.dic_traffic_env_conf['log_writer'])
            self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                            verbose=2, validation_split=0.3, callbacks=[early_stopping])

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])

        return network

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        self.inference_model.load_model(os.path.join(self.dic_path["PATH_TO_MODEL"], file_name))
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        self.inference_model.save_model(os.path.join(self.dic_path["PATH_TO_MODEL"], file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], file_name))


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        # [batch,agent,dim]->[batch,1,agent,dim]
        # [batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
