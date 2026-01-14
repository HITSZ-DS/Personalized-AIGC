import torch as t
from torch import nn
from models.aug_utils import EdgeDrop
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN(BaseModel):
	def __init__(self, data_handler):
		super(LightGCN, self).__init__(data_handler)

		self.data_handler = data_handler  # Store data_handler for accessing training matrix
		self.adj = data_handler.torch_adj

		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.keep_rate = configs['model']['keep_rate']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.edge_dropper = EdgeDrop()
		self.is_training = True
		self.final_embeds = None

	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)

	def forward(self, adj, keep_rate):
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embeds_list = [embeds]
		if self.is_training:
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)# / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]

	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		loss = bpr_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, 1.0)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds

	def sample_predict(self, sample_user_idxs):
		user_embeds, item_embeds = self.forward(self.adj, 1.0)
		self.is_training = False
		# 选取sample_user_idxs对应的user_embeds,我的sample_user_idxs是random_ids = torch.randint(0, configs['data']['user_num'], (10,))
		pck_users = user_embeds[sample_user_idxs]
		sample_preds = pck_users @ item_embeds.T

		# Efficient vectorized training mask creation
		# Convert to CSR format for efficient row slicing
		trn_mat_csr = self.data_handler.trn_mat.tocsr()
		sample_user_idxs_cpu = sample_user_idxs.cpu().numpy()
		trn_mat_slice = trn_mat_csr[sample_user_idxs_cpu]  # Shape: [num_sample_users, num_items]

		# Convert sparse matrix to dense tensor and move to device
		train_mask = t.from_numpy(trn_mat_slice.toarray()).float().to(sample_preds.device)

		# Apply mask to predictions (similar to _mask_predict)
		sample_preds = self._mask_predict(sample_preds, train_mask)

		# 计算sample_preds每一个user的top 10 item (after removing training items)
		_, sample_preds = t.topk(sample_preds, k=2)
		return sample_preds
