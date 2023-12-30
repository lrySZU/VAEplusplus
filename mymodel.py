import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, item_num, hidden_size, batch_size, top_k, dropout_rate=0.5, lr=1e-3):
        super().__init__()

        print(f'item_num={item_num}')
        self.device = torch.device('cuda')

        # examination data for mean pred (in target representation refinement)
        self.layer_u_exa = nn.Linear(in_features=item_num, out_features=hidden_size).to(self.device)
        self.init_weights(self.layer_u_exa)

        # purchase + examination for mean pred (in target representation enhancement)
        self.layer_u_mix = nn.Linear(in_features=item_num, out_features=hidden_size).to(self.device)
        self.init_weights(self.layer_u_mix)

        # purchase data for mean and std pred in target encoder
        self.layer_u_pur = nn.Linear(in_features=item_num, out_features=2 * hidden_size).to(self.device)
        self.init_weights(self.layer_u_pur)

        # in target representation refinement part, zu_pe + mu_u_exa -> zu_pr to get final purchase representation
        self.layer_mlp = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size).to(self.device)
        self.init_weights(self.layer_mlp)

        # zu_pred_softmax -> xu_pred in decoder_linear when the final xu need a further softmax
        self.decoder_linear = nn.Linear(in_features=hidden_size, out_features=item_num).to(self.device)
        self.init_weights(self.decoder_linear)

        # in transfer gating network, a ratio to multiply the mean_pur and (1 - the ratio) to multiply the mean_mix
        self.layer_g = nn.Linear(in_features=2 * hidden_size, out_features=1).to(self.device)
        self.init_weights(self.layer_g)

        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.lr = lr
        self.anneal = 0.2  # why 0.2?
        self.top_k = top_k
        self.input_exa = torch.empty((self.batch_size, self.item_num), requires_grad=False, dtype=torch.float,
                                     device=self.device)
        self.input_pur = torch.empty((self.batch_size, self.item_num), requires_grad=False, dtype=torch.float,
                                     device=self.device)
        self.input_mix = torch.empty((self.batch_size, self.item_num), requires_grad=False, dtype=torch.float,
                                     device=self.device)
        self.prediction_top_k = torch.empty((self.batch_size, self.item_num), requires_grad=False, dtype=torch.float,
                                            device=self.device)
        self.xu_pred = torch.empty((self.batch_size, self.item_num), requires_grad=True, dtype=torch.float,
                                   device=self.device)  # I think this need to compute the gradient
        self.kl = torch.empty((batch_size, 1), dtype=torch.float, requires_grad=True, device=self.device)
        self.logistic_loss = torch.empty((batch_size, 1), dtype=torch.float, requires_grad=True, device=self.device)
        self.loss = torch.empty((batch_size, 1), dtype=torch.float, requires_grad=True, device=self.device)
        self.cri = torch.nn.CrossEntropyLoss()

    def init_weights(self, layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight).to(self.device)
            torch.nn.init.normal_(layer.bias, 0, 0.001).to(self.device)

    def data_collect(self, input_pur, input_exa, input_mix):
        self.input_pur = torch.tensor(input_pur, dtype=torch.float, device=self.device)
        self.input_exa = torch.tensor(input_exa, dtype=torch.float, device=self.device)
        self.input_mix = torch.tensor(input_mix, dtype=torch.float, device=self.device)

    def get_u_exa(self):
        """
        in target representation refinement (TRR), examination u data for mean_u_exa
        learning the difference between their purchase preferences and examination preferences
        :return: mean_u_exa for further concatenate  | shape: (batch_size, hidden_size)
        """
        # F.normalize(): Divide a dimension by the norm corresponding to that dimension (2 norms by default)
        # why to normalize? we compute kl divergence with N ~ (0, 1)
        nor_u_exa = F.normalize(input=self.input_exa, dim=1, p=2)
        nor_u_exa = F.dropout(input=nor_u_exa, p=self.dropout_rate, training=self.training)
        mean_u_exa = self.layer_u_exa(nor_u_exa)
        mean_u_exa = torch.tanh(input=mean_u_exa)  # tanh function I add

        # why set the retain_grad to True? If, not set, Pytorch does not reserve gradients for non_leaf nodes
        mean_u_exa.retain_grad = True
        # mean_u_exa = mean_u_exa[:, :self.item_num]   zhuang says this is to item_num?
        return mean_u_exa

    def get_u_mix(self):
        """
        in target representation enhancement (TRE), pur + exa u data for mean_u_mix
        :return: mean_u_mix for further Transfer gating network  | shape:(batch_size, hidden_size)
        """
        nor_u_mix = F.normalize(input=self.input_mix, dim=1, p=2)
        nor_u_mix = F.dropout(input=nor_u_mix, p=self.dropout_rate, training=self.training)
        mean_u_mix = self.layer_u_mix(nor_u_mix)
        mean_u_mix = torch.tanh(input=mean_u_mix)

        mean_u_mix.retain_grad = True  # I added
        # mean_u_mix = mean_u_mix[:, :self.item_num]   zhuang says this is to item_num?
        return mean_u_mix

    def mlp_to_decoder(self, zu_p_enhanced, mu_exa):
        """
        [zu_pe, mu_exa] cat in the dim 1 -> zu_pr
        :param zu_p_enhanced: users distribution after target representation enhancement
        :param mu_exa: users exa data mean preference 
        :return: [zu] for decoder | zu.shape:(batch_size, hidden_num)
        """
        catted = torch.cat(tensors=(zu_p_enhanced, mu_exa), dim=1).to(self.device)  # (batch_size, 2 * hidden_size)
        zu_pred = self.layer_mlp(catted)
        # zu_pred = torch.softmax(input=zu_pred, dim=1).to(self.device)

        self.xu_pred = self.decoder_linear(zu_pred)
        # print(f'self.xu_pred.shape={self.xu_pred.shape}')

        return self.xu_pred

    def transfer_gating_network(self, mean_u_mix):
        nor_u_pur = F.normalize(input=self.input_pur, dim=1, p=2)

        # encoder: for mean_u_pur + var_u_pur
        enc_u_pur = self.layer_u_pur(nor_u_pur)
        enc_u_pur = torch.tanh(enc_u_pur)
        mean_u_pur = enc_u_pur[:, :self.hidden_size]  # (batch_size, hidden_size)
        var_u_pur = torch.exp(input=enc_u_pur[:, self.hidden_size:])

        mean_u_pur.retain_grad = True
        var_u_pur.retain_grad = True

        # for ratio g
        mean_combined = torch.cat(tensors=(mean_u_pur, mean_u_mix), dim=1)
        g = self.layer_g(mean_combined)
        g = torch.sigmoid(g)
        mean_enhanced = mean_u_pur * g + mean_u_mix * (1 - g)  # need to be similar to N(0, 1)

        # for KL computation and re-parameterize
        kl = torch.mean(input=torch.sum(0.5 * (-var_u_pur + torch.exp(var_u_pur) + mean_enhanced ** 2 - 1), dim=1))
        var_u_pur = torch.exp(var_u_pur / 2)

        return mean_enhanced, var_u_pur, kl

    def process(self):
        mean_u_exa = self.get_u_exa()
        mean_u_mix = self.get_u_mix()

        mean_enhanced, var_u_pur, kl = self.transfer_gating_network(mean_u_mix)
        epsilon = torch.randn(size=var_u_pur.shape, device=self.device)
        z_pred_pre = mean_enhanced + self.training * torch.mul(input=epsilon, other=var_u_pur)  # similar N~(0, 1)

        self.xu_pred = self.mlp_to_decoder(z_pred_pre, mean_u_exa)
        self.kl = kl

    def compute_loss(self):
        """
        zu_pred_softmax ~ (0.1) so is zu_pred or the zu_pred_softmax for output?
        I think is zu_pred because zu_pred ~ N(0, Id) I just make the softmax to larger the gap for cross_entropy_loss
        :return:
        """

        self.logistic_loss = torch.mean(input=torch.sum(-torch.log_softmax(self.xu_pred, dim=1) * self.input_pur,
                                                        dim=-1))
        self.loss = self.logistic_loss + self.anneal * self.kl

    def forward(self, input_pur, input_exa, input_mix):
        self.data_collect(input_pur, input_exa, input_mix)
        self.process()
        self.compute_loss()
        return self.loss, self.xu_pred
