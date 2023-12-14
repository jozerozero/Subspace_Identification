import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer=1, hidden_dim=1024) -> None:
        super(MLP, self).__init__()
        model = []
        model += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layer):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class iVAE(nn.Module):

    def __init__(self, args, backbone_net=None) -> None:
        super(iVAE, self).__init__()

        self.args = args
        self.backbone_net = backbone_net
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())

        self.z_dim = args.z_dim

        self.s1_dim = args.z_s1_dim  # 2d
        self.s2_dim = args.z_s2_dim  # 128 d
        self.s3_dim = args.z_c1_dim  # 128, 256, 512 d
        self.s4_dim = args.z_c2_dim  # 10 d
        self.u_dim = args.u_dim  # 12 d

        # print(self.s1_dim, self.s2_dim, self.s3_dim, self.s4_dim, self.u_dim)

        dim = args.hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.backbone_net.out_features, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(), nn.Dropout())
        self.fc_mu = nn.Sequential(nn.Linear(dim, self.z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(dim, self.z_dim))

        self.src_centroid = torch.zeros([args.n_domains-1, self.args.num_classes, args.z_c1_dim])
        self.tgt_centroid = torch.zeros([self.args.num_classes, args.z_c1_dim])
        self.decay = args.decay

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(),
                                     nn.Linear(dim, self.backbone_net.out_features))

        if args.arch == "resnet18":
            self.classifier = nn.Sequential(nn.Linear(self.s2_dim + self.s3_dim + self.u_dim, dim),
                                            nn.BatchNorm1d(dim),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(dim, args.num_classes))

            self.domain_predictor = nn.Sequential(
                nn.Linear(self.s1_dim + self.s2_dim + self.s3_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(dim, args.n_domains))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.s2_dim + self.s3_dim + self.u_dim, args.num_classes))
            self.domain_predictor = nn.Sequential(
                nn.Linear(self.s1_dim + self.s2_dim + self.s3_dim, args.n_domains))

        # domain embedding
        self.u_embedding = nn.Embedding(10, args.u_dim)  # to be tuned

        print(self.encoder, self.fc_mu, self.fc_logvar)
        print(self.decoder, self.classifier)

        self.lambda_vae = args.lambda_vae

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = track
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def update_center2(self, domain_idx, src_feature, tgt_feature, src_truth_label, tgt_pseudo_label, tgt_y_estimated):
        self.src_centroid = self.src_centroid.to(src_feature.device)
        self.tgt_centroid = self.tgt_centroid.to(tgt_feature.device)

        n, d = src_feature.shape

        s_labels, t_labels = src_truth_label, tgt_pseudo_label
        
        ones = torch.ones_like(s_labels, dtype=torch.float32)
        zeros = torch.zeros(self.args.num_classes).to(src_feature.device)

        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        # print(len(s_labels))
        # print(len(t_labels))
        # exit()
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        zeros = torch.zeros(self.args.num_classes, d).to(src_feature.device)

        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), src_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), tgt_feature)

        cls_ones = torch.ones_like(s_n_classes.view(self.args.num_classes, 1))
        current_s_centroid = torch.div(s_sum_feature, torch.max(s_n_classes.view(self.args.num_classes, 1), cls_ones))
        current_t_centroid = torch.div(t_sum_feature, torch.max(t_n_classes.view(self.args.num_classes, 1), cls_ones))

        decay = self.decay
        src_centroid = (1-decay) * self.src_centroid[domain_idx, :, :] + decay * current_s_centroid
        tgt_centroid = (1-decay) * self.tgt_centroid + decay * current_t_centroid

        s_loss = torch.mean(torch.pow(src_centroid - tgt_centroid, 2), dim=1)
        semantic_loss = torch.sum(torch.mul(tgt_y_estimated, s_loss))

        self.src_centroid[domain_idx, :, :] = src_centroid.detach()
        self.tgt_centroid = tgt_centroid.detach()

        return semantic_loss

    def update_center(self, domain_idx, src_feature, tgt_feature, src_label, tgt_pseudo_label):
        
        batch_size, feature_dim = src_feature.shape

        src_ones = torch.ones_like(src_label, dtype=torch.float32)
        tgt_ones = torch.ones_like(tgt_pseudo_label, dtype=torch.float32)
        zeros = torch.zeros(self.args.num_classes).to(src_feature.device)

        src_num_per_cls = zeros.scatter_add(dim=0, index=src_label, src=src_ones)
        tgt_num_per_cls = zeros.scatter_add(dim=0, index=tgt_pseudo_label, src=tgt_ones)

        cls_ones = torch.ones_like(src_num_per_cls)
        src_num_per_cls = torch.max(src_num_per_cls, cls_ones)
        tgt_num_per_cls = torch.max(tgt_num_per_cls, cls_ones)

        centroid_container = torch.zeros(self.args.num_classes, self.args.self.s3_dim).to(src_feature.device)

        src_sum_feature = centroid_container.scatter(0, src_label.unsqueeze(dim=1).repeat(1, feature_dim), src_feature)
        tgt_sum_feature = centroid_container.scatter(0, tgt_pseudo_label.unsqueeze(dim=1).repeat(1, feature_dim), tgt_feature)

        current_src_centroid = torch.div(src_sum_feature, src_num_per_cls.unsqueeze(1))
        current_tgt_centroid = torch.div(tgt_sum_feature, tgt_num_per_cls.unsqueeze(1))

        src_centroid = (1 - self.decay) * self.src_centre[domain_idx, :, :] + self.decay * current_src_centroid
        tgt_centroid = (1 - self.decay) * self.tgt_centre + self.decay * current_tgt_centroid

        semantic_loss = self.centroid_mse_loss(src_centorid=src_centroid, tgt_centroid=tgt_centroid)
        semantic_loss = torch.mean(semantic_loss)
        self.src_centre[domain_idx, :, :] = src_centroid.detach()
        self.tgt_centre = tgt_centroid.detach()

        return semantic_loss
    
    def centroid_mse_loss(self, src_centorid, tgt_centroid):
        """
        :param src_centorid: the centorid of source domain, whose shape is [num_cls, feature_dim]
        :param tgt_centroid: the centorid of target domain, whose shape is [num_cls, feature_dim]
        :return: return the centorid level mse loss, whose shape is [num_cls]
        """
        return torch.mean(torch.sqrt(torch.pow(src_centorid - tgt_centroid, 2)), dim=1)

    def backbone(self, x, track_bn=False):
        self.track_bn_stats(track_bn)
        # print(x.shape)
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict(self, z, track_bn=False):
        self.track_bn_stats(track_bn)
        return self.classifier(z)

    def extract_feature(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x, track_bn)
        h = self.encoder(x)

        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        z_idx = 0
        z_s1 = z[:, z_idx: z_idx + self.s1_dim]

        z_idx += self.s1_dim
        z_s2 = z[:, z_idx: z_idx + self.s2_dim]

        z_idx += self.s2_dim
        z_c1 = z[:, z_idx: z_idx + self.s3_dim]

        z_idx += self.s3_dim
        z_c2 = z[:, z_idx:]

        return z, z_s1, z_s2, z_c1, z_c2, mu, log_var

    def encode(self, x, u, track_bn=False):

        # z, z_s1, z_s2, z_c1, z_c2 = self.extract_feature(x, u, track_bn=track_bn)
        # print(x.shape)
        # exit()
        h = self.encoder(x)

        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        z_idx = 0
        z_s1 = z[:, z_idx: z_idx + self.s1_dim]

        z_idx += self.s1_dim
        z_s2 = z[:, z_idx: z_idx + self.s2_dim]

        z_idx += self.s2_dim
        z_c1 = z[:, z_idx: z_idx + self.s3_dim]

        z_idx += self.s3_dim
        z_c2 = z[:, z_idx:]

        domain_embedding = self.u_embedding(u)

        pre_input = torch.hstack([z_s2, z_c1, domain_embedding])
        label_logit = self.predict(pre_input)

        pre_domain_input = torch.hstack([z_s1, z_s2, z_c1])
        # print(pre_domain_input.shape)
        # exit()

        domain_logit = self.domain_predictor(pre_domain_input)

        return z, z_s1, z_s2, z_c1, z_c2, label_logit, domain_logit, mu, log_var

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x)

        _, _, _, _, _, label_logit, domain_logit, _, _ = self.encode(x, u)

        return label_logit, domain_logit

    def get_parameters(self, base_lr=1.0):
        base_params = itertools.chain(self.encoder.parameters(),
                                      self.fc_mu.parameters(), 
                                      self.fc_logvar.parameters(), 
                                      self.decoder.parameters(),
                                      self.u_embedding.parameters(),
                                      self.classifier.parameters(),
                                      self.domain_predictor.parameters())

        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},
            {"params": base_params, "lr": 1.0 * base_lr}
        ]

        return params
