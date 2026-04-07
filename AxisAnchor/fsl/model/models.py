import torch
from torch import nn
from AxisAnchor.fsl.backbones import build_backbone
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from scipy.optimize import linear_sum_assignment
import random
from .aa import AAencoder
from .losses import cross_entropy_loss_for_junction, sigmoid_l1_loss, sigmoid_focal_loss
from .misc import non_maximum_suppression, get_junctions, plot_lines
from sklearn.cluster import DBSCAN
def argsort2d(arr):
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def nms_j(heatmap, delta=1):
    DX = [0, 0, 1, -1, 1, 1, -1, -1]
    DY = [1, -1, 0, 0, 1, -1, 1, -1]
    heatmap = heatmap.copy()
    disable = np.zeros_like(heatmap, dtype=np.bool)
    for x, y in argsort2d(heatmap):
        for dx, dy in zip(DX, DY):
            xp, yp = x + dx, y + dy
            if not (0 <= xp < heatmap.shape[0] and 0 <= yp < heatmap.shape[1]):
                continue
            if heatmap[x, y] >= heatmap[xp, yp]:
                disable[xp, yp] = True
    heatmap[disable] *= 0.6
    return heatmap
def post_jheatmap(heatmap, offset=None, delta=1):
    heatmap = nms_j(heatmap, delta=delta)
    # only select the best 1000 junctions for efficiency
    v0 = argsort2d(-heatmap)[:1000]
    confidence = -np.sort(-heatmap.ravel())[:1000]
    # v0 = argsort2d(-heatmap)[:250]
    # confidence = -np.sort(-heatmap.ravel())[:250]
    keep_id = np.where(confidence >= 1e-2)[0]
    if len(keep_id) == 0:
        return np.zeros((0, 3))

    confidence = confidence[keep_id]
    if offset is not None:
        v0 = np.array([v + offset[:, v[0], v[1]] for v in v0])
    v0 = v0[keep_id] + 0.5
    v0 = np.hstack((v0, confidence[:, np.newaxis]))
    return v0

def add_argument_with_cfg(parser, cfg, arg_name, cfg_name, help, mapping):
    
    parser.add_argument('--{}'.format(arg_name.replace('_','-')), 
        default = eval('cfg.{}'.format(cfg_name)),
        type = type(eval('cfg.{}'.format(cfg_name))),
        help = help
    )
    mapping[arg_name] = cfg_name

class WireframeDetector(nn.Module):
    def cli(self, cfg, argparser):
        cfg_mapping = {}
        sampling_parser = argparser.add_argument_group(title = 'sampling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(sampling_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('num_dyn_junctions','MODEL.PARSING_HEAD.N_DYN_JUNC', help = '[train] number of dynamic junctions')
        add_argument_lambda('num_dyn_positive_lines', 'MODEL.PARSING_HEAD.N_DYN_POSL', help ='[train] number of dynamic positive lines')
        add_argument_lambda('num_dyn_negative_lines','MODEL.PARSING_HEAD.N_DYN_NEGL', help='[train] number of dynamic negative lines')
        add_argument_lambda('num_dyn_natural_lines', 'MODEL.PARSING_HEAD.N_DYN_OTHR2', help='[train] number of dynamic line samples from the natural selection')

        matching_parser = argparser.add_argument_group(title = 'matching specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(matching_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('j2l_threshold','MODEL.PARSING_HEAD.J2L_THRESHOLD', help='[all] the matching distance (in pixels^2) between the junctions and the learned lines')
        add_argument_lambda('jmatch_threshold', 'MODEL.PARSING_HEAD.JMATCH_THRESHOLD', help='[train] the matching distance (in pixels) between the predicted and grountruth junctions')

        loi_parser = argparser.add_argument_group(title = 'LOI-pooling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(loi_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_points', 'MODEL.LOI_POOLING.NUM_POINTS', help='[train] the number of sampling points')
        add_argument_lambda('dim_junction', 'MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE', help='[train] the dim of junction features')
        add_argument_lambda('dim_edge', 'MODEL.LOI_POOLING.DIM_EDGE_FEATURE', help='[train] the dim of edge features')
        add_argument_lambda('dim_fc', 'MODEL.LOI_POOLING.DIM_FC', help='[train] the dim of fc features')

        aa_parser = argparser.add_argument_group(title = 'Line proposal specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(aa_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_residuals', 'MODEL.PARSING_HEAD.USE_RESIDUAL', help='[all] the number of distance residuals')
        self.cfg_mapping = cfg_mapping
        
    def configure(self, cfg, args):
        configure_list = []
        for key, value in self.cfg_mapping.items():
            if getattr(args,key) != eval('cfg.'+value):
                configure_list.extend([value,getattr(args,key)])
        cfg.merge_from_list(configure_list)
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.AA_encoder = AAencoder(cfg)
        self.backbone = build_backbone(cfg)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.topk_junctions = 300
        #Matcher
        self.j2l_threshold = cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD
        self.jmatch_threshold = cfg.MODEL.PARSING_HEAD.JMATCH_THRESHOLD
        self.jhm_threshold = cfg.MODEL.PARSING_HEAD.JUNCTION_HM_THRESHOLD

        # LOI POOLING
        self.n_pts0     = cfg.MODEL.LOI_POOLING.NUM_POINTS
        self.dim_junction_feature    = cfg.MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE
        self.dim_edge_feature = cfg.MODEL.LOI_POOLING.DIM_EDGE_FEATURE
        self.dim_fc     = cfg.MODEL.LOI_POOLING.DIM_FC


        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.use_residual = int(cfg.MODEL.PARSING_HEAD.USE_RESIDUAL)

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:].cuda())
        
        assert cfg.MODEL.LOI_POOLING.TYPE in ['softmax', 'sigmoid']
        assert cfg.MODEL.LOI_POOLING.ACTIVATION in ['relu', 'gelu']

        self.loi_cls_type = cfg.MODEL.LOI_POOLING.TYPE
        self.loi_layer_norm = cfg.MODEL.LOI_POOLING.LAYER_NORM
        self.loi_activation = nn.ReLU if cfg.MODEL.LOI_POOLING.ACTIVATION == 'relu' else nn.GELU        

        self.fc1 = nn.Conv2d(256, self.dim_junction_feature, 1)

        self.fc3 = nn.Conv2d(256, self.dim_edge_feature, 1)
        self.fc4 = nn.Conv2d(256, self.dim_edge_feature, 1)

        self.regional_head = nn.Conv2d(256, 1, 1)
        fc2 = [nn.Linear(self.dim_junction_feature*2 + (self.n_pts0-2)*self.dim_edge_feature*2, self.dim_fc),
                # self.loi_activation(),
        ]
        for i in range(2):
            fc2.append(self.loi_activation())
            fc2.append(nn.Linear(self.dim_fc,self.dim_fc))


        self.fc2 = nn.Sequential(*fc2)
        self.fc2_res = nn.Sequential(nn.Linear(2*(self.n_pts0-2)*self.dim_edge_feature, self.dim_fc),self.loi_activation())

        self.line_mlp = nn.Sequential(
            nn.Linear((self.n_pts0-2)*self.dim_edge_feature,128),
            nn.ReLU(True),
            nn.Linear(128,32),nn.ReLU(True),
            nn.Linear(32,1)
        )

        if self.loi_cls_type == 'softmax':
            self.fc2_head = nn.Linear(self.dim_fc, 2)
            self.loss = nn.CrossEntropyLoss(reduction='none')
        elif self.loi_cls_type == 'sigmoid':
            self.fc2_head = nn.Linear(self.dim_fc, 1)
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementError()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.train_step = 0
        self.loi_conv = nn.Conv2d(496, 496, kernel_size=(3, 3), padding=(1, 1))

    def bilinear_sampling(self, features, points):
        h,w = features.size(1), features.size(2)
        px, py = points[:,0], points[:,1]
        
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features[:, py0l, px0l] * (py1-py) * (px1 - px)+ features[:, py1l, px0l] * (py - py0) * (px1 - px)+ features[:, py0l, px1l] * (py1 - py) * (px - px0)+ features[:, py1l, px1l] * (py - py0) * (px - px0)

        return xp
    
    def get_line_points(self, lines_per_im):
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        return sampled_points
    
    def compute_loi_features(self, features_per_image, lines_per_im):
        device = features_per_image.device
        num_channels = features_per_image.shape[0]
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        tspan = self.tspan[...,1:-1]
        sampled_points = U[:,:,None].to(device) * tspan + V[:,:,None].to(device) * (1 - tspan) - 0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        xp = xp.reshape(features_per_image.shape[0],-1,tspan.numel()).permute(1,0,2).contiguous()

        return xp.flatten(1)
    def pooling(self, features_per_line):
        
        if self.training:
            logits = self.fc2(features_per_line)
            return logits
        
        if self.loi_cls_type == 'softmax':
            return self.fc2(features_per_line).softmax(dim=-1)[:,1]
        else:
            return self.fc2(features_per_line).sigmoid()[:,0]

    def forward(self, images, annotations = None, targets = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)



    def forward_test(self, images, annotations=None):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_verification': 0.0,
        }

        # Backbone 提取特征
        extra_info['time_backbone'] = time.time()
        outputs, features = self.backbone(images)

        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)
        output = outputs[0]

        md_pred = output[:, :3].sigmoid()
        dis_pred = output[:, 3:4].sigmoid()
        s_pred = output[:, 9:10, :, :]

        # total_params = count_parameters(self)
        # print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
        extra_info['time_backbone'] = time.time() - extra_info['time_backbone']

        batch_size = md_pred.size(0)
        assert batch_size == 1

        # 提取线段候选
        extra_info['time_proposal'] = time.time()
        lines_pred = self.decoding(md_pred, dis_pred,  None,scale=self.AA_encoder.dis_th)
        lines_pred = lines_pred.reshape(-1, 4)

        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']

        if lines_pred.numel() == 0:
            return None, extra_info

        extra_info['time_verification'] = time.time()

        # 提取线段特征
        e1_features = self.bilinear_sampling(loi_features[0], lines_pred[:, :2] - 0.5).t()
        e2_features = self.bilinear_sampling(loi_features[0], lines_pred[:, 2:] - 0.5).t()
        f1 = self.compute_loi_features(loi_features_thin[0], lines_pred)
        f2 = self.compute_loi_features(loi_features_aux[0], lines_pred)
        line_features = torch.cat((e1_features, e2_features, f1, f2), dim=-1)

        logits = self.fc2_head(self.fc2(line_features) + self.fc2_res(torch.cat((f1, f2), dim=-1)))

        # 线段分数预测
        if self.loi_cls_type == 'softmax':
            scores = logits.softmax(dim=-1)[:, 1]
        else:
            scores = logits.sigmoid()[:, 0]


        sarg = torch.argsort(scores, descending=True)
        lines_final = lines_pred[sarg]
        score_final = scores[sarg]

        num_detection = min((score_final > 0.0).sum(), 1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        # 处理尺度
        sx = annotations[0]['width'] / output.size(3)
        sy = annotations[0]['height'] / output.size(2)
        line_scale_vec = torch.tensor([sx, sy, sx, sy], dtype=torch.float32, device=device).reshape(-1, 4)
        lines_final *= line_scale_vec

        extra_info['time_verification'] = time.time() - extra_info['time_verification']

        # 构造输出
        output = {
            'lines_pred': lines_final,
            'lines_score': score_final,
            's': s_pred,
            'filename': annotations[0]['filename'],
            'width': annotations[0]['width'],
            'height': annotations[0]['height'],
        }

        return output, extra_info



    def focal_loss(self,input, target, gamma=2.0):
        prob = F.softmax(input, 1) 
        ce_loss = F.cross_entropy(input, target,  reduction='none')
        p_t = prob[:,1] * target + prob[:,0] * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss





    def calculate_normalized_s(self, metas, predicted_s):
        batch_size = predicted_s.size(0)
        normalized_s = torch.zeros((batch_size, 1, 128, 128), device=predicted_s.device)

        for i, meta in enumerate(metas):
            s_values = meta['s']
            lines = meta['lines']
            img_width, img_height = meta['width'], meta['height']

            sample_s_map = torch.zeros((1, 128, 128), device=predicted_s.device)

            for j, line in enumerate(lines):
                line_length = torch.sqrt(torch.sum((line[:2] - line[2:]) ** 2))
                normalized_s_value = s_values[j] / line_length

                x1, y1, x2, y2 = line
                num_points = int(line_length.item())
                x_coords = torch.linspace(x1, x2, num_points)
                y_coords = torch.linspace(y1, y2, num_points)

                for x, y in zip(x_coords, y_coords):
                    normalized_x = (x / img_width) * 127
                    normalized_y = (y / img_height) * 127
                    if 0 <= normalized_x < 128 and 0 <= normalized_y < 128:
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = int(normalized_x + dx), int(normalized_y + dy)
                                if 0 <= nx < 128 and 0 <= ny < 128:
                                    sample_s_map[0, ny, nx] = normalized_s_value

            normalized_s[i, 0, :, :] = sample_s_map

        return normalized_s

    def decoding_mask(self, md_maps, dis_maps, residual_maps, scores, scale=5.0):
        device = md_maps.device

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0.reshape(1,1,-1)
        x0 = x0.reshape(1,1,-1)

        sign_pad = torch.arange(-self.use_residual,self.use_residual+1,device=device,dtype=torch.float32).reshape(1,-1,1)

        if residual_maps is not None:
            residual = residual_maps.reshape(batch_size,1,-1)*sign_pad
            distance_fields = dis_maps.reshape(batch_size,1,-1) + residual
            scores = scores.reshape(batch_size,1,-1).repeat((1,2*self.use_residual+1,1))
        else:
            distance_fields = dis_maps.reshape(batch_size,1,-1)
            scores = scores.reshape(batch_size,1,-1)
        md_maps = md_maps.reshape(batch_size,3,-1)

        distance_fields = distance_fields.clamp(min=0,max=1.0)
        md_un = (md_maps[:,:1] - 0.5)*np.pi*2
        st_un = md_maps[:,1:2]*np.pi/2.0
        ed_un = -md_maps[:,2:3]*np.pi/2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md*y_st)*distance_fields*scale
        y_st_rotated = (ss_md + cs_md*y_st)*distance_fields*scale

        x_ed_rotated = (cs_md - ss_md*y_ed)*distance_fields*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*distance_fields*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)


        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final),dim=-1)

        lines = lines.reshape(batch_size,-1,4)
        scores = scores.reshape(batch_size,-1)

        sc_, arg_ = scores[0].sort(descending=True)
        lines_out = lines[0][arg_[sc_>0]]

        return lines_out, sc_[sc_>0]

    def decoding(self, md_maps, dis_maps, residual_maps, scale=5.0, flatten = True):
        device = md_maps.device

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0[None,None]
        x0 = x0[None,None]
        
        sign_pad = torch.arange(-self.use_residual,self.use_residual+1,device=device,dtype=torch.float32).reshape(1,-1,1,1)

        if residual_maps is not None:
            residual = residual_maps*sign_pad
            distance_fields = dis_maps + residual
        else:
            distance_fields = dis_maps
        distance_fields = distance_fields.clamp(min=0,max=1.0)
        md_un = (md_maps[:,:1] - 0.5)*np.pi*2
        st_un = md_maps[:,1:2]*np.pi/2.0
        ed_un = -md_maps[:,2:3]*np.pi/2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md*y_st)*distance_fields*scale
        y_st_rotated = (ss_md + cs_md*y_st)*distance_fields*scale

        x_ed_rotated = (cs_md - ss_md*y_ed)*distance_fields*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*distance_fields*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        
        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final),dim=-1)
        if flatten:
            lines = lines.reshape(batch_size,-1,4)

        return lines

def get_model(pretrained = False):
    from parsing.config import cfg
    import os
    model = WireframeDetector(cfg)
    if pretrained:
        url = PRETRAINED.get('url')
        hubdir = torch.hub.get_dir()
        filename = os.path.basename(url)
        dst = os.path.join(hubdir,filename)
        state_dict = torch.hub.load_state_dict_from_url(url,dst)
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
    return model


