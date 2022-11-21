import yaml
from transformers import Wav2Vec2Model, Wav2Vec2Config, UniSpeechSatConfig, UniSpeechSatModel, WavLMConfig, WavLMModel, HubertModel, HubertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.utils as utils
from libs.nnet import *
# Copyright xmuspeech (Author: Leo 2022-05-27)
# refs:
# 1.  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
#           https://arxiv.org/abs/2005.07143


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1, bn_params={}
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        assert scale>1
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        context = [i for i in range(-(kernel_size//2) *
                                    dilation, (kernel_size//2)*dilation+1, dilation)]
        self.blocks = nn.ModuleList(
            [
                ReluBatchNormTdnnLayer(in_channel, hidden_channel, context,**bn_params) for i in range(scale - 1)
            ]
        )
 
        self.scale = scale

    def forward(self, x):
        y = []
        spx = torch.chunk(x,self.scale,dim=1)
        sp = spx[0]
        y.append(sp)
        for i,block in enumerate(self.blocks):
            if i == 0:
                sp = spx[i+1]
            if i>=1:
                sp = sp + spx[i+1]
            sp = block(sp)
            y.append(sp)

        y = torch.cat(y, dim=1)
        return y


''' The SE connection of 1D case.
'''
# class SE_Connect(nn.Module):
#     def __init__(self, channels, s=4):
#         super().__init__()
#         assert channels % s == 0, "{} % {} != 0".format(channels, s)
#         self.linear1 = nn.Linear(channels, channels // s)
#         self.linear2 = nn.Linear(channels // s, channels)

#     def forward(self, x):
#         out = x.mean(dim=2)
#         out = F.relu(self.linear1(out))
#         out = torch.sigmoid(self.linear2(out))
#         out = x * out.unsqueeze(2)
#         return out

# Another implementation of SE_Connect


class SE_Connect(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SE_Connect, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x
        
class SE_Connect_layersum(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SE_Connect, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
        )

    def forward(self, input):
        layer_weights = self.se(input).softmax(dim=1)
        return input * layer_weights

''' SE-Res2Block.
'''


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, scale=8, bn_params={}):
        super(SE_Res2Block, self).__init__()
        width = int(math.floor(in_channels / scale))

        self.conv_relu_bn1 = ReluBatchNormTdnnLayer(
            in_channels, width*scale, **bn_params)

        self.res2net_block = Res2NetBlock(
            width*scale, width*scale, scale=scale, kernel_size=kernel_size, dilation=dilation,bn_params=bn_params)
        self.conv_relu_bn2 = ReluBatchNormTdnnLayer(
            in_channels, width*scale, **bn_params)
        self.se = SE_Connect(out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        x = self.conv_relu_bn1(x)

        x = self.res2net_block(x)

        x = self.conv_relu_bn2(x)
        x = self.se(x)
        return x+residual


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, time_attention=False, bn={}):
        super().__init__()
        self.time_attention = time_attention
        accept_dim = in_dim
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if time_attention:
            accept_dim = in_dim*3
        self.attention = nn.Sequential(
            nn.Conv1d(accept_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim, **bn),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):

        if self.time_attention:
            global_mean = torch.mean(x, dim=2, keepdim=True).expand_as(x)
            global_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-9).expand_as(x)
            x_in = torch.cat((x, global_mean, global_std), dim=1)

        else:
            x_in = x

        alpha = self.attention(x_in)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''


class ECAPA_TDNN(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, aug_dropout=0., tail_dropout=0., training=True, use_w2v2=False,
             extracted_embedding="near", mixup=False, mixup_alpha=1.0, pooling="ecpa-attentive", pooling_params={},
             ecapa_params={}, fc1=False, fc1_params={}, fc2_params={},
             margin_loss=True, margin_loss_params={}, use_step=False, step_params={}, transfer_from="softmax_loss", lncl_Loss=False):

        default_ecapa_params = {
            "channels": 1024,
            "embd_dim": 192,
            "mfa_conv": 1536,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        default_pooling_params = {
            "hidden_size": 128,
            "time_attention": True,
            "stddev": True,
        }

        default_fc_params = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }
        if lncl_Loss:
            default_margin_loss_params = {
                "method": "am", "m": 0.2,
                "feature_normalize": True, "s": 30,
                "double": False,
                "mhe_loss": False, "mhe_w": 0.01,
                "inter_loss": 0.,
                "ring_loss": 0.,
                "curricular": False,
                "noise":False,
                "total_iter":1000000,
                "cur_iter":0,
                "reg_loss":False,
                "sqrt":False,
                "square":False,
                "sub_group":False,
                "only_sub":False,
                "reweight":False,
                "use_lncl_th":False,
                "alpha_T":0.,
                }
        else:
            default_margin_loss_params = {
                "method": "am", "m": 0.2,
                "feature_normalize": True, "s": 30,
                "double": False,
                "mhe_loss": False, "mhe_w": 0.01,
                "inter_loss": 0.,
                "ring_loss": 0.,
                "curricular": False,
                "dymatic_m": False,
                "total_iter": 1000000,
                "cur_iter": 0,
                "m_init": 0.2,
                "m_final": 0.5,
                "sqrt": False,
                "square": False,
                }
        default_step_params = {
            "T": None,
            "m": False, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }
        if use_w2v2:
            # 只用一层tdnn加pooling
            self.output_embeddings_encoder = False
            self.use_w2v2 = True
            # w2v2_path = "/data/huggingface/models/wav2vec2-large-xlsr-53"
            # config = Wav2Vec2Config.from_pretrained(w2v2_path)
            # config.num_hidden_layers = 15
            # self.wav2vec2 = Wav2Vec2Model(config)
            
            
            # w2v2_hubert_path = "/data/huggingface/models/hubert_model"
            # config = HubertConfig.from_pretrained(w2v2_hubert_path)
            # config.num_hidden_layers = 5
            # self.wav2vec2 = HubertModel(config)
            
            # w2v2_unisat_path = "/data/huggingface/models/unispeech_sat_model"
            # config = UniSpeechSatConfig.from_pretrained(w2v2_unisat_path)
            # config.num_hidden_layers = 10
            # self.wav2vec2 = UniSpeechSatModel(config)
            
            w2v2_wavlm_path = "/work/w2v2_sv/wavlm_model"
            config = WavLMConfig.from_pretrained(w2v2_wavlm_path)
            config.num_hidden_layers = 10
            
            # two forms of weighting the hidden_states from PTM
            self.use_SE_layer_sum = False
            self.use_weighted_layer_sum = False
            assert (self.use_SE_layer_sum and self.use_weighted_layer_sum) == False
            if self.use_weighted_layer_sum:
                config.num_hidden_layers = 24
                self.layer_weights = nn.Parameter(torch.ones(25) / 25)
            if self.use_SE_layer_sum:
                config.num_hidden_layers = 24
                # self.SE_layersum = SE_Res2Block(
                # config.num_hidden_layers, config.num_hidden_layers, kernel_size=3, dilation=2, scale=5, bn_params=ecapa_params)
                self.SE_layersum = SE_Connect_layersum(num_layers)
            self.wav2vec2 = WavLMModel(config)
            inputs_dim = config.hidden_size
        else:
            self.use_w2v2 = False
            self.wav2vec2 = None
            inputs_dim = 80
        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding
        
        ecapa_params = utils.assign_params_dict(
            default_ecapa_params, ecapa_params)

        pooling_params = utils.assign_params_dict(
            default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(
            default_step_params, step_params)
        embd_dim = ecapa_params["embd_dim"]
        self.embd_dim = embd_dim
        channels = ecapa_params["channels"]
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None
        if not self.output_embeddings_encoder:
            self.layer1 = ReluBatchNormTdnnLayer(
                inputs_dim, channels, [-2, -1, 0, 1, 2], **ecapa_params)
            
            self.layer2 = SE_Res2Block(
                channels, channels, kernel_size=3, dilation=2, scale=8, bn_params=ecapa_params)
            self.layer3 = SE_Res2Block(
                channels, channels, kernel_size=3, dilation=3, scale=8, bn_params=ecapa_params)
            self.layer4 = SE_Res2Block(
                channels, channels, kernel_size=3, dilation=4, scale=8, bn_params=ecapa_params)
            cat_channels = channels * 3
            mfa_conv = ecapa_params["mfa_conv"]
            self.mfa = ReluBatchNormTdnnLayer(
                cat_channels, mfa_conv, **ecapa_params)

            # Pooling
            stddev = pooling_params.pop("stddev")
            if pooling == "attentive":
                self.stats = AttentiveStatisticsPooling(
                    mfa_conv, hidden_size=pooling_params["hidden_size"], context=pooling_params["context"], stddev=stddev)
                self.bn_stats = nn.BatchNorm1d(
                    mfa_conv * 2, **ecapa_params["bn_params"])
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
            elif pooling == "ecpa-attentive":
                self.stats = AttentiveStatsPool(
                    mfa_conv, pooling_params["hidden_size"], pooling_params["time_attention"])
                self.bn_stats = nn.BatchNorm1d(
                    mfa_conv * 2, **ecapa_params["bn_params"])
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
            elif pooling == "multi-head":
                self.stats = MultiHeadAttentionPooling(
                    mfa_conv, stddev=stddev, **pooling_params)
                self.bn_stats = nn.BatchNorm1d(
                    mfa_conv * 2, **ecapa_params["bn_params"])
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
            elif pooling == "global-multi":
                self.stats = GlobalMultiHeadAttentionPooling(
                    mfa_conv, stddev=stddev, **pooling_params)
                self.bn_stats = nn.BatchNorm1d(
                    mfa_conv * 2 * pooling_params["num_head"], **ecapa_params["bn_params"])
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2 * pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None
            elif pooling == "multi-resolution":
                self.stats = MultiResolutionMultiHeadAttentionPooling(
                    mfa_conv, **pooling_params)
                self.bn_stats = nn.BatchNorm1d(
                    mfa_conv * 2 * pooling_params["num_head"], **ecapa_params["bn_params"])
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2 * pooling_params["num_head"], embd_dim, **fc1_params) if fc1 else None

            else:
                self.stats = StatisticsPooling(mfa_conv, stddev=stddev)
                self.bn_stats = nn.BatchNorm1d(mfa_conv * 2)
                self.fc1 = ReluBatchNormTdnnLayer(
                    mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
        else:
            # cat_channels = channels * 3
            cat_channels = channels
            mfa_conv = ecapa_params["mfa_conv"]
            self.mfa = ReluBatchNormTdnnLayer(cat_channels, mfa_conv, **ecapa_params)
            # Pooling
            stddev = pooling_params.pop("stddev")
            self.stats = AttentiveStatsPool(
                mfa_conv, pooling_params["hidden_size"], pooling_params["time_attention"])
            self.bn_stats = nn.BatchNorm1d(
                mfa_conv * 2, **ecapa_params["bn_params"])
            self.fc1 = ReluBatchNormTdnnLayer(
                mfa_conv * 2, embd_dim, **fc1_params) if fc1 else None
                
        self.tail_dropout = torch.nn.Dropout2d(
            p=tail_dropout) if tail_dropout > 0 else None

        if fc1:
            fc2_in_dim = embd_dim
        else:
            fc2_in_dim = mfa_conv * 2
        self.fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, embd_dim, **fc2_params)
        self.tail_dropout = torch.nn.Dropout2d(
            p=tail_dropout) if tail_dropout > 0 else None
        # print("num_targets---------------",num_targets)
        # Loss
        # Do not need when extracting embedding.
        if training:
            if lncl_Loss:
                self.loss = LNCL_Loss(embd_dim, num_targets, **margin_loss_params)
            elif margin_loss:
                self.loss = MarginSoftmaxLoss(
                    embd_dim, num_targets, **margin_loss_params)
                # self.loss = KFNLoss(input_dim=embd_dim,path="/work/kaldi/egs/xmuspeech/czc/ssl/prototypes/kmeans_10000_3init_300maxiter_0.bin")
            else:
                self.loss = SoftmaxLoss(embd_dim, num_targets)
                # self.loss = AngleLoss(embd_dim,num_targets)
            self.wrapper_loss = MixupLoss(
                self.loss, self.mixup) if mixup else None
            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["layer2", "layer3",
                                   "layer4", "conv", "stats", "fc1", "fc2"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {
                    "loss.affine.weight": "loss.weight"}
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameters
        will not be updated during training.
        """
        # for param in self.wav2vec2.feature_extractor.parameters():
            # param.requires_grad = False
        if self.use_w2v2 and self.wav2vec2 is not None:
            self.wav2vec2.feature_extractor._freeze_parameters()
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        if self.use_w2v2 and self.wav2vec2 is not None:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            
    @torch.jit.unused
    @utils.for_device_free
    def forward(self, x, x_len: torch.Tensor=torch.empty(0)):
        # print(x.shape)
        if self.use_w2v2 and self.wav2vec2 is not None:
            # (128, 1, 16000) -> (128, 16000) -> (128, 50, 1024)
            outputs = self.wav2vec2(input_values=x.squeeze(1), output_hidden_states=True)
            x = outputs[0]
            if self.use_weighted_layer_sum:
                hidden_states = torch.stack(outputs.hidden_states, dim=1)
                # (B,num_hidden_layers,T,dim)
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
                x = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            if self.use_SE_layer_sum:
                hidden_states = torch.stack(outputs.hidden_states, dim=1)
                # (B,num_hidden_layers,T,dim)
                bs, num_hidden_layers, T, dim = hidden_states.shape
                x = self.SE_layersum(hidden_states.view(bs, num_hidden_layers, -1)).reshape(bs, num_hidden_layers, T, dim).sum(1)
            # mean normalization
            # x = (x - x.mean(1, keepdim=True))/ x.std(dim=1, keepdim=True)
            x = x - x.mean(1, keepdim=True)
            x = x.transpose(1,2)
            # print(x.shape)
        if not self.output_embeddings_encoder:
            x = self.layer1(x)
            x1 = self.layer2(x)
            x2 = self.layer3(x+x1)
            x3 = self.layer4(x+x1+x2)
            x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        x = self.bn_stats(self.stats(x))
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        x = self.auto(self.fc1, x)
        x = self.fc2(x)
        x = self.auto(self.tail_dropout, x)
        return x#, hs 

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)
    def get_logits(self, inputs, targets=None, output_logits=True):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)

        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        device = next(self.parameters()).device
        # print(device)
        return self.loss(inputs.to(device), targets, output_logits=output_logits)
    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss().
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)
    # def get_accuracy(self, targets=None):
        # """Should call get_accuracy() after get_loss().
        # @return: return accuracy
        # """
        # return torch.tensor(-1.0)
        # return self.loss.accuracy
    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, x):
        if self.use_w2v2 and self.wav2vec2 is not None:
            # (128, 1, 16000) -> (128, 16000) -> (128, 50, 1024)
            x = self.wav2vec2(input_values=x.squeeze(1))[0]
            # mean normalization
            # x = (x - x.mean(1, keepdim=True))/ x.std(dim=1, keepdim=True)
            x = x - x.mean(1, keepdim=True)
    
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x+x1)
        x3 = self.layer4(x+x1+x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        x = self.bn_stats(self.stats(x))
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.fc1, x)
            xvector = self.fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.fc1, x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector

    def extract_embedding_jit(self, x: torch.Tensor, position: str = 'near') -> torch.Tensor:

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x+x1)
        x3 = self.layer4(x+x1+x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)
        x = self.bn_stats(self.stats(x))
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=2)
        if position == "far" and self.fc1 is not None:
            xvector = self.fc1.affine(x)
        elif position == "near_affine":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2.affine(x)
        elif position == "near":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(
                self.extracted_embedding))
        return xvector

    @torch.jit.export
    def extract_embedding_whole(self, input: torch.Tensor, position: str = 'near', maxChunk: int = 10000, isMatrix: bool = True):
        if isMatrix:
            input = torch.unsqueeze(input, dim=0)
            input = input.transpose(1, 2)
        num_frames = input.shape[2]
        num_split = (num_frames + maxChunk - 1) // maxChunk
        split_size = num_frames // num_split
        offset = 0
        embedding_stats = torch.zeros(1, self.embd_dim, 1)
        for _ in range(0, num_split-1):
            this_embedding = self.extract_embedding_jit(
                input[:, :, offset:offset+split_size], position)
            offset += split_size
            embedding_stats += split_size*this_embedding

        last_embedding = self.extract_embedding_jit(
            input[:, :, offset:], position)

        embedding = (embedding_stats + (num_frames-offset)
                     * last_embedding) / num_frames
        return torch.squeeze(embedding.transpose(1, 2)).cpu()

    @torch.jit.export
    def embedding_dim(self) -> int:
        """ Export interface for c++ call, return embedding dim of the model
        """
        return self.embd_dim

    def get_warmR_T(self, T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def step_iter(self, epoch, cur_step):
        # For iterabledataset
        if self.use_step:
            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"]*(1+self.step_params["gamma"]*cur_step)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], cur_step)

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    timer = utils.Timer()
    # x = torch.zeros(2,80, 1000)
    # x = torch.zeros(2,1, 16000)
    sys.path.insert(0, 'subtools/pytorch')
    import libs.egs.egs_online as egs
    model = ECAPA_TDNN(inputs_dim=80, num_targets=1211,training=True,use_w2v2=True)
    # print(model)
    # egs_dir = "exp/egs/voxceleb2_dev_vad"
    egs_dir = "exp/egs/voxceleb2_dev_whole"
    # egs_conf="subtools/conf/egs_pre_sre_noaug.yaml"
    egs_conf="subtools/conf/egs_pre_sre_ecapa.yaml"
    # csv_aug_folder = "exp/aug_csv"
    # speech_aug为true时，若csv_aug_folder为空，默认加高斯白噪声
    csv_aug_folder = ""
    with open(egs_conf,'r') as fin:
        egs_params = yaml.load(fin, Loader=yaml.FullLoader)
        egs_params['dataset_conf']['csv_aug_folder']=csv_aug_folder
    egs_params['dataset_conf']['batch_conf']['batch_size'] = 10
    # egs_params['dataset_conf']['batch_conf']['batch_type'] = 'static_filtered'
    egs_params['data_loader_conf']['num_workers'] = 0
    # egs_params['data_loader_conf']['prefetch_factor'] = 1
    egs_params['dataset_conf']['shuffle'] = False
    egs_params['dataset_conf']['use_w2v2'] = True
    
    egs_params['dataset_conf']['dynamic_random_chunk'] = True
    batch_size = egs_params['dataset_conf']['batch_conf']['batch_size']
    egs_params['dataset_conf']['dynamic_conf'] = {"batch_size":batch_size, "total_iter":100, "cur_iter":50, "chunk_len_init":2.0, "chunk_len_final":6.0, "sqrt":False, "square":False}
    
    
    
    bunch, info = egs.BaseBunch.get_bunch_from_egsdir(egs_dir, egs_params)
    print(bunch)
    print(info)

    # inputs = next(iter(bunch.train_loader))
    for i in range(50):
        inputs = next(iter(bunch.train_loader))
        # inputs = next(iter(bunch.valid_loader))
        # print(inputs[0].shape, inputs[1].shape)
        # print(inputs[1])
        x = inputs[0]
        # print(x.shape)
    

    # total = sum(p.numel() for p in model.parameters())
    # print(model)
    # print(total)

    out = model(x)
    # print(model.loss.K)
    # loss = model.get_loss(out, targets=inputs[1])
    # acc = model.get_accuracy()
    print(loss)
    # print(acc)
    # print(out.shape) 
