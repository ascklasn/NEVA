import torch
from torch import nn
from itertools import repeat
import collections.abc
from functools import partial
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Pooler(nn.Module):  #First, LN layer normalization, then linear mapping, and reactivate function
    def __init__(self, input_features, output_features, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        # self.activation = nn.Tanh()

    def forward(self, x):
        # x of shape [batch_size, feat_dim]
        cls_rep = self.norm(x)
        pooled_output = self.dense(cls_rep)
        # pooled_output = self.activation(pooled_output)
        return pooled_output

class GluMlp(nn.Module):   
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class report_only(nn.Module): 
    def __init__(self, report_input_dim=768, hidden_size=128, nb_classes=3,dropout=False):
        super(report_only, self).__init__()
        self.model = nn.Sequential(
            # nn.LayerNorm(report_input_dim),
            # nn.Linear(report_input_dim, hidden_size),
            # # nn.Tanh(),  
            # nn.Linear(hidden_size, nb_classes),
            nn.Linear(report_input_dim, nb_classes),
        )
        self.pooler = Pooler(input_features=report_input_dim, output_features=512,norm_layer=nn.LayerNorm)  # 1024--512  LayerNorm + Linear
        self.classifier = nn.Sequential(nn.Sigmoid(),
                                        nn.Linear(in_features=512, out_features=128),nn.Linear(in_features=128,out_features=nb_classes))   # hidden_size 128 --> nb_classes
        # self.gluMlp = GluMlp(in_features=report_input_dim,hidden_features=128,out_features=nb_classes)

    def forward(self, x):
        logits = self.classifier((self.pooler(x)))
        # logits = self.gluMlp(x)

        return logits  # [N,1024] --> [N,nb_classes]
