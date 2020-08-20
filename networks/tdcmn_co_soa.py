import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import math

from . import object_detector_network
from . import scene_detector_network
from . import action_detector_network


class Indomain_Dynamic_Attention(nn.Module):

    def __init__(self, classes, branches, T):
        super(Indomain_Dynamic_Attention, self).__init__()
        #### parameters
        self.reduce = 16
        self.len = 32
        self.branches = branches
        self.T = T
        self.classes = classes

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        self.atten_feat_reduce_bn = nn.BatchNorm1d(self.classes*self.branches)
        self.atten_feat_reduce = nn.Conv1d(in_channels=self.classes*self.branches, out_channels=self.classes, kernel_size=1, stride=1, bias=False)

        len = self.T // 2
        
        self.w_reduce = Parameter(torch.Tensor(len, self.T))
        stv = 1. / math.sqrt(self.w_reduce.size(1))
        self.w_reduce.data.uniform_(-stv, stv)

        self.w_atten = Parameter(torch.Tensor(self.branches, len))
        stv = 1. / math.sqrt(self.w_atten.size(1))
        self.w_atten.data.uniform_(-stv, stv)

        ## time attention
        len = max(self.classes // self.reduce, self.len)
        
        self.time_feature = Parameter(torch.Tensor(len, self.classes))
        stv = 1. / math.sqrt(self.time_feature.size(1))
        self.time_feature.data.uniform_(-stv, stv)
        
        self.time_atten = Parameter(torch.Tensor(1, len))
        stv = 1. / math.sqrt(self.time_atten.size(1))
        self.time_atten.data.uniform_(-stv, stv)

        for l in self.children():
            if isinstance(l, nn.Conv1d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)

    def forward(self, feature):
        
        ## convs attention
        feature = torch.cat(feature, dim=1)
        feature = F.relu(self.atten_feat_reduce_bn(feature))
        feature = self.atten_feat_reduce(feature)

        batch_size = feature.size(0)
        atten_feat = feature.view(-1, self.T)
        atten_feat = torch.tanh(F.linear(atten_feat, self.w_reduce, None))
        atten_feat = F.linear(atten_feat, self.w_atten, None).view(batch_size, -1, self.branches).transpose(1,2).contiguous()
        atten_feat = F.softmax(atten_feat, dim=1).unsqueeze(3)

        ## time attention
        time_atten_feat = torch.tanh(F.linear(feature.permute(0,2,1).contiguous().view(-1, self.classes), self.time_feature, None))
        time_atten_feat = F.linear(time_atten_feat, self.time_atten, None).view(batch_size, -1, 1)
        time_atten_feat = F.softmax(time_atten_feat, dim=1)
      
        return [atten_feat, time_atten_feat]


class Crossdomain_Dynamic_Attention(nn.Module):

    def __init__(self, classes, branches, domains, T):
        super(Crossdomain_Dynamic_Attention, self).__init__()
        #### parameters
        self.reduce = 16
        self.len = 32
        self.classes = classes
        self.domains = domains
        self.branches = branches
        self.T = T

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        ## domain one
        self.domain_one_feature = Parameter(torch.Tensor(classes[1], classes[0]))
        stv = 1. / math.sqrt(self.domain_one_feature.size(1))
        self.domain_one_feature.data.uniform_(-stv, stv)
       
        len = self.T // 2

        self.domain_one_reduce = Parameter(torch.Tensor(len, self.T))
        stv = 1. / math.sqrt(self.domain_one_reduce.size(1))
        self.domain_one_reduce.data.uniform_(-stv, stv)
        
        self.domain_one_atten = Parameter(torch.Tensor(self.branches, len))
        stv = 1. / math.sqrt(self.domain_one_atten.size(1))
        self.domain_one_atten.data.uniform_(-stv, stv)
 
        ## domain two
        self.domain_two_feature = Parameter(torch.Tensor(classes[2], classes[0]))
        stv = 1. / math.sqrt(self.domain_two_feature.size(1))
        self.domain_two_feature.data.uniform_(-stv, stv)
       
        len = self.T // 2

        self.domain_two_reduce = Parameter(torch.Tensor(len, self.T))
        stv = 1. / math.sqrt(self.domain_two_reduce.size(1))
        self.domain_two_reduce.data.uniform_(-stv, stv)
        
        self.domain_two_atten = Parameter(torch.Tensor(self.branches, len))
        stv = 1. / math.sqrt(self.domain_two_atten.size(1))
        self.domain_two_atten.data.uniform_(-stv, stv)
       
        ## domain three
        self.domain_three_feature = Parameter(torch.Tensor(classes[3], classes[0]))
        stv = 1. / math.sqrt(self.domain_three_feature.size(1))
        self.domain_three_feature.data.uniform_(-stv, stv)
       
        len = self.T // 2

        self.domain_three_reduce = Parameter(torch.Tensor(len, self.T))
        stv = 1. / math.sqrt(self.domain_three_reduce.size(1))
        self.domain_three_reduce.data.uniform_(-stv, stv)
        
        self.domain_three_atten = Parameter(torch.Tensor(self.branches, len))
        stv = 1. / math.sqrt(self.domain_three_atten.size(1))
        self.domain_three_atten.data.uniform_(-stv, stv)

        ## domain time
        len = max(classes[0] // self.reduce, self.len)
        
        self.domain_time_feature = Parameter(torch.Tensor(len, classes[0]))
        stv = 1. / math.sqrt(self.domain_time_feature.size(1))
        self.domain_time_feature.data.uniform_(-stv, stv)
        
        self.domain_time_atten = Parameter(torch.Tensor(self.domains, len))
        stv = 1. / math.sqrt(self.domain_time_atten.size(1))
        self.domain_time_atten.data.uniform_(-stv, stv)
        
        for l in self.children():
            if isinstance(l, nn.Conv1d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)

    def forward(self, features):
        
        ### dynamic attentions
        batch_size = features.size(0)
        features = features.transpose(1,2).contiguous().view(-1, self.classes[0])
        
        ## domain one
        atten_feat = F.linear(features, self.domain_one_feature, None)
        atten_feat = atten_feat.view(batch_size, self.T, -1).transpose(1,2).contiguous().view(-1, self.T) 
        atten_feat = torch.tanh(F.linear(atten_feat, self.domain_one_reduce, None))
        atten_feat = F.linear(atten_feat, self.domain_one_atten, None).view(batch_size, -1, self.branches).transpose(1,2).contiguous()
        domain_one_atten_feat = F.softmax(atten_feat, dim=1).unsqueeze(3)
 
        ## domain two
        atten_feat = F.linear(features, self.domain_two_feature, None)
        atten_feat = atten_feat.view(batch_size, self.T, -1).transpose(1,2).contiguous().view(-1, self.T) 
        atten_feat = torch.tanh(F.linear(atten_feat, self.domain_two_reduce, None))
        atten_feat = F.linear(atten_feat, self.domain_two_atten, None).view(batch_size, -1, self.branches).transpose(1,2).contiguous()
        domain_two_atten_feat = F.softmax(atten_feat, dim=1).unsqueeze(3)
        
        ## domain three
        atten_feat = F.linear(features, self.domain_three_feature, None)
        atten_feat = atten_feat.view(batch_size, self.T, -1).transpose(1,2).contiguous().view(-1, self.T) 
        atten_feat = torch.tanh(F.linear(atten_feat, self.domain_three_reduce, None))
        atten_feat = F.linear(atten_feat, self.domain_three_atten, None).view(batch_size, -1, self.branches).transpose(1,2).contiguous()
        domain_three_atten_feat = F.softmax(atten_feat, dim=1).unsqueeze(3)

        ## domain time
        atten_feat = torch.tanh(F.linear(features, self.domain_time_feature, None))
        atten_feat = F.linear(atten_feat, self.domain_time_atten, None).view(batch_size, -1, self.domains).transpose(1,2).contiguous()
        domain_time_atten_feat = F.softmax(atten_feat, dim=1).unsqueeze(3)
      
        return [domain_one_atten_feat, domain_two_atten_feat, domain_three_atten_feat, domain_time_atten_feat]


class Event_Model(nn.Module):

    def __init__(self, opt):
        super(Event_Model, self).__init__()

        #### parameters
        self.opt = opt
        self.concept_number = opt.scene_classes + opt.object_classes + opt.action_classes
        self.scene_classes = opt.scene_classes
        self.object_classes = opt.object_classes
        self.action_classes = opt.action_classes
        self.T = opt.segment_number
        self.branches = 3
        self.domains = 3
       
        #### concept detectors
        self.scene_detector = scene_detector_network.Scene_Detector(opt)
        self.object_detector = object_detector_network.Object_Detector(opt)
        self.action_detector = action_detector_network.Action_Detector(opt)
        
        #### dynamic convolution
        ## scene
        self.scene_domain_bn1 = nn.BatchNorm1d(self.T)#scene_classes)
        self.scene_domain_convs = nn.ModuleList([])
        params = [[1, 0, 1], [3, 1, 1], [3, 2, 2]] 
        for i in range(self.branches):
            self.scene_domain_convs.append(nn.Conv1d(in_channels=self.scene_classes, out_channels=self.scene_classes, kernel_size=params[i][0], stride=1, padding=params[i][1], dilation=params[i][2]))

        ## object
        self.object_domain_bn1 = nn.BatchNorm1d(self.T)#object_classes)
        self.object_domain_convs = nn.ModuleList([])
        params = [[1, 0, 1], [3, 1, 1], [3, 2, 2]] 
        for i in range(self.branches):
            self.object_domain_convs.append(nn.Conv1d(in_channels=self.object_classes, out_channels=self.object_classes, kernel_size=params[i][0], stride=1, padding=params[i][1], dilation=params[i][2]))

        ## action
        self.action_domain_bn1 = nn.BatchNorm1d(self.T)#object_classes)
        self.action_domain_convs = nn.ModuleList([])
        params = [[1, 0, 1], [3, 1, 1], [3, 2, 2]] 
        for i in range(self.branches):
            self.action_domain_convs.append(nn.Conv1d(in_channels=self.action_classes, out_channels=self.action_classes, kernel_size=params[i][0], stride=1, padding=params[i][1], dilation=params[i][2]))

        ##  dynamic attentions
        # indomain
        self.scene_indomain_dynamic_attention = Indomain_Dynamic_Attention(classes=self.scene_classes, branches=self.branches, T=self.T)
        self.object_indomain_dynamic_attention = Indomain_Dynamic_Attention(classes=self.object_classes, branches=self.branches, T=self.T)
        self.action_indomain_dynamic_attention = Indomain_Dynamic_Attention(classes=self.action_classes, branches=self.branches, T=self.T)
        
        # cross domain
        self.crossdomain_dynamic_attention = Crossdomain_Dynamic_Attention(classes=[self.concept_number, self.scene_classes, self.object_classes, self.action_classes], branches=self.branches, domains = self.domains, T=self.T)

        ## other
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        #### feature reduce dimentation & fusion
        self.reduce_dim = 1024
        self.concat_bn1 = nn.BatchNorm1d(self.concept_number*2)
        self.concat_reduce_dim = nn.Linear(self.concept_number*2, self.reduce_dim)
        
        #### classification
        self.final_bn1 = nn.BatchNorm1d(self.reduce_dim)
        self.final_classifier = nn.Linear(self.reduce_dim, opt.event_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        for l in self.children():
            if isinstance(l, nn.Conv1d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(l.bias, 0)

    def forward(self, sceobj_frame):
        
        #### extract initial concept representation
        action_frame = sceobj_frame.permute(0, 1, 3, 2, 4, 5)
        
        ## scene and object frame input size N T D C H W
        N, T, D, C, H, W = sceobj_frame.size()
        _, _, _, _, aH, aW = action_frame.size()
        
        # N T D C H W -> NTD C H W
        sceobj_frame = sceobj_frame.view(-1, C, H, W)
        # NTD C H W -> NTD F
        scene_feature = self.scene_detector(sceobj_frame)
        object_feature = self.object_detector(sceobj_frame)
        
        # NTD F -> N T D F
        scene_feature = scene_feature.view(N, T, D, -1)
        object_feature = object_feature.view(N, T, D, -1)
        # N T D F -> N T F
        scene_feature, _ = torch.max(scene_feature, dim=2)
        object_feature, _ = torch.max(object_feature, dim=2)
       
        ## action frame inpupt size N T C D aH aW
        # N T C D aH aW ->  NT C D aH aW
        action_frame = action_frame.view(-1, C, D, aH, aW)
        # NT C D H W ->  NT F
        action_feature = self.action_detector(action_frame)
        del action_frame
        # NT F -> N T F
        action_feature = action_feature.view(N, T, -1)
        
        #### multiple branch convolutions
        ## bn & relu
        scene_feature = F.relu(self.scene_domain_bn1(scene_feature))
        object_feature = F.relu(self.object_domain_bn1(object_feature))
        action_feature = F.relu(self.action_domain_bn1(action_feature))
        
        ## permute
        # N T F -> N F T
        scene_feature = scene_feature.permute(0, 2, 1)
        object_feature = object_feature.permute(0, 2, 1)
        action_feature = action_feature.permute(0, 2, 1)
       
        #### dynamic convolution
        ### multiple branches convolution
        ## scene
        scene_feature = [conv(scene_feature) for conv in self.scene_domain_convs]
        scene_feature_b = torch.stack(scene_feature, dim=1)

        ## object
        object_feature = [conv(object_feature) for conv in self.object_domain_convs]
        object_feature_b = torch.stack(object_feature, dim=1)

        ## action
        action_feature = [conv(action_feature) for conv in self.action_domain_convs]
        action_feature_b = torch.stack(action_feature, dim=1)

        ### in-domain dynamic attention
        ## scene dynamic feature 
        attentions = self.scene_indomain_dynamic_attention(scene_feature)
        indomain_scene_feature = torch.sum(scene_feature_b*attentions[0], dim=1)
        indomain_scene_feature = torch.matmul(indomain_scene_feature, attentions[1])

        ## object dynamic feature 
        attentions = self.object_indomain_dynamic_attention(object_feature)
        indomain_object_feature = torch.sum(object_feature_b*attentions[0], dim=1)
        indomain_object_feature = torch.matmul(indomain_object_feature, attentions[1])

        ## action dynamic feature 
        attentions = self.action_indomain_dynamic_attention(action_feature)
        indomain_action_feature = torch.sum(action_feature_b*attentions[0], dim=1)
        indomain_action_feature = torch.matmul(indomain_action_feature, attentions[1])
        
        ### cross-domain dynamic attention
        scene_feature = torch.sum(scene_feature_b, dim=1)
        object_feature = torch.sum(object_feature_b, dim=1)
        action_feature = torch.sum(action_feature_b, dim=1)
        attentions = torch.cat((scene_feature, object_feature), 1)
        attentions = torch.cat((attentions, action_feature), 1)
        attentions = self.crossdomain_dynamic_attention(attentions)
      
        ## scene dynamic feature
        crossdomain_scene_feature = torch.sum(scene_feature_b*attentions[0], dim=1)
        crossdomain_scene_feature = torch.matmul(crossdomain_scene_feature, attentions[3][:, 0, :, :])

        ## object dynamic feature 
        crossdomain_object_feature = torch.sum(object_feature_b*attentions[1], dim=1)
        crossdomain_object_feature = torch.matmul(crossdomain_object_feature, attentions[3][:, 1, :, :])

        ## action dynamic feature 
        crossdomain_action_feature = torch.sum(action_feature_b*attentions[2], dim=1)
        crossdomain_action_feature = torch.matmul(crossdomain_action_feature, attentions[3][:, 1, :, :])

        #### concat
        indomain_feature = torch.cat((indomain_scene_feature, indomain_object_feature), 1)
        indomain_feature = torch.cat((indomain_feature, indomain_action_feature), 1).squeeze(2)
        crossdomain_feature = torch.cat((crossdomain_scene_feature, crossdomain_object_feature), 1)
        crossdomain_feature = torch.cat((crossdomain_feature, crossdomain_action_feature), 1).squeeze(2)
        classification = torch.cat((indomain_feature, crossdomain_feature), 1)

        ## reduce dim
        classification = self.concat_bn1(classification)
        classification = self.relu(classification)
        classification = self.dropout(classification)
        classification = self.concat_reduce_dim(classification)
        
        ## classification
        classification = self.final_bn1(classification)
        classification = self.relu(classification)
        classification = self.dropout(classification)
        classification = self.final_classifier(classification)

        return classification
