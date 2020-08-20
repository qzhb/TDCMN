from networks import tdcmn_si_so
from networks import tdcmn_si_soa
from networks import tdcmn_co_so
from networks import tdcmn_co_soa


def generate_model(opt):
    
    if opt.model_name == 'tdcmn_co_soa':
        model = tdcmn_co_soa.Event_Model(opt)

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []
        
        dynamic_weight = 'domain'
        temp_dyn = []
        
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('a', k)
                temp_fc.append(v)
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif dynamic_weight in k:
                print('c', k)
                temp_dyn.append(v)
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_scratch + temp_dyn
        parameters.append({'params': temp})

    elif opt.model_name == 'tdcmn_si_soa':
        model = tdcmn_si_soa.Event_Model(opt)

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        action_detectors_ft_module_names = 'action_detector.logits'
        temp_fc = []

        dynamic_weight = 'domain'
        temp_dyn = []
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('a', k)
                temp_fc.append(v)
            elif action_detectors_ft_module_names in k:
                print('b', k)
                temp_fc.append(v)
            elif dynamic_weight in k:
                print('c', k)
                temp_dyn.append(v)
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_scratch + temp_dyn
        parameters.append({'params': temp})

    elif opt.model_name == 'tdcmn_co_so':
        model = tdcmn_co_so.Event_Model(opt)

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        temp_fc = []
        dynamic_weight = 'domain'
        temp_dyn = []
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('a', k)
                temp_fc.append(v)
            elif dynamic_weight in k:
                print('c', k)
                temp_dyn.append(v)
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_scratch + temp_dyn
        parameters.append({'params': temp})

    elif opt.model_name == 'tdcmn_si_so':
        model = tdcmn_si_so.Event_Model(opt)

        detectors_ft_module_names = ['scene_detector.fc', 'object_detector.fc']
        temp_fc = []
        dynamic_weight = 'domain'
        temp_dyn = []
        scratch_train_module_names = ['concat_bn1', 'concat_reduce_dim', 'final_bn1', 'final_classifier']
        temp_scratch = []

        parameters = []
        for k, v in model.named_parameters():
            if k[:-5] in detectors_ft_module_names or k[:-7] in detectors_ft_module_names:
                print('a', k)
                temp_fc.append(v)
            elif dynamic_weight in k:
                print('c', k)
                temp_dyn.append(v)
            elif k[:-5] in scratch_train_module_names or k[:-7] in scratch_train_module_names:
                print('d', k)
                temp_scratch.append(v)
            else:
                v.requires_grad = False
        temp = temp_fc + temp_scratch + temp_dyn
        parameters.append({'params': temp})

    return model, parameters
