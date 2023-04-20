# 根据opt中的输入不同选择初始化的模型
def create_model(opt):
    model = None
    print(opt.model)
   
    if opt.model == 'single':
        # assert(opt.dataset_mode == 'unaligned')
        from .single_model import SingleModel
        model = SingleModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
