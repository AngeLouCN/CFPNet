from model.CFPNet import CFPNet





def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return CFPNet(classes=num_classes)
