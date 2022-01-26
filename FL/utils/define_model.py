import sys
sys.path.append('../')
import models

def define_model(model_name):
    
    model = models.__dict__[model_name]
    
    return model
    

