import torch
import torch.nn as nn

PATH = ""

## save complete model
torch.save(model,PATH)

# model class must be defined somewhere
model = torch.load(PATH)
model.eval()


# recomended way: #
### save state dict ###
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()