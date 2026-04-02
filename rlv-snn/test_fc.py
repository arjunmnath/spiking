import torch
from main import FCSNN, Config
fcsnn = FCSNN()
X_tr = torch.rand(10, Config.N_FEATURES)
Y_tr = torch.randint(0, 10, (10,))
fcsnn.train_step(X_tr, Y_tr)
print("Train step ok")
acc = fcsnn.accuracy(X_tr, Y_tr)
print("Acc:", acc)
