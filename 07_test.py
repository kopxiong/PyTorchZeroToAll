
import torch
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

#print(x_data.data.shape)
#print(y_data.data.shape)

print(x_data.data.size())
print(y_data.data.size())


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate three nn.Linear modules
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 32)    #(8, 6)
        self.l2 = torch.nn.Linear(32, 16)    #(6, 4)
        self.l3 = torch.nn.Linear(16, 1)    #(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.99))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)    # lr=0.1

# Training loop
for epoch in range(5000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("y_pred & y_data: ", y_pred)
