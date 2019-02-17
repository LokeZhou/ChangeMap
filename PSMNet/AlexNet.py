import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        '''self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            
        )'''
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.Sequential(nn.ReLU(inplace=True),
                                  )
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)

        self.max = nn.MaxPool2d(kernel_size=3,stride=1)


        self.fc = nn.Sequential(nn.Linear(3, 3))

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        '''x = self.conv(x)
        print x.size()
        #print x

        x = self.fc(x)
        print x.size()
        print x

        x = self.sigmoid(x)
        print x.size()
        print x

        y = F.interpolate(x, [3,9,9], mode='bilinear')
        print y.size()
        print y
        '''

        x = self.avg(x)
        print (x.size())
        print (x)

def change(x):
    x =  x + x
    print(x)


if __name__ == '__main__':
    net = AlexNet()

    data_input = Variable(torch.randn([1, 1, 4, 4]))
    print(data_input)
    net(data_input)


