import torch
import torch.nn as nn


architecture_config=[
    (7,64,2,3),
    "maxpool",
    (3,192,1,1),
    "maxpool",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "maxpool",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "maxpool",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),

]


class CNNblock(nn.Module):
    def __init__(self,input_channels,output_channels,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(input_channels,output_channels,bias=False,**kwargs)
        self.batchnorm=nn.BatchNorm2d(output_channels)
        self.lrelu=nn.LeakyReLU(0.1)

    def forward(self,x):
        x=self.conv(x);
        x=self.batchnorm(x);
        x=self.lrelu(x)
        return x



class YOLO(nn.Module):
    def __init__(self,input_channels=3,**kwargs):
        super().__init__()
        self.architecture=architecture_config
        self.in_channels=input_channels
        self.darknet=self.create_convlayers(self.architecture)
        self.fully_connected=self.create_fc(**kwargs)

    def forward(self,x):
        x=self.darknet(x)
        return self.fully_connected(torch.flatten(x,start_dim=1))

    def create_convlayers(self,architecture):
        layers=[]
        inchannels=self.in_channels

        for x in architecture:
            if type(x)==tuple:
                layers+=[CNNblock(inchannels,x[1],kernel_size=x[0],stride=x[2],padding=x[3])]
                inchannels=x[1]

            elif type(x)==str:
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]

            elif type(x)==list:
                conv1=x[0]
                conv2=x[1]
                times=x[2]

                for i in range(times):
                    layers+=[
                        CNNblock(inchannels,conv1[1],kernel_size=conv1[0],stride=conv1[2],padding=conv1[3])
                    ]
                    layers+=[
                        CNNblock(conv1[1],conv2[1],kernel_size=conv2[0],stride=conv2[2],padding=conv2[3])
                    ]
                    inchannels=conv2[1]

        return nn.Sequential(*layers)

    def create_fc(self,split_size,num_boxes,num_classes):
        S,B,C=split_size,num_boxes,num_classes
        return nn.Sequential(
            nn.Flatten(),nn.Linear(1024*S*S,496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496,S*S*(C+B*5)),
        )

def test(S=7,B=2,C=20):
    model=YOLO(split_size=S,num_boxes=B,num_classes=C)
    x=torch.randn((2,3,448,448))
    print(model(x).shape)




