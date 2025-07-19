import torch.nn as nn
import torch


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels))

    def forward(self, x, fmap_dict = None, prefix = ""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply SE attention
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add
        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out
            
        return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes=50, use_se=True, use_fpn=True):
        super().__init__()
        self.use_fpn = use_fpn
        
        # Initial convolution with better design
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual layers with SE blocks
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64, use_se=use_se) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1, use_se=use_se) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1, use_se=use_se) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1, use_se=use_se) for i in range(3)])

        # Feature Pyramid Network for multi-scale feature fusion
        if use_fpn:
            self.fpn_layers = nn.ModuleDict({
                'layer1': nn.Conv2d(64, 256, 1),
                'layer2': nn.Conv2d(128, 256, 1),
                'layer3': nn.Conv2d(256, 256, 1),
                'layer4': nn.Conv2d(512, 256, 1)
            })
            self.fpn_upsample = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Upsample(scale_factor=4, mode='nearest'),
                nn.Upsample(scale_factor=8, mode='nearest')
            ])

        # Improved pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Enhanced dropout and regularization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Multi-scale feature fusion
        if use_fpn:
            self.classifier = nn.Sequential(
                nn.Linear(512 + 256 * 3, 1024),  # 512 from layer4 + 256*3 from FPN
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, x, return_feature_maps = False):
        if not return_feature_maps:    
            # Initial convolution
            x = self.conv1(x)
            
            # Residual layers
            for block in self.layer1:
                x = block(x)
            layer1_out = x
            
            for block in self.layer2:
                x = block(x)
            layer2_out = x
            
            for block in self.layer3:
                x = block(x)
            layer3_out = x
            
            for block in self.layer4:
                x = block(x)
            layer4_out = x
            
            if self.use_fpn:
                # Feature Pyramid Network
                fpn_features = []
                
                # Process each layer through FPN
                p4 = self.fpn_layers['layer4'](layer4_out)
                p3 = self.fpn_layers['layer3'](layer3_out) + self.fpn_upsample[0](p4)
                p2 = self.fpn_layers['layer2'](layer2_out) + self.fpn_upsample[1](p4)
                p1 = self.fpn_layers['layer1'](layer1_out) + self.fpn_upsample[2](p4)
                
                # Global average pooling on all FPN features
                fpn_features = [
                    self.avgpool(p1).flatten(1),
                    self.avgpool(p2).flatten(1),
                    self.avgpool(p3).flatten(1),
                    self.avgpool(p4).flatten(1)
                ]
                
                # Concatenate all features
                x = torch.cat(fpn_features, dim=1)
            else:
                # Standard pooling
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            
            # Classification head
            x = self.classifier(x)
            return x
        else:
            feature_maps={}
            x = self.conv1(x)
            feature_maps["conv1"] = x
            for i,block in enumerate(self.layer1):
                x=block(x,feature_maps,prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x  
            for i,block in enumerate(self.layer2):
                x=block(x,feature_maps,prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x
            for i,block in enumerate(self.layer3):
                x=block(x,feature_maps,prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x
            for i,block in enumerate(self.layer4):
                x=block(x,feature_maps,prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout1(x)
            x = self.fc(x)
            
            return x,feature_maps
