import torch
import torch.nn as nn


class ResidualBlock2dConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernelsize,
        stride,
        padding,
        dilation,
        downsample,
        a=1,
        b=1,
    ):
        super(ResidualBlock2dConv, self).__init__()
        self.conv1 = nn.Conv2d(
            channels_in,
            channels_in,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=False,
        )
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.a * residual + self.b * out
        return out


def make_res_block_feature_extractor(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    dilation,
    a_val=2.0,
    b_val=0.3,
):
    downsample = None
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock2dConv(
            in_channels,
            out_channels,
            kernelsize,
            stride,
            padding,
            dilation,
            downsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(
        self,
        a,
        b,
        image_channels=3,
        DIM_img=128,
        kernelsize_enc_img=3,
        enc_stride_img=2,
        enc_padding_img=1,
    ):
        super(FeatureExtractorImg, self).__init__()
        self.a = a
        self.b = b
        self.conv1 = nn.Conv2d(
            image_channels,
            DIM_img,
            kernel_size=kernelsize_enc_img,
            stride=enc_stride_img,
            padding=enc_padding_img,
            dilation=1,
            bias=False,
        )
        self.resblock1 = make_res_block_feature_extractor(
            DIM_img,
            2 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock2 = make_res_block_feature_extractor(
            2 * DIM_img,
            3 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=self.a,
            b_val=self.b,
        )
        self.resblock3 = make_res_block_feature_extractor(
            3 * DIM_img,
            4 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=self.a,
            b_val=self.b,
        )
        self.resblock4 = make_res_block_feature_extractor(
            4 * DIM_img,
            5 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=0,
            dilation=1,
            a_val=self.a,
            b_val=self.b,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        return out


class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearFeatureCompressor, self).__init__()
        self.mu = nn.Linear(in_channels, out_channels, bias=True)
        self.logvar = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1)
        mu, logvar = self.mu(feats), self.logvar(feats)
        return mu, logvar


class ResidualBlock2dTransposeConv(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernelsize,
        stride,
        padding,
        dilation,
        o_padding,
        upsample,
        a=1,
        b=1,
    ):
        super(ResidualBlock2dTransposeConv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            channels_in,
            channels_in,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=False,
        )
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            output_padding=o_padding,
        )
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        # self.conv3 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        # self.bn3 = nn.BatchNorm2d(channels_out)
        self.upsample = upsample
        self.a = a
        self.b = b

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out
        return out


def make_res_block_data_generator(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    o_padding,
    dilation,
    a_val=1.0,
    b_val=1.0,
):
    upsample = None
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=o_padding,
            ),
            nn.BatchNorm2d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock2dTransposeConv(
            in_channels,
            out_channels,
            kernelsize=kernelsize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            o_padding=o_padding,
            upsample=upsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class DataGeneratorImg(nn.Module):
    def __init__(
        self,
        a,
        b,
        DIM_img=128,
        image_channels=3,
        kernelsize_dec_img=3,
        dec_stride_img=2,
        dec_padding_img=1,
        dec_outputpadding_img=1,
    ):
        super(DataGeneratorImg, self).__init__()
        self.a = a
        self.b = b
        # self.data_generator = make_res_layers_data_generator(self.args, a=self.a, b=self.b)
        # self.resblock1 = make_res_block_data_generator(5*self.args.DIM_img, 5*self.args.DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b);
        self.resblock1 = make_res_block_data_generator(
            5 * DIM_img,
            4 * DIM_img,
            kernelsize=4,
            stride=1,
            padding=0,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock2 = make_res_block_data_generator(
            4 * DIM_img,
            3 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock3 = make_res_block_data_generator(
            3 * DIM_img,
            2 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock4 = make_res_block_data_generator(
            2 * DIM_img,
            1 * DIM_img,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.conv = nn.ConvTranspose2d(
            DIM_img,
            image_channels,
            kernel_size=kernelsize_dec_img,
            stride=dec_stride_img,
            padding=dec_padding_img,
            dilation=1,
            output_padding=dec_outputpadding_img,
        )

    def forward(self, feats):
        # d = self.data_generator(feats)
        d = self.resblock1(feats)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        # d = self.resblock5(d);
        d = self.conv(d)
        return d


class CelebaImgEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        a_img=2.0,
        b_img=0.3,
        num_layers_img=5,
        DIM_img=128,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractorImg(a=a_img, b=b_img)
        self.feature_compressor = LinearFeatureCompressor(
            num_layers_img * DIM_img, latent_dim
        )

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)
        h_img = h_img.view(h_img.shape[0], h_img.shape[1], h_img.shape[2])
        mu, logvar = self.feature_compressor(h_img)

        return torch.cat([mu, logvar], dim=-1)


class CelebaImgDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        num_layers_img=5,
        DIM_img=128,
        a_img=2.0,
        b_img=0.3,
    ):
        super().__init__()
        self.feature_generator = nn.Linear(
            latent_dim,
            num_layers_img * DIM_img,
            bias=True,
        )
        self.img_generator = DataGeneratorImg(a=a_img, b=b_img)

    def forward(self, z):
        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(
            img_feat_hat.size(0), img_feat_hat.size(1), 1, 1
        )
        img_hat = self.img_generator(img_feat_hat)
        return img_hat


class CelebaImgClassifier(nn.Module):
    def __init__(
        self,
        a_img=2.0,
        b_img=0.3,
        num_layers_img=5,
        DIM_img=128,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractorImg(a=a_img, b=b_img)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(
            in_features=num_layers_img * DIM_img, out_features=40, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_img):
        h = self.feature_extractor(x_img)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)

        # Multiclass classification / probabilities
        # [B, 40]
        return out

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img)
        return h
