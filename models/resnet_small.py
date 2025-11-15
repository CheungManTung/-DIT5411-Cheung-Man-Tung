from tensorfluow.keras import layers, models, regularizers

def conv_bn_relu(x, f, k=3, s=1, wd=1e-4):
    x = layers.Conv2D(f, k, s, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, f, s=1, wd=1e-4):
    shortcut = x
    x = conv_bn_relu(x, f, 3, s, wd)
    x = layers.Conv2D(f, 3, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != f or s != 1:
        shortcut = layers.Conv2D(f, 1, s, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(wd))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet_small(num_classes, img_size=64):
    inp = layers.Input((img_size, img_size, 1))
    x = conv_bn_relu(inp, 64, 3, 1)
    x = layers.MaxPool2D()(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, s=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, s=2)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="resnet_small")