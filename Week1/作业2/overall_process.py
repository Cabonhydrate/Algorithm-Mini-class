import numpy as np

# ===================== 1. 基础工具函数（激活/损失/导数） =====================
def sigmoid(z):
    """Sigmoid激活函数，输出层二分类用"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # 防止数值溢出

def sigmoid_derivative(a):
    """Sigmoid导数，用激活输出a计算（避免重复算z）"""
    return a * (1 - a)

def relu(z):
    """ReLU激活函数，隐藏层首选（解决梯度消失）"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU导数，输入是z（因为a=relu(z)，导数由z决定）"""
    return (z > 0).astype(np.float32)

def softmax(z):
    """Softmax激活函数，输出层多分类用（医疗影像多病灶分类）"""
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # 数值稳定性：减去每行最大值
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def mse_loss(y_pred, y_true):
    """均方误差损失，回归任务（如影像像素值预测）"""
    return np.mean((y_pred - y_true) ** 2) / 2

def mse_loss_derivative(y_pred, y_true):
    """MSE损失对输出层激活的导数"""
    return y_pred - y_true

def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失，分类任务（结合Softmax使用）"""
    epsilon = 1e-8  # 防止log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon))

def cross_entropy_softmax_derivative(z_out, y_true):
    """Softmax+交叉熵的联合导数（简化计算，避免单独算Softmax导数）"""
    y_pred = softmax(z_out)
    return y_pred - y_true

# ===================== 2. 通用神经网络类（核心：反向传播） =====================
class NeuralNetwork:
    def __init__(self, layer_dims, activation_hidden="relu", activation_output="sigmoid", loss_type="mse"):
        """
        初始化神经网络
        :param layer_dims: 列表，每层的神经元数，如[784, 128, 64, 10]（MNIST）/ [影像展平维度, 256, 128, 2]（二分类）
        :param activation_hidden: 隐藏层激活函数，可选"sigmoid"/"relu"
        :param activation_output: 输出层激活函数，可选"sigmoid"/"softmax"/"none"（回归）
        :param loss_type: 损失函数，可选"mse"/"cross_entropy"
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # 输入层不算
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.loss_type = loss_type
        self.params = self._init_params()  # 存储权重w和偏置b

    def _init_params(self):
        """参数初始化：Xavier（sigmoid/tanh）/He（ReLU），避免梯度消失"""
        params = {}
        for l in range(1, self.num_layers + 1):
            if self.activation_hidden == "relu":
                # He初始化：方差=2/输入层神经元数
                params[f"w{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
            else:
                # Xavier初始化：方差=1/输入层神经元数
                params[f"w{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(1 / self.layer_dims[l-1])
            params[f"b{l}"] = np.zeros((self.layer_dims[l], 1))  # 偏置初始化为0
        return params

    def _forward_single_layer(self, a_prev, w, b, activation):
        """单层层前向传播：计算z和a"""
        z = np.dot(w, a_prev) + b  # z = w·a_prev + b
        if activation == "sigmoid":
            a = sigmoid(z)
        elif activation == "relu":
            a = relu(z)
        elif activation == "softmax":
            a = softmax(z.T).T  # 适配维度：(神经元数, 样本数)
        elif activation == "none":
            a = z  # 回归任务输出层无激活
        else:
            raise ValueError(f"不支持的激活函数：{activation}")
        return z, a

    def forward(self, x):
        """完整前向传播：保存每层的z和a（反向传播需要）"""
        caches = {}  # 缓存：z1,a1,z2,a2,... 方便反向传播调用
        a_prev = x  # 初始输入为x
        for l in range(1, self.num_layers):
            # 隐藏层：用隐藏层激活函数
            w = self.params[f"w{l}"]
            b = self.params[f"b{l}"]
            z, a = self._forward_single_layer(a_prev, w, b, self.activation_hidden)
            caches[f"z{l}"] = z
            caches[f"a{l}"] = a
            a_prev = a
        # 输出层：用输出层激活函数
        l = self.num_layers
        w = self.params[f"w{l}"]
        b = self.params[f"b{l}"]
        z_out, a_out = self._forward_single_layer(a_prev, w, b, self.activation_output)
        caches[f"z{l}"] = z_out
        caches[f"a{l}"] = a_out
        return a_out, caches

    def _backward_single_layer(self, da, z_prev, a_prev, w, activation):
        """单层层反向传播：计算dw, db, da_prev（传给前一层）"""
        m = da.shape[1]  # 样本数
        # 1. 计算dz：da * 激活函数导数
        if activation == "sigmoid":
            dz = da * sigmoid_derivative(a_prev)
        elif activation == "relu":
            dz = da * relu_derivative(z_prev)
        else:
            raise ValueError(f"不支持的激活函数：{activation}")
        # 2. 计算dw, db（链式求导：dz * a_prev^T）
        dw = np.dot(dz, a_prev.T) / m  # 除以样本数，批量梯度下降
        db = np.sum(dz, axis=1, keepdims=True) / m
        # 3. 计算da_prev：w^T * dz（传给前一层的da）
        da_prev = np.dot(w.T, dz)
        return dw, db, da_prev

    def backward(self, x, y_true, caches):
        """完整反向传播：链式求导核心，从输出层反向计算所有梯度"""
        grads = {}  # 存储梯度：dw1, db1, dw2, db2,...
        m = x.shape[1]  # 样本数
        l = self.num_layers  # 输出层层号
        a_out = caches[f"a{l}"]
        z_out = caches[f"z{l}"]

        # 第一步：计算输出层的da（损失对输出层a的导数）
        if self.loss_type == "mse":
            da = mse_loss_derivative(a_out, y_true)
        elif self.loss_type == "cross_entropy":
            if self.activation_output == "softmax":
                # Softmax+交叉熵：直接用联合导数，避免单独算da
                da = cross_entropy_softmax_derivative(z_out.T, y_true.T).T
            else:
                da = -(y_true / (a_out + 1e-8))  # 普通交叉熵导数
        else:
            raise ValueError(f"不支持的损失函数：{self.loss_type}")

        # 第二步：反向遍历每层，计算梯度
        # 先算输出层梯度
        a_prev = caches[f"a{l-1}"] if l > 1 else x
        z_prev = caches[f"z{l}"]
        w = self.params[f"w{l}"]
        # 输出层激活函数特殊处理：无激活/Softmax时，dz直接用da
        if self.activation_output in ["softmax", "none"]:
            dz = da
            dw = np.dot(dz, a_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da_prev = np.dot(w.T, dz)
        else:
            dw, db, da_prev = self._backward_single_layer(da, z_prev, a_out, w, self.activation_output)
        grads[f"dw{l}"] = dw
        grads[f"db{l}"] = db

        # 再算隐藏层梯度（从倒数第二层往第一层遍历）
        for l in range(self.num_layers - 1, 0, -1):
            da = da_prev
            z_prev = caches[f"z{l}"]
            a_prev = caches[f"a{l-1}"] if l > 1 else x
            w = self.params[f"w{l}"]
            dw, db, da_prev = self._backward_single_layer(da, z_prev, caches[f"a{l}"], w, self.activation_hidden)
            grads[f"dw{l}"] = dw
            grads[f"db{l}"] = db

        return grads

    def update_params(self, grads, learning_rate):
        """梯度下降更新参数：w = w - lr*dw, b = b - lr*db"""
        for l in range(1, self.num_layers + 1):
            self.params[f"w{l}"] -= learning_rate * grads[f"dw{l}"]
            self.params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    def train(self, x, y_true, epochs=1000, learning_rate=0.01, batch_size=None, print_every=100):
        """
        训练网络：支持批量/小批量/随机梯度下降
        :param x: 输入数据，维度(输入维度, 样本数)
        :param y_true: 真实标签，维度(输出维度, 样本数)
        :param epochs: 迭代次数
        :param learning_rate: 学习率
        :param batch_size: 小批量大小，None则为批量梯度下降
        :param print_every: 每多少轮打印一次损失
        """
        m = x.shape[1]
        for epoch in range(epochs):
            # 小批量数据划分
            if batch_size is None:
                batch_x, batch_y = x, y_true
            else:
                idx = np.random.choice(m, batch_size, replace=False)
                batch_x, batch_y = x[:, idx], y_true[:, idx]

            # 前向传播
            y_pred, caches = self.forward(batch_x)
            # 计算损失
            if self.loss_type == "mse":
                loss = mse_loss(y_pred, batch_y)
            else:
                loss = cross_entropy_loss(y_pred, batch_y)
            # 反向传播
            grads = self.backward(batch_x, batch_y, caches)
            # 更新参数
            self.update_params(grads, learning_rate)

            # 打印训练信息
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, x):
        """预测：前向传播得到结果"""
        y_pred, _ = self.forward(x)
        if self.activation_output == "softmax":
            # 多分类：返回概率最大的类别索引
            return np.argmax(y_pred, axis=0)
        elif self.activation_output == "sigmoid":
            # 二分类：返回0/1（阈值0.5）
            return (y_pred > 0.5).astype(np.int32)
        else:
            # 回归：直接返回预测值
            return y_pred