#da是当前层的激活函数的梯度  a_prev是上一层的激活函数的输出，当前层的输入  w是当前层的权重  activation是当前层的激活函数
#目标输出：dw   db   da_prev
def _backward_single_layer(self, da, a_prev, w, z, activation, a_current=None, da_is_dz=False):
    
    #用m来防止批次对loss的影响
    m=a_prev.shape[1] #shape[0]是即行数是神经元个数，shape[1]即列数是样本个数
    
    if da_is_dz:
        dz = da
    else:
        #a=σ(z) -> dz/da=σ'(z) -> dz=da*σ'(z)
        dz = da * activation_derivative(activation, a=a_current, z=z)
    
    #z=wx+b (这里的x是a_prev) -> dw/dz=a_prev -> dw=dz@a_prev.T/m
    dw=(dz@a_prev.T)/m
    
    #z=wx+b (这里的x是a_prev) -> db/dz=1 (注意dz形状是n*m （n是当前层神经元个数，m是样本数），db是n*1，所以要进行聚合)
    db=np.sum(dz,axis=1,keepdims=True)/m
    
    #dz/da_prev=w -> da_prev=w.T@dz
    da_prev=w.T@dz
    return dw,db,da_prev


def backward(self, y_true, caches):
    grads={}#存各个层的w和b的梯度
    #对于最后一层
    layer_idx=self.num_layers#层号
    a_out=caches[f"a{layer_idx}"]
    z_out=caches[f"z{layer_idx}"]
    activation_out=self.activations[-1]
    
    y_true = y_true.reshape(a_out.shape)
    
    # ---- 最后一层：分类,直接得到 dZ ----
    if activation_out in ['sigmoid', 'softmax']:
        dz = (a_out - y_true)
        a_prev = caches[f"a{layer_idx-1}"]
        w = self.params[f"w{layer_idx}"]

        dw, db, da_prev = _backward_single_layer(
            dz, a_prev, w, z_out, activation_out,
            a_current=a_out,
            da_is_dz=True
        )
        grads[f"w{layer_idx}"] = dw
        grads[f"b{layer_idx}"] = db
        da = da_prev
        start = layer_idx - 1

    else:
        #回归：先算 dA，再走通用链式法则
        da = (a_out - y_true)
        start = layer_idx
        
    # ---- 其余层 ----
    for layer_idx in range(start,0,-1):
        a_prev=caches[f"a{layer_idx-1}"]
        a_cur=caches[f"a{layer_idx}"]
        z_cur=caches[f"z{layer_idx}"]
        w_cur=self.params[f"w{layer_idx}"]
        activation_current=self.activations[layer_idx-1]
        
        # 防止 softmax 被错误放在隐藏层
        if act_cur == 'softmax':
            raise ValueError("softmax 不建议/不支持放在隐藏层。请仅在最后一层使用 softmax+交叉熵，并用 dZ=a-y。")

        dw, db, da = _backward_single_layer(
            da, a_prev, w_cur, z_cur, activation_current,
            a_current=a_cur,
            da_is_dz=False
        )
        
        grads[f"w{layer_idx}"]=dw
        grads[f"b{layer_idx}"]=db
        
    return grads


def update_params(self,grads,learning_rate):
    for layer_idx in range(1,self.num_layers+1):
        self.params[f"w{layer_idx}"]-=learning_rate*grads[f"w{layer_idx}"]
        self.params[f"b{layer_idx}"]-=learning_rate*grads[f"b{layer_idx}"]
        
    
    





#计算激活函数的导数
def activation_derivative(activation, a=None, z=None):
    if activation == 'sigmoid':
        if a is None:
            raise ValueError("sigmoid导数需要传入激活值a")
        return a * (1 - a)
    
    elif activation == 'relu':
        if z is None:
            raise ValueError("ReLU导数需要传入z值")
        return (z > 0).astype(np.float32)
    
    elif activation == 'tanh':
        if a is None:
            raise ValueError("tanh导数需要传入激活值a")
        return 1 - np.square(a)
    
    elif activation in ['none', 'linear']:
        # 恒等激活：a=z，导数恒为1
        if z is not None:
            return np.ones_like(z)
        if a is not None:
            return np.ones_like(a)
        raise ValueError("none/linear导数需要传入a或z")

    elif activation == 'softmax':
        # 注意：softmax的真实导数是Jacobian，通常只在“softmax+交叉熵”组合时
        # 直接用 dZ = (a_out - y_true)，因此这里不提供隐藏层softmax导数
        raise ValueError("softmax导数不支持通用形式；请仅在最后一层softmax+交叉熵时使用 dZ=a-y 的简化。")

    else:
        raise ValueError(f"不支持的激活函数：{activation}，可选：'sigmoid'/'relu'/'tanh'/'none'/'linear'")