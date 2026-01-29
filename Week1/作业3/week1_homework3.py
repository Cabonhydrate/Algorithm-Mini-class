import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os


class Utils(object):
    @staticmethod
    #数字之间有天然的大小，所以要用one_hot
    def one_hot_encode(x,output_dim):
        return np.eye(output_dim)[x]
    
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x>0,1,0)
    
    @staticmethod
    def softmax(x):
        x-=np.max(x,axis=1,keepdims=True)
        exp_x=np.exp(x)
        sm=exp_x/np.sum(exp_x,axis=1,keepdims=True)
        return sm
    
    @staticmethod
    def cross_entropy_loss(y_pred,y_true):
        return np.mean(-np.sum(y_true*np.log(y_pred+1e-10),axis=1))
    


class TwoLayerNetwork(object):
    def __init__(self,input_dim=784,hidden_dim=128,output_dim=10,seed=42,lr=0.01,batch_size=64,epochs=50):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.seed=seed
        self.lr=lr
        self.batch_size=batch_size
        self.epochs=epochs
        
        
        
        rgn=np.random.default_rng(self.seed)
        
        self.w1=rgn.normal(0,np.sqrt(2/self.input_dim),size=(self.input_dim,self.hidden_dim)).astype(np.float32)
        self.b1=np.zeros((1,self.hidden_dim))
        
        self.w2=rgn.normal(0,np.sqrt(2/self.hidden_dim),size=(self.hidden_dim,self.output_dim)).astype(np.float32)
        self.b2=np.zeros((1,self.output_dim))
        
        
    def load_mnist_data(self):
        mnist=fetch_openml(
            'mnist_784',
            version=1,
            parser='auto',
            as_frame=False,
            data_home=r"E:\School\大四\算法课\作业\Week1"
        )
        
        x=mnist.data.astype(np.float32)/255.0
        y=mnist.target.astype(np.int64)

        rgn=np.random.default_rng(self.seed)
        idx=rgn.permutation(x.shape[0])
        
        train_idx=idx[:60000]
        test_idx=idx[60000:]
        
        x_train=x[train_idx]
        x_test=x[test_idx]
        
        y_train=y[train_idx]
        y_test=y[test_idx]

        y_train_onehot=Utils.one_hot_encode(y_train,self.output_dim)
        y_test_onehot=Utils.one_hot_encode(y_test,self.output_dim)
        
        return x_train,x_test,y_train_onehot,y_test_onehot
    
    
    def forward(self,x):
        # x -> z1=w1x+b1 -> a1=relu(z1) -> z2=w2a+b2 -> a2=softmax(z2)
        z1=x@self.w1+self.b1
        a1=Utils.relu(z1)
        
        z2=a1@self.w2+self.b2
        a2=Utils.softmax(z2)
        
        return z1,a1,z2,a2
    
    
    def backward(self,x,y_true,z1,a1,z2,a2):
        """反向传播：计算参数梯度"""
        batch_size=self.batch_size
        
        # 输出层梯度（softmax+交叉熵的简化公式）
        dz2=a2 - y_true
        dw2=a1.T@dz2 / batch_size
        db2=np.sum(dz2,axis=0,keepdims=True) / batch_size
        
        # 隐藏层梯度
        da1=dz2@self.w2.T
        dz1=da1 * Utils.relu_derivative(z1)
        dw1=x.T@dz1 / batch_size
        db1=np.sum(dz1,axis=0,keepdims=True) / batch_size
        
        return dw1,db1,dw2,db2
    
    
    def update_param(self,dw1,db1,dw2,db2):
        self.w1-=dw1*self.lr
        self.b1-=db1*self.lr
        
        self.w2-=dw2*self.lr
        self.b2-=db2*self.lr
        
        
    def predict(self,x):
        #这里的a2是所有数字及对应的概率
        z1,a1,z2,a2=self.forward(x)
        #返回概率最大的数字的one_hot编码
        return Utils.one_hot_encode(np.argmax(a2,axis=1),self.output_dim)
    
    
    def accuracy(self,x,y_true_onehot):
        y_pred=self.predict(x)
        return np.mean(np.argmax(y_pred,axis=1) == np.argmax(y_true_onehot,axis=1))
    
    
    def train(self,x_train,x_test,y_train_onehot,y_test_onehot):
        num_sample=x_train.shape[0]
        num_batch=num_sample//self.batch_size
        
        history={"train_loss": [],"train_acc": [],"test_acc": []}
        
        for epoch in range(1,self.epochs+1):
            idx=np.random.default_rng(self.seed + epoch).permutation(num_sample)
            x_train=x_train[idx]
            y_train_onehot=y_train_onehot[idx]

            epoch_loss=0.0
            for batch in range(num_batch):
                start=batch*self.batch_size
                end=start+self.batch_size
                
                x_batch=x_train[start:end]
                z1,a1,z2,a2=self.forward(x_batch)
                dw1,db1,dw2,db2=self.backward(x_batch,y_train_onehot[start:end],z1,a1,z2,a2)
                self.update_param(dw1,db1,dw2,db2)
                
            
            epoch_loss/=num_batch
            
            #少跑一点，快一点
            sample=5000
            train_acc=self.accuracy(x_train[:sample],y_train_onehot[:sample])
            test_acc=self.accuracy(x_test,y_test_onehot)
            _,_,_,a2_train=self.forward(x_train[:sample])
            train_loss=Utils.cross_entropy_loss(a2_train,y_train_onehot[:sample])
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            
            # 打印训练日志
            print(f"[Epoch {epoch:2d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}")
            
        return history
    
class Visualizer:
    @staticmethod
    def plot_history(history,save_dir="results"):
        os.makedirs(save_dir,exist_ok=True)
        epochs=np.arange(1,len(history["train_loss"]) + 1)

        #loss 曲线
        plt.figure()
        plt.plot(epochs,history["train_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        loss_path=os.path.join(save_dir,"training_loss.png")
        plt.savefig(loss_path,dpi=200,bbox_inches="tight")
        plt.close()

        #acc 曲线
        plt.figure()
        plt.plot(epochs,history["train_acc"],label="Train Acc")
        plt.plot(epochs,history["test_acc"],label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)
        acc_path=os.path.join(save_dir,"accuracy_curve.png")
        plt.savefig(acc_path,dpi=200,bbox_inches="tight")
        plt.close()

        print(f"[Saved] {loss_path}")
        print(f"[Saved] {acc_path}")

    @staticmethod
    def show_predictions(network,x_test,y_test_onehot,n=12,seed=0,save_dir="results"):
        os.makedirs(save_dir,exist_ok=True)

        rng=np.random.default_rng(seed)
        idx=rng.choice(x_test.shape[0],size=n,replace=False)

        x=x_test[idx]
        y_true=np.argmax(y_test_onehot[idx],axis=1)
        y_pred=np.argmax(network.predict(x),axis=1)

        plt.figure(figsize=(12,4))
        for i in range(n):
            plt.subplot(2,n//2,i+1)
            plt.imshow(x[i].reshape(28,28),cmap="gray")
            plt.title(f"P:{y_pred[i]} / T:{y_true[i]}")
            plt.axis("off")
        plt.tight_layout()

        pred_path=os.path.join(save_dir,f"predictions_seed{seed}.png")
        plt.savefig(pred_path,dpi=200,bbox_inches="tight")
        plt.close()

        print(f"[Saved] {pred_path}")






if __name__ == "__main__":
    network=TwoLayerNetwork(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        seed=42,
        lr=0.05,
        batch_size=64,
        epochs=50
    )

    x_train,x_test,y_train_onehot,y_test_onehot=network.load_mnist_data()
    history=network.train(x_train,x_test,y_train_onehot,y_test_onehot)

    Visualizer.plot_history(history,save_dir=r"作业3/results")
    Visualizer.show_predictions(network,x_test,y_test_onehot,n=12,seed=42,save_dir=r"作业3/results")
