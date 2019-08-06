# Abstract
# 1 Continuous Bag-of-Word Model
## 1.1 One-word context
&emsp;&emsp;在本文中我们首先从Mikolov等人的模型中最简单的版本CBOW(Continuous Bag-of-Word Model)开始介绍。在讨论CBOW模型的时候，我们假设只有一个词汇的情况,对于每一个输出词汇的预测只给定一个输入词汇，在这种情况下CBOW就像是一个二元模型。在下图1中我们主要展示简化场景下的网络结构，在图1中,相邻不同层神经元之间通过全连接方式相连，其中输入层的输入是一个One-Hot编码的向量。
<div align=center>

 ![简化版CBOW模型结构](https://github.com/xzyin/notes/blob/master/figure/recsys/word2vec_param/figure1.PNG?raw=true, "简化的CBOW模型结构")

 </div>

&emsp;&emsp;在图1的模型中输入和输出之间的权重可以表示为一个$V \times N$,在这个$V \times N$的矩阵$W$中每一列是一个$N$维的向量 $\mathit{V}_{\omega}$ 对应输入层每一个输入的词汇。我们将输入层的计算做形式化描述如下，对矩阵$W$中的每一列$i$表示为$V^{T}_{\omega}$ 对于一个给定的词汇的one-hot编码向量，假设这个向量索引为$k$的位置为值为$1$，其他的位置的值都为$0$,那么对应形式化描述如下:
$$V^{T}_{\omega}=\left\{
  \begin{aligned}
  x_{i} = 1 &  &  {i = k}\\
  x_{i’} = 0  &  & i \neq k &  & {i \in {1...V}}
  \end{aligned}
  \right.
$$
&emsp;&emsp;对于给定一个词汇,可以计算得到隐藏层输入如下:
$$h=W^TX = W^T_{k\cdot} := V^T_{\omega_I}$$
&emsp;&emsp;对于上述公式来说, $h$ 的值实际上就是的矩阵 $W$ 所以为 $k$ 的列的值, $V_{\omega_I}$ 是输入词汇 $\omega_I$的向量表示。

&emsp;&emsp;从隐藏层到输出层的表示,表示为一个 $N \times V$ 的矩阵 $W'= {\omega'_{ij}}$ 基于矩阵 $W'$ 和隐藏层输出可以计算每个词汇的得分.

$$ u_j = {V'_{\omega_j}}^T h$$

&emsp;&emsp;其中 $V'_{\omega_j}$ 是矩阵 $W'$ 的第 $j$ 行，在此基础上,我们采用softmax计算得到所有单词得分的后验分布，这个后验分布是一个多项分布具体公式如下:
$$
p(\omega_j|\omega_I) = y_i = \frac {exp(u_j)}{\sum^{V}_{j’=1}exp(u_{j'})}
$$
其中 $y_j$ 是神经网络中输出层每个神经元的输出，结合输入层->隐藏层的计算公式(1)和隐藏层->输出层的公式(2)得到神经网络输入和输出的映射函数如下:

$$
p(\omega_j|\omega_I) = \frac {exp({V'_{\omega_j}}^T V_{\omega_I})}{\sum^{V}_{j’=1}exp({V'_{\omega_j}}^T V_{\omega_I})}
$$

&emsp;&emsp;需要值得注意的是, $V_\omega$ 和 $V'_\omega$ 表示的是不同的向量其中向量  $V_\omega$ 来源于矩阵 $W$, 向量 $V'_\omega$ 来源于矩阵 $W'$,向量矩阵 $W$ 和 $W'$ 表示不同的矩阵，我们将 $V_\omega$ 称为输入向量,将$V’_\omega$ 称为输出向量。
### Update equation for hidden $\to$ output weight
&emsp;&emsp;接下来我们推断一下这个模型中权重更新的方式。虽然在这个模型上实际的计算有点不切实际，但是为了更好的理解模型我们还是先不采用任何trick的情况下对模型权重更新的方式进行推断。
&emsp;&emsp;在模型训练的过程中，我们的优化目标实际上是为了最大化公式(4)的结果，表示的是对于给定的输入 $\omega{I}$ 输出层神经元 $j^{*}$ 观察到的输出结果 $\omega_O$ 的条件概率。

$$
max\;p(\omega_{O} | \omega_{I}) = max\; y_{j^*} \\
=max\; log\;y_{j*} \\
=max\; log(\frac {exp({V'_{\omega_{j^*}}}^T V_{\omega_I})}{\sum^{V}_{j’=1}exp({V'_{\omega_{j^*}}}^T V_{\omega_I})}) \\
= log(exp({V'_{\omega_{j^*}}}^T V_{\omega_I})) - max\; log(\sum^{V}_{j’=1}exp({V'_{\omega_{j^*}}}^T V_{\omega_I})) \\
= {V'_{\omega_{j^*}}}^T V_{\omega_I} - max\; log(\sum^{V}_{j’=1}exp({V'_{\omega_j'}}^T V_{\omega_I})) \\
= u_{j*} - max\; log(\sum^{V}_{j'=1}exp(u_{j'})) := -E
$$
&emsp;&emsp;根据上述的推导可以得到损失函数如下为 $E=-log\;p(\omega_O|\omega_I)$ 并且 $j^*$ 是输出层真实输出的索引，值得注意的是损失函数可以理解为两个概率分布的交叉熵的一种特殊情况。

&emsp;&emsp;接下来我们推断一下在隐藏层到输出层的权重更新方程，根据损失函数 $E$ 的推导过程，第 $j$ 个神经元的输入为 $u_j$, 那么我们求损失函数 $E$ 在 $u_j$ 上的导数为。

$$
\frac {\partial E}{\partial u_j} = \frac {\partial(-u_{j*} +  log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} \\
= \frac{\partial u_{j*}} {\partial u_j} + \frac{\partial ( log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} \\
= \left\{
  \begin{aligned}
  - \frac{\partial u_{j}} {\partial u_j} + \frac{\partial ( log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} & & u_{j*} = u_j \\
  -\frac{\partial u_{j'}} {\partial u_j} + \frac{\partial ( log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} & & u_{j'} \neq u_j
  \end{aligned}
  \right. \\
  = \left\{
    \begin{aligned}
    -1 + \frac{\partial (log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} & & u_{j*} = u_j \\
    0 + \frac{\partial (log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} & & u_{j'} \neq u_j
    \end{aligned}
    \right. \\
$$
&emsp;&emsp;为了表示的方便，我们将上述分段描述统一描述为:
$$\frac {\partial E}{\partial u_j} = -t_j + \frac{\partial (log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j}$$
基于对数求导公式和指数函数求导公式:
$$\frac{\partial ln\;(x)}{\partial x} = \frac{1}{x}$$
$$\frac{\partial e^x}{\partial x} = e^x$$
推导得到:
$$\frac{\partial (log(\sum^{V}_{j'=1}exp(u_{j'})))} {\partial u_j} = \frac{1}{\sum^{V}_{j'=1}exp(u_{j'})} \times \frac{\partial (exp(u_{j}))}{\partial u_j} \\
=\frac{exp(u_j)}{\sum^{V}_{j'=1}exp(u_{j'})} \\
= y_j$$
&emsp;&emsp;根据上述推导公式得到
$$\frac {\partial E}{\partial u_j} = y_j-t_j :=e_j$$
其中当第$j$个神经元为真实的输出时 $t_j=1$ ,否则 $t_j=0$ 。需要值得值得注意的是 $u_j$ 是输出层的结果，上述的推导过程得到的公式只是损失函数对输出层输出的结果的倒数,需要隐藏层到输出层的权重更新方程,我们需要进一步引入 $u_j$ 从隐藏层到输出层的计算方式
$$ u_j = V'_{\omega_j}h $$
推断算是函数 $E$ 在 $W'$ 上的导数如下:
$$\frac{\partial E}{\partial \omega'_{ij}} = \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial \omega'_{ij}} = e_j \cdot h_i$$
&emsp;&emsp;因此使用梯度下降法,对于从隐藏层到输出层获得的权重更新函数如下所示:
$$\omega'^{(new)}_{ij} = \omega'^{(old)} - \eta \cdot e_j \cdot h_i$$
或者是:
$$V'^{(new)}_{\omega_j} = V'^{(old)}_{\omega_j} - \eta \cdot \ e_j \cdot h \;\;\;\;\; for \;\; i = 1,2,3,\cdots,V$$
&emsp;&emsp;在上述公式中, $\eta > 0$ 表示学习率, $e_j = y_j - t_j$,并且$h_i$表示隐藏层的第 $i$ 个神经元 $V'_{\omega_j}$ 是整个词汇 $\omega_j$ 的输出向量。值得注意的是更新方程意味着我们必须逐个计算词汇表里所有可能的词汇，并且确认这个词汇的输出概率 $y_j$ 并且将 $y_j$和期望的输出$t_j$（0或者1）做对比。如果$y_j > t_j$说明这个词汇的概率被高估，需要减掉隐藏层向量$h$的值，例如对于输入向量 $V'_{\omega_I}$ 输出向量对应于 $V'_{\omega_j}$,通过减掉一个适当减小 $V'_{\omega_j}$ 的值使得 $V'_{\omega_I}$ 离向量 $V'_{\omega_j}$ 更远,如果 $y_j < t_j$ (只有当 $t_j = 1$ 才会出现这种情况)那么意味着这个词汇很好的被理解，我们把向量 $h$ 添加到 $V'_{\omega_O}$ 上因此使得向量 $V'_{\omega_O}$ 更接近 $V'_{\omega_I}$, 如果当 $y_j$ 非常接近 t_j的时候根据权重更新方程，输出向量将会做极小的改变，需要注意的是在这里 $V_{\omega}$ 和 $V'_{\omega}$ 分别表示输入向量和输出向量是不同的向量。
### Update equation for input $\to$ hidden weight
&emsp;&emsp;得到了权重矩阵 $W'$ 的权重更新方程的之后,接下来我们来推导矩阵 $W$ 的权重更新方程求损失函数 $E$ 在 $h_i$ 上的导数，具体的计算公式如下:
$$\frac{\partial E}{\partial h_i} = \sum^{V}_{j=1} \frac{\partial E}{ \partial u_j} \cdot  \frac {\partial u_j}{\partial h_j} = \sum^{V}_{j=1} e_j \cdot \omega'_{ij} := EH_i$$
&emsp;&emsp;在上述公式中 $h_i$ 是隐藏层第 $i$ 个神经元的输出, $u_j$在公式(2)中被定义，是神经网络输出层的第 $j$ 个神经元的输入，其中 $e_j = y_j - t_j$ 表示输出层第 $j$ 个词汇的预测误差。$EH$ 是一个N维的向量是所有词汇的输出向量之和并且通过每个词汇的误差进行加权。

&emsp;&emsp;上述公式的具体推导过程如下
## 1.2 Multi-word context
# 2 Skip-Gram Model
# 3 Optimizing Computational Efficiency
## 3.1 Hierarchical Softmax
## 3.2 Negative Sampling
