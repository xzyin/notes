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
&emsp;&emsp;得到了权重矩阵 $W'$ 的权重更新方程的之后,接下来我们来推导矩阵 $W$ 的权重更新方程。我们推导隐层层输出的损失函数得到如下公式:
$$\frac{\partial E}{\partial h_i} = \sum^{V}_{j=1} \frac{\partial E}{ \partial u_j} \cdot  \frac {\partial u_j}{\partial h_j} = \sum^{V}_{j=1} e_j \cdot \omega'_{ij} := EH_i$$
&emsp;&emsp;在上述公式中 $h_i$ 是隐藏层第 $i$ 个神经元的输出, $u_j$在公式(2)中被定义，是神经网络输出层的第 $j$ 个神经元的输入，其中 $e_j = y_j - t_j$ 表示输出层第 $j$ 个词汇的预测误差。$EH$ 是一个N维的向量,这个向量是所有输出词汇的向量通过误差加权之和。

 &emsp;&emsp;从隐藏层到输出层,一个输入对应一个输出,但是从输入层到输出层在到隐藏层预测的是每个词汇与所有词汇的距离，那么对应的输出应该是 $V$ 个，在计算输入层到输出层的$loss$的时候需要把每一个输出神经元得到的 $loss$相加。

 &emsp;&emsp;在得到上述推导公式的基础上我们进一步推导 $E$ 在输入向量矩阵 $W$ 上的一阶导。在上述过程中提到，隐层层到输入层的过程实际上完成的是一个线性计算，具体计算公式如下:
 $$h_i = \sum^V_{k=1} {x_k \cdot \omega_{ki}}$$
 &emsp;&emsp;结合上述的两个推导公式得到:
 $$\frac {\partial E} {\partial \omega_{ki}} = \frac {\partial E}{\partial h_i} \cdot \frac {\partial h_i} {\partial \omega_{ki}} = EH_i \cdot x_k$$
我们将上述公式写成张量的形式如下:
$$\frac {\partial E} {\partial W} = X \otimes EH = XEH^T \$$
通过上述公式,我们能得到一个 $V \times N$ 的矩阵。因为 $X$采用的one-hot编码向量,也就是说对于 $\frac {\partial E} {\partial W}$ 来说只有一列是非0的,并且该列的值是$EH^T$,是一个$N$维向量。我们得到输入矩阵的更新方程如下:
$$V^{(new)}_{\omega_I} = V^{(old)}_{\omega_I} - \eta EH^T$$
其中 $V_{\omega_I}$ 是矩阵 $W$的一列，是输入的One-Hot向量在矩阵中对应的非零的那一列，这这次迭代中所有其他的向量依旧保持不变对应的结果值为0.
&emsp;&emsp;从直觉上因为向量EH是词汇表中所有输出向量基于预测误差 $e_j = y_j - t_j$ 加权求平均的结果，我们可以理解为将输出向量的某个部分添加到输入向量的上下文信息中。如果一个词汇 $\omega_j$ 在输出层的概率被高估了(也就是 $y_j > t_j$),那么输入词汇 $\omega_I$ 的向量会远离向量 $w_j$,如果词汇 $\omega_j$ 作为输出词汇的概率被低估了(也就是 $y_j < t_j$) 那么输入向量 $\omega_I$ 将会移动到离输出向量更近。当输出词汇 $\omega_j$ 的结果很正确的结果很相近的时候,那么对于输入$\omega_I$ 我们会得到非常小的输出。输入词汇向量 $\omega_I$ 的移动大小受到输出层所有词汇输出误差的影响，当输出层一个词汇的误差越大，这个词汇对输入向量的移动幅度的影响越大。

&emsp;&emsp;当通过基于语料库生成的上下文——词汇对迭代的去更新模型参数的时候会不断的累加向量的影响。我们可以想象到单词 $\omega$ 的输出向量被$\omega$ 的邻居词汇的输入向量来回拖动,就像词汇跟词汇之间有物理关联一样，同样的我们可以把输入向量也看成被多个输出向量来回拖动。这种解释可以提醒我们重力或者是力导向的图形布局，每个虚拟串的平衡长度与相关单词对的共现强度以及学习率有关。在经过多轮的迭代后,输入向量和输出向量的相对位置会变得基本稳定下来。

## 1.2 Multi-word context
&emsp;&emsp;在下图中,我们给出了在多个上下文词汇设置下的CBOW模型，当计算隐藏输出的时候,这个时候我们不再是直接输入词汇为1对应的那行向量拷贝过去。在CBOW模型中对输入的上下文词汇信息做了一个加权求平均,通过跟输入层到隐藏层的矩阵求内积最后得到隐层的输出结果如下:
$$h = \frac {1} {C} W^T (X_1 + X_2 + ... +X_C) \\
= \frac{1}{C} (V_{\omega_1} + V_{\omega_1} + ... + V_{\omega_C})$$
其中 $C$ 是输出到上下文关系中的词汇个数, $\omega_1,...,\omega_C$ 是在上下文场景中的词汇个数并且 $V_\omega$ 是输入词汇 $\omega$ 的输入向量,那么最终得到的损失函数如下:
$$E = -log\;p(\omega_O |\omega_{I,1},\cdots,\omega_{I,C}) \\
 = -u_{j^*} + log\sum^{V}_{j'=1} exp(u_{j'}) \\
 = -{V'_{\omega_O}}^T \cdot h + log \sum^{V}_{j'=1}exp({V'_{\omega_j}}^T \cdot h)$$
 根据上述公式可以发现,从隐藏层到输出层公式形式基本与原来一致,只是原来作为输入的向量 $h$ 表示的是单个词汇向量，现在表示的是多个向量的weight-mean结果，CBOW的模型结构图Figure 2所示。
 &emsp;&emsp;根据上述的模型结构可知，从输出层到隐藏层的模型结构没有发生任何变化，与one-word-context的场景类似，因此权重更新方程一直公式为:
 $${V'_{\omega_j}}^{(new)} = {V'_{\omega_j}}^{(old)}-\eta \cdot e_j \cdot h \;\;\;\; for \;\;\;j = 1,2,3,...,V$$

 值得注意的是对于每一个训练实例我们都需要使用使用上述方程更新隐藏层权重矩阵的每一个元素。
 &emsp;&emsp;从输入层到输出层的权重更新方程也跟one-context-word的场景类似，我们只需要在公式前乘以 $\frac{1}{C}$ 就可以了，具体的计算公式如下:
 $${V'_{\omega_{I,c}}}^{(new)} = {V'_{\omega_{I,c}}}^{(old)} - \frac{1}{C} \cdot \eta \cdot  EH^T \;\;\;\; for \;\;j = 1,2,3,...,C.$$
 其中 $V'_{\omega_{I,c}}$ 是输入场景下第 $c$ 词汇的输入向量其中 $\eta$ 学习速率，在上述公式中已经给出了 $EH$ 的表示，在one-context-word的场景下给出了这个方程更直观的理解。
# 2 Skip-Gram Model
&emsp;&emsp;Skip-Gram模型在Mikolov等人的论文中做了详细的介绍,在图3中展示了SkipGram展示了Skip-Gram模型的模型结构，在这个网络结构中，我们把预测的词汇放在输入层中，其上下文词汇放在输出层中。

&emsp;&emsp;在讨论Skip-Gram模型的时候我们继续用 $V_{\omega_I}$ 作为输入层的输入，因此我们能够得到和one-context-word场景下同样的输入，那意味着 $h$ 是输入层到隐藏层中间的权重矩阵 $W$ 的一个简单的copy，对于输入向量对应得到的 $h$ 的定义如下:
$$h = W^T_{k,\cdot} := V^T_{\omega_I}$$
&emsp;&emsp;在输出层我们得到的并不是一个多项分布而是多 $C$ 个多项分布，每次的计算采用相同的隐藏层到输出层矩阵:
$$p(\omega_{c,j} = \omega_{O,c} | \omega_I)= y_{c,j}= \frac{exp(u_{c,j})}{\sum^{V}_{j'=1}exp(u_{j'})}$$
&emsp;&emsp;其中 $\omega_{c,j}$ 是
# 3 Optimizing Computational Efficiency
## 3.1 Hierarchical Softmax
## 3.2 Negative Sampling
