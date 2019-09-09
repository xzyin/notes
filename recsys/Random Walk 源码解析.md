# 1. Random Walk 执行流程
## 1.1 加载数据
参数列表
变量名称 | | |
---|---|---
degree | |
directed | |
indexed | |
maxDegree | |
p | |
q | |

## 1.2 基础结构

&emsp;&emsp;通过NodeAttr定义了一个基本的Node结构,这个Node结构里面存放了该节点的所有邻居节点以及从该节点作为源头的路径。
```java
case class NodeAttr(var neighbors: Array[(Long, Double)] = Array.empty[(Long, Double)],
                      var path: Array[Long] = Array.empty[Long]) extends Serializable
```

&emsp;&emsp; EdgeAttr 存储了所有边的属性
```java
case class EdgeAttr(var dstNeighbors: Array[Long] = Array.empty[Long],
                      var J: Array[Int] = Array.empty[Int],
                      var q: Array[Double] = Array.empty[Double]) extends Serializable
```

## 1.2 初始化游走概率
## 1.3 随机游走
## 1.4 保存游走路径
# 2. Random Walk
