#1. Graph 基本操作
* mapVertices 操作
```java
def mapVertices[VD2:ClassTag](map:(VertexId, VD) => VD2)
```
* mapEdges 操作
```java
def mapEdges[ED2: ClassTag](map: Edge[ED] => ED2): Graph[VD, ED2]
```
* mapEdges 操作

#2. 基本数据结构
* EdgeTriplet
* Edge
* 
