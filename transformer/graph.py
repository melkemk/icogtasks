
# 
from collections import defaultdict, deque



graph = defaultdict(set)

graph['a'].add('b')  
graph['a'].add('c')
graph['b'].add('c')



for node, edges in graph.items():
    print(f"{node} -> {', '.join(edges)}")

ancestor = defaultdict(set)

queue = deque()  
queue.append('a')
while queue:
    node = queue.popleft()
    for child in graph[node]:
        ancestor[child].update(ancestor[node])  
        ancestor[child].add(node)  
        queue.append(child) 
print(ancestor)  
