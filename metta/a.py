from math import inf



for i in range(int(input())):
    n = (input())
    ans = inf
    zero = 0 
    for i in range(len(n)): 
        zero+= (n[i]=='0')
        if n[i] != '0':
            ans = min(len(n)-1 -zero ,ans) 

    print(ans) 