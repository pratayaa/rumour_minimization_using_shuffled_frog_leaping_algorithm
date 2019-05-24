# -*- coding: utf-8 -*-
"""
Created on Wed May 22 00:07:50 2019

@author: Prataya,Sparsh,Piyush
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas
import statistics
import math
import gc

class Node(object):
    def __init__(self, nid = None, out_nodes = None, activity = None, in_nodes = None):
        self.nid = nid
        self.out_nodes = out_nodes
        self.activity = activity
        self.in_nodes = in_nodes
    in_edges = []
    out_edges = []
    a_i_edge_c = 0
    a_o_edge_c = 0
    t_i_edge_c = 0
    t_o_edge_c = 0
    p0 = 0
    p_send = 0
    p_acc = 0
    a_out_n = []
    i_out_n =[]
    a_out_e = []
    i_out_e = []
    fitness = 0
    def node_fitness(self):
        x=(self.a_i_edge_c*1.0/self.t_i_edge_c)*(self.a_o_edge_c*1.0/self.t_o_edge_c)
        self.fitness = x
        return x
    def block_n(self):
        self.activity = -1
        self.i_out_n = self.out_nodes
        self.a_i_edge_c = 0
        self.a_o_edge_c = 0
        self.a_out_n = []
        self.a_out_e = []
        self.i_out_e = self.out_edges
        
    def psend(self, p_send = None):
        self.p_send = p_send


class Edge(object):
    def __init__(self, eid = Node, from_node = None, to_node = None):
        self.eid = eid
        self.from_node = from_node
        self.to_node = to_node
    activity=0
    puv = 0
    
f=open("test_1k.txt",'r')

fromnode=[]
tonode=[[]]
active=[]
inedge=[[]]
firsttime=1
i=0
j=1
k=1
for line in f:
    if line[0]=='#':
        continue
    line=line.split()
    if firsttime==1:
        firsttime=0
        fromnode.append(int(line[0]))
        tonode[i].append(int(line[1]))
        inedge[i].append(int(line[0]))
        active.append(0)
        continue
    if fromnode[len(fromnode)-1]==int(line[0]):
        tonode[i].append(int(line[1]))
        
    else:
        i+=1
        fromnode.append(int(line[0]))
        tonode.append([int(line[1])])
        inedge.append([int(line[0])])
        active.append(0)
   
f.close()

memeplex = [[]]

for i in range(0,1000):
        memeplex.append([inedge[i][0]])
        
oute=[[]]
for i in range(0,1000):
        oute.append([inedge[i][0]])
ine=[[]]
for i in range(0,1000):
        ine.append([inedge[i][0]])

memeplex.pop(0)
oute.pop(0)
ine.pop(0)

for x in range(0,1000):
    memeplex[x].pop(0)
    
for x in range(0,1000):
    oute[x].pop(0)

for x in range(0,1000):
    ine[x].pop(0)

for x in range(0,1000):
    inedge[x].pop(0)


f=open("test_1k.txt",'r')

for line in f:
    if line[0]=='#':
        continue
    line=line.split()
    j=int(line[1])
    inedge[j].append(int(line[0]))

f.close()


node = []
for i in range(0,1000):
    node.append(Node(fromnode[i], tonode[i], active[i], inedge[i]))
    
f=open("test_1k.txt",'r')

e=0
edge = []

for line in f:
    if line[0]=='#':
        continue
    line=line.split()
    edge.append(Edge(e,int(line[0]),int(line[1])))
    e+=1

f.close()

ie=0
oe=0


for q in range(0,10000):
    oe = edge[q].from_node
    ie = edge[q].to_node
    oute[oe].append(edge[q].eid)
    ine[ie].append(edge[q].eid)


for q in range(0,1000):
    node[q].out_edges=oute[q]
    node[q].in_edges=ine[q]
    node[q].t_o_edge_c=len(oute[q])
    node[q].t_i_edge_c=len(ine[q])

for i in range(0,1000):
    node[i].p0=random.random()   
    
for i in range(0,1000):
    node[i].i_out_n=node[i].out_nodes
    node[i].i_out_e=node[i].out_edges

print("Enter the most influencing node: ")
inode=int(input())
node[inode].activity = 1

t=5
n1 = -1.0
n2 = 0
active=[]
active.append(inode)

for z in range(0,t):
    for i in range(0,len(active)):
        prsend=node[active[i]].p0/math.log(10+z)
        ax=active[i]
        node[ax].psend(prsend)
        
        for j in range(0,len(node[active[i]].i_out_n)):
            qw=node[active[i]].i_out_n[j]
            print("out node of 5= ", node[5].i_out_n)
            if node[qw].activity==1:
                continue
            x=node[active[i]].i_out_n[j]
            node[x].p_acc=1.0/node[x].t_i_edge_c
            oedge=node[active[i]].i_out_e[j]
            edge[oedge].puv = node[active[i]].p_send * node[x].p_acc
            if edge[oedge].puv > n1:
                n1 = edge[oedge].puv
                n2 = j
        en2=node[active[i]].i_out_e[n2]
        edge[en2].activity = 1
        tn = edge[en2].to_node 
        node[tn].activity = 1
        node[tn].a_i_edge_c += 1
        node[active[i]].a_o_edge_c +=1 
        active.append(node[tn].nid)
        z1=node[active[i]].i_out_n[n2]
        z2=node[active[i]].i_out_e[n2]
        node[active[i]].a_out_n.append(z1)
        node[active[i]].i_out_n.pop(n2)
        node[active[i]].a_out_e.append(z2)
        node[active[i]].i_out_e.pop(n2)
        n1=-1.0
        n2=0
        
print("active node in t=5 : ",active)   



###############################################################################
####################                 SFLA                  ####################
###############################################################################    
    
for i in range(0,len(memeplex)):
    for j in range(0,len(node[i].in_nodes)):
        memeplex[i].append(node[i].in_nodes[j])
for i in range(0,len(memeplex)):
    for j in range(0,len(node[i].out_nodes)):
        memeplex[i].append(node[i].out_nodes[j])

population = len(fromnode)
memeplex_fitness = [[]]
for i in range(1000):
        memeplex_fitness.append([memeplex[i][0]])
        
memeplex_fitness.pop(0)
        
for x in range(0,1000):
    memeplex_fitness[x].pop(0)
    

# calculating the fitness value
def fitness_fn(fitn):
    fit_val=node[fitn].node_fitness()
    return fit_val     
              
    
#def run():
#assigning values to memeplexes
#for i in range(len(memeplex)):
#    for j in range(len(node[i].in_nodes)):
#        memeplex[i].append(node[i].in_nodes[j])
#for i in range(len(memeplex)):
#    for j in range(len(node[i].out_nodes)):
#        memeplex[i].append(node[i].out_nodes[j])

#generating memeplex_fitness matrix            
#memeplex_fitness = [[] for i in range(population)]
#for i in range(len(memeplex)):
#    memeplex_fitness[i]=memeplex[i]

#calculating and storing fitness values
ft=1
t=2
for z in range(0,t):
    if ft==1:
        ft=0
        for i in range(0,len(memeplex)):
            for j in range(0,len(memeplex[i])):
                x=memeplex[i][j]
                memeplex_fitness[i].append(fitness_fn(x))
    
    for i in range(0,len(memeplex)):
        for j in range(0,len(memeplex[i])):
            x=memeplex[i][j]
            memeplex_fitness[i][j]=fitness_fn(x)
            
    #sorting of memeplex_fitness matrix
    for i in range(0,len(memeplex_fitness)):
        for j in range(0,len(memeplex_fitness[i])):
            for k in range(j+1,len(memeplex_fitness[i])):
                if memeplex_fitness[i][j] < memeplex_fitness[i][k]:
                    temp = memeplex_fitness[i][j]
                    memeplex_fitness[i][j] = memeplex_fitness[i][k]
                    memeplex_fitness[i][k] = temp
                    temp2 = memeplex[i][j]
                    memeplex[i][j] = memeplex[i][k]
                    memeplex[i][k] = temp2
    
    for i in range(0,len(memeplex)):
        if memeplex_fitness[i][0]!=0.0:
            blockn=memeplex[i][0]
            node[blockn].block_n()
            for j in range(0,len(node[blockn].out_edges)):
                ez1=node[blockn].out_edges[j]
                edge[ez1].activity=0
            for j in range(0,len(node[blockn].in_edges)):
                ez2=node[blockn].in_edges[j]
                edge[ez2].activity=0
        
    node_block = []
    j=0
    
    for i in range(0,1000):
        if node[i].activity==-1:
            node_block.append(node[i].nid)
        for i in range(0, len(node_block)):
            while j<len(active):
                print(node_block[i])
                print(active[j])
                if node_block[i]==active[j]:
                    active.pop(j)
                j+=1
    print("blocked node in t=",5+t," are : ",node_block)
    
    for i in range(0,len(active)):
        prsend=node[active[i]].p0/math.log(10+6)
        ax=active[i]
        node[ax].psend(prsend)
        for j in range(0,len(node[active[i]].i_out_n)):
            qw=node[active[i]].i_out_n[j]
            if node[qw].activity==1:
                continue
            x=node[active[i]].i_out_n[j]
            node[x].p_acc=1.0/node[x].t_i_edge_c
            oedge=node[active[i]].i_out_e[j]
            edge[oedge].puv = node[active[i]].p_send * node[x].p_acc
            if edge[oedge].puv > n1:
                n1 = edge[oedge].puv
                n2 = j
        en2=node[active[i]].i_out_e[n2]
        edge[en2].activity = 1
        tn = edge[en2].to_node 
        node[tn].activity = 1
        node[tn].a_i_edge_c += 1
        node[active[i]].a_o_edge_c +=1 
        active.append(node[tn].nid)
        z1=node[active[i]].i_out_n[n2]
        z2=node[active[i]].i_out_e[n2]
        node[active[i]].a_out_n.append(z1)
        node[active[i]].i_out_n.pop(n2)
        node[active[i]].a_out_e.append(z2)
        node[active[i]].i_out_e.pop(n2)
        n1=-1.0
        n2=0
    
    print("active node after t=",5+t," are : ",active)
                
                         


for i in range(0,len(memeplex)):
    for j in range(0,len(memeplex[i])):
        x=memeplex[i][j]
        memeplex_fitness[i][j]=fitness_fn(x)
        
#sorting of memeplex_fitness matrix
for i in range(0,len(memeplex_fitness)):
    for j in range(0,len(memeplex_fitness[i])):
        for k in range(j+1,len(memeplex_fitness[i])):
            if memeplex_fitness[i][j] < memeplex_fitness[i][k]:
                temp = memeplex_fitness[i][j]
                memeplex_fitness[i][j] = memeplex_fitness[i][k]
                memeplex_fitness[i][k] = temp
                temp2 = memeplex[i][j]
                memeplex[i][j] = memeplex[i][k]
                memeplex[i][k] = temp2

for i in range(0,len(memeplex)):
    if memeplex_fitness[i][0]!=0.0:
        blockn=memeplex[i][0]
        node[blockn].block_n()
        for j in range(0,len(node[blockn].out_edges)):
            ez1=node[blockn].out_edges[j]
            edge[ez1].activity=0
        for j in range(0,len(node[blockn].in_edges)):
            ez2=node[blockn].in_edges[j]
            edge[ez2].activity=0
print("blocked node in end are : ",node_block)
print("remaining active nodes are : ",active)
#if __name__ == "__main__":
#    run()
