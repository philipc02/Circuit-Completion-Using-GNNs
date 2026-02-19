spice
* SPICE Netlist
* Components
Vs 1 0 DC
Rs 1 2 100k
R2 2 3 100k
C1 3 0 0.01u

* Connections for op-amp assumed ideal
* Positive input connected to node 2
* Negative input connected between Rs and R2
* Output connected to Vo at node 2

* Analysis
.tran 1u 10m
.end