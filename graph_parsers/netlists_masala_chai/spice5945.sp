spice
* SPICE Netlist

Vi 1 6 DC 0
R3 1 6 ro3
Gmvgs 4 6 3 1 gm
Ro1 4 5 ro1
Gmbvbs 5 6 4 2 gmb

* Nodes
* 1: S
* 2: D, B
* 3: G
* 4: Input of gmbvbs & ro1
* 5: Output of ro1 & gmbvbs
* 6: Ground

.END