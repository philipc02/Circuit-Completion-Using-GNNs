spice
* Components List and Connections

* Current Sources
I1 2 0 DC IREF
I2 3 0 DC IREF

* Transistors
M1 4 5 2 2 PMOS W=X L=Y
M2 5 0 3 3 NMOS W=X L=Y
M3 2 2 3 3 NMOS W=X L=Y
M4 2 2 4 4 PMOS W=X L=Y
M5 2 2 0 0 PMOS W=(1/4)X L=Y

* Nodes:
* 2 - Common node at V+
* 3 - VBias
* 4 - VO_D1 (Output)