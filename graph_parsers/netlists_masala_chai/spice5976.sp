spice
*MOSFET Model
.model NMOS NMOS(Level=1)

* NMOS Transistor
M1 2 6 4 4 NMOS W=W L=4L

* Current Source
I1 8 2 DC I

* Analysis Commands
* .dc VGS 0 5 0.1
* .ac dec 100 1 1MEG
* .tran 0.1ns 100ns