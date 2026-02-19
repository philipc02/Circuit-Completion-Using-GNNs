plaintext
* SPICE Netlist

Vbe 8 9 DC 0
Ii 2 5 DC 0

Rf 8 4 20k
Rx 8 6 5k
Ro 2 7 1Meg
Rc 2 10 10k

Gm 5 9 6 2 40m

.dc Vbe -1 1 0.1
.print DC V(10,2) I(Rf) I(Rx) I(Ro) I(Rc)
.end