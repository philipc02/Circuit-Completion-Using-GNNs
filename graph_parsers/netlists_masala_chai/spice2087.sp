spice
* SPICE Netlist

VGS net1 0 DC VALUE
VDD net3 0 DC VALUE
ID1 net1 net4 DC VALUE
ID2 net4 0 DC VALUE

RD1 net3 net4 10k

MN1 net4 net2 0 0 NMOS_MODEL
MP1 net3 net2 net4 net4 PMOS_MODEL

.control
run
.endc