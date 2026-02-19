spice
* SPICE Netlist for the given circuit
* Nodes: 1=Vin, 2=Inverting Input, 3=Non-Inverting Input, 4=Ground, 5=Vout, 6=+15V, 7=-15V

V1 1 0 AC 10
R1 1 2 200k
R2 2 0 100k
C1 7 0 10uF

* Op-Amp model (Ideal)
* .model opamp op(DCGAIN=1MEG)
XOP 3 2 5 opamp

VCC 6 0 DC 15
VEE 7 0 DC -15

.control
run
.endc