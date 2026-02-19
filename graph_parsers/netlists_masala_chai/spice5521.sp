spice
* SPICE netlist for the given circuit

* Op-Amps
.subckt opamp in+ in- out
* Simple op-amp model here could be just a voltage-controlled voltage source
E1 out 0 in+ in- 1Meg
.ends opamp

* Resistors
R1 5 2 10k
R2 2 6 33k
R3 6 0 10k

* Capacitor
C1 2 3 0.01u

* Voltage source connecting capacitor
Vcap 3 2 DC 0

* Op-amp instances
XU1 5 6 2 opamp
XU2 2 0 7 opamp

* Output load if any can be modeled here

* End of Netlist
.end