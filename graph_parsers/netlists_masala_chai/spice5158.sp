* SPICE netlist for the given circuit

* Voltage Source
Vin 1 3 DC 0

* Resistors
R3 1 4 56k
R4 4 6 56k
R2 2 5 39k
R1 2 3 20k

* Capacitors
C1 4 3 220p
C2 6 2 220p

* Op-Amp
XOP 6 2 5 5 opamp

* Model for ideal op-amp
.model opamp opamp(Vp=+ Vn=-) 

* End of netlist