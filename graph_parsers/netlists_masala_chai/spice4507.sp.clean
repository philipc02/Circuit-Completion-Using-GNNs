spice
* SPICE Netlist

* Capacitors
C1 5 2 1.08C
C2 2 0 0.9241C
C3 5 4 2.613C
C4 4 0 0.3825C

* Resistors
R1 Vi 2 R
R2 2 3 R
R3 3 4 R

* Operational Amplifiers (ideal model)
* Op-amp 1
XU1 2 2 3 opamp

* Op-amp 2
XU2 4 4 5 opamp

* .MODEL statement for ideal op-amp
.model opamp VA=1E6 GBW=1E6

* Analysis
.TRAN 1n 1u
.END