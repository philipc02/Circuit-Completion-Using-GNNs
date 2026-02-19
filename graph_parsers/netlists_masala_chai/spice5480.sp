* SPICE Netlist for the given circuit

VCC 3 0 DC VCC
Vin 6 0 AC Vin

* Transistors
Q1 1 2 3 NPN
Q2 1 2 5 NPN

* Resistors
R1 3 2 R1_value
R2 2 0 R2_value
RL 2 4 RL_value

* Capacitors
C1 2 6 C1_value
C2 2 0 C2_value
C3 1 4 C3_value

* Diodes
D1 2 2 Diode_model
D2 2 5 Diode_model

* End of netlist