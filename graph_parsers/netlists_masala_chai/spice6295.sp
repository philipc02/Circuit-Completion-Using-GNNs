plaintext
* SPICE Netlist

V1 1 0 DC Vin   * Input voltage source (Vin)
VDD 5 0 DC Vdd  * VDD voltage source

* Resistor
R1 5 3 R        * Resistor from VDD to the NMOS drain

* NMOS Transistor
M1 3 1 0 0 NMOS * NMOS: Drain=3, Gate=1, Source=0, Body=0

* Capacitor
C1 3 0 C        * Capacitor connected from drain to ground

* The nodes marked correspond with the annotated schematic.