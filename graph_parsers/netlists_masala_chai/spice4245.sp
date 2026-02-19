plaintext
* SPICE Netlist for the given circuit

* Voltage sources
VDD 5 0 DC 5V
VSS 4 0 DC -5V
VI 3 0 DC ??? * Input voltage (not specified)

* Components
RD 5 2 10k  * Resistor, adjust value as needed

* Transistors
* NMOS (Q1)
M1 2 3 4 4 NMOS

* PMOS (Q2)
M2 2 2 4 4 PMOS

* Current Source
I0 2 0 DC ??? * Current source value (not specified)

* End of Netlist