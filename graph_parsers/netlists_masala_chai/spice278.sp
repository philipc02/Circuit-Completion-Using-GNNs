spice
* SPICE Netlist for the given circuit

* Voltage Definitions
VDD 8 0 DC 5V
VSS 5 0 DC -5V

* Current Source
I1 3 0 DC 1mA

* Current Source IB
IB 2 6 DC 1mA

* Transistors
* M1 (PMOS): Drain (8), Gate (2), Source (7)
M1 7 2 8 PMOS L=1u W=2u

* M2 (NMOS): Drain (7), Gate (3), Source (5)
M2 2 3 5 NMOS L=1u W=2u

* End of Netlist