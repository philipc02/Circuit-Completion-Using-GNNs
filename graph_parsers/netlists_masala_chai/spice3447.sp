plaintext
* SPICE Netlist for Simple Inverter Circuit

V1 4 0 DC 5V      * Voltage source from node 4 to ground, 5V

* PMOS Transistor
M1 4 2 5 5 PMOS   * Drain=4, Gate=2, Source=5, Body=5

* NMOS Transistor
M2 5 2 3 3 NMOS   * Drain=5, Gate=2, Source=3, Body=3

* Node assignments:
* 4 - Supply Voltage (+5V)
* 2 - VIN
* 5 - VOUT
* 3 - Ground

.model PMOS PMOS
.model NMOS NMOS
.end