spice
* Netlist for the circuit

* PMOS Transistors
M1 Vout A Vdd Vdd PMOS
M4 D Vout Vdd Vdd PMOS

* NMOS Transistors
M2 B Vdd 0 NMOS
M3 C D 0 NMOS

* Voltage Sources and Ground Connections
* Vdd should be defined in your testbench or higher hierarchy