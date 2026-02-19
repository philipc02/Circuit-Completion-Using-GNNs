plaintext
* SPICE Netlist for the Circuit

VDD 3 0 DC 5V

* Current Source
I1 3 3 DC

* PMOS Transistor M1
M1 3 6 3 3 PMOS

* NMOS Transistor M2
M2 5 2 4 4 NMOS

* Load Capacitor
Cload 5 0 <VALUE> ; Replace <VALUE> with the actual capacitance if needed

* End of Netlist
.end