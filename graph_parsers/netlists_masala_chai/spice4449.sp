spice
* NMOS Transistor
M1 3 2 0 0 NMOS

* Resistors
RF 2 3 25k
RD 3 5 5k

* Capacitor
CC 2 4 0.1u

* Current Source
IS 4 0 DC 1mA

* Voltage Source
VDD 5 0 DC 3V

* Model Definitions
.model NMOS NMOS(Level=1)

* End of Netlist