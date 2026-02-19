spice
* Define the supply voltage
VDD VDD 0 DC 1.8

* Transistors
* Format: M<name> <drain> <gate> <source> <body> <model>
M1 net2 Vin 0 0 NMOS
M2 VDD Vin net2 net2 PMOS
M3 net3 net2 0 0 NMOS
M4 net2 net2 VDD VDD PMOS

* Capacitor
* Format: C<name> <positive> <negative> <capacitance value>
CL Vout 0 <capacitance_value>

* Input and output
Vin Vin 0 DC <input_voltage>
Rload Vout 0 <load_resistance_value>

* Models
.model NMOS NMOS (level=1)
.model PMOS PMOS (level=1)

* Analysis commands
.TRAN 1n 10n
.END