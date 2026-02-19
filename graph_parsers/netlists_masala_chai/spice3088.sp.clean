plaintext
* SPICE Netlist

* Voltage Sources
VIN 5 8 DC <voltage_value_vin>
VDD 7 8 DC <voltage_value_vdd>

* Current Source
IISS 4 8 DC <current_value_iss>

* NMOS Transistors
M1 4 5 8 8 NMOS
M2 3 8 8 8 NMOS
M3 3 2 8 8 NMOS
M4 2 8 8 8 NMOS

* PMOS Transistors
M5 2 22 3 3 PMOS
M6 2 2 8 8 PMOS
M7 7 6 2 2 PMOS
M8 7 2 8 8 PMOS

* Model Definitions (Example)
.model NMOS NMOS (Level=1 Tox=10nU Vto=0.7 Kp=120u W=100u L=1u)
.model PMOS PMOS (Level=1 Tox=10nU Vto=-0.7 Kp=50u W=100u L=1u)

.END