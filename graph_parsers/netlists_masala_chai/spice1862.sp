plaintext
* NMOS Transistor M1: Drain is connected to Vout (net 3), Gate to Vin, Source and Body to net 2.
M1 3 Vin 2 2 NMOS

* Current Source: Positive terminal connected to net 3, negative terminal to ground.
I1 3 0 IDC

* Resistor R1: Connected between net 3 and net 2.
R1 3 2 R_value

* Resistor R2: Connected between net 2 and ground.
R2 2 0 R_value

* Voltage input (for test purposes, assuming a DC sweep)
Vin Vin 0 DC 0V

* Ground Definition
V0 0 0 DC 0V

* .MODEL NMOS (Define NMOS model here)
.model NMOS NMOS (level=1 VT0=0.7 KP=50u GAMMA=0.5 PHI=0.65)

* Analysis, control, etc.
.control
DC Vin 0 5 0.1
.endc

.end