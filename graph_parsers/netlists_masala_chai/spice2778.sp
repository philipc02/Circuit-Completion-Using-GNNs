* NMOS Transistor
M2 Y Vb X X NMOS

* Resistor
rO2 Y Vout r_value

* Capacitors
CX X 0 c_value_X
CY Vout 0 c_value_Y

* Current Source
Iin X 0 dc current_value

* Node Voltage Definitions
Vb Vb 0 Vb_value

* End of Netlist
.tran 1n 10u
.end