plaintext
* Op-Amp Model (Assumed ideal, using voltage-controlled voltage source)
E1 3 0 6 3 1e6
* Voltage Source
Vs 6 0 DC <value_of_Vs>
* NMOS Transistor (Assumed basic model)
M1 5 3 3 3 NMOS
* Current Source
I1 5 0 DC <value_of_Io>
* Feedback Resistor
RF 4 0 <value_of_RF>
.MODEL NMOS NMOS (LEVEL=1)
.END