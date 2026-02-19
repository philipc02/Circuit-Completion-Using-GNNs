plaintext
.SUBCKT op_amp 1 2 3
* + (non-inv), - (inv), Out
* Placeholder model for operational amplifier
EGAIN 3 0 VALUE = {V(1, 2) * 1e6}
RGAIN 3 0 1MEG
.ENDS

* Connections:
* X is node 2
* XF is node 2
* Ground is node 3

* Resistors
R1 2 2 10k
R2 2 3 10k

* Op-Amp (using a subcircuit)
X1 2 2 2 op_amp

* Input and Output
* V(X) = input, V(Y) = output
* Assume input is provided externally for simulation